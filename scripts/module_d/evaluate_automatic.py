#!/usr/bin/env python3
"""Rule scoring and offline model-judge orchestration for Module D.

This module never calls a model API.  ``prepare`` creates deterministic,
model-blind judge requests and a separate secret key.  An external runner may
fill the ``judgment`` field in each request.  ``score`` then rejects incomplete
or modified requests, unblinds Base/LoRA, combines the two rule metrics and one
model score per persona layer, and reports LoRA - Base deltas.

All reusable functions depend only on the Python standard library and remain
compatible with Python 3.7.
"""

from __future__ import print_function

import argparse
import copy
import hashlib
import json
import math
import os
import re
import sys
from pathlib import Path


WORKSPACE = Path(__file__).resolve().parents[2]
if str(WORKSPACE) not in sys.path:
    sys.path.insert(0, str(WORKSPACE))

from scripts.module_d.build_review_sheet import (  # noqa: E402
    BlindReviewError,
    build_blind_review,
    file_sha256,
    load_comparisons,
    validate_generation_manifest,
)
from scripts.module_d.rubric import (  # noqa: E402
    GUARD_DIMENSIONS,
    PERSONA_LAYERS,
    RUBRIC_SCHEMA_VERSION,
    SCORE_DIMENSIONS,
    SCORE_RUBRICS,
    public_rubric_payload,
    rubric_sha256,
)


CONFIG_SCHEMA_VERSION = "module_d.three_layer_eval.v1"
JUDGE_REQUEST_SCHEMA_VERSION = "module_d.judge_request.v2"
JUDGE_KEY_SCHEMA_VERSION = "module_d.judge_key.v1"
SUMMARY_SCHEMA_VERSION = "module_d.automatic_summary.v1"
DEEPSEEK_RUNNER_SCHEMA_VERSION = "module_d.deepseek_judge_run.v1"
DEFAULT_CONFIG_PATH = WORKSPACE / "configs" / "module_d" / "hutao_three_layer_eval.json"

RULE_METRICS = {
    "surface_style": ("style_marker_control", "syntax_register_match"),
    "knowledge_relationship": ("factual_support", "relationship_constraints"),
    "value_worldview": ("principle_action_coverage", "conflict_safety"),
}
MODEL_METRIC = "model_layer_score"


class AutomaticEvaluationError(ValueError):
    """Raised when an automatic-evaluation artifact violates its contract."""


def canonical_json_sha256(value):
    payload = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def text_sha256(value):
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _is_number(value):
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        return False
    try:
        return math.isfinite(float(value))
    except (OverflowError, ValueError):
        return False


def _is_sha256(value):
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _require_nonempty_text(value, field):
    if not isinstance(value, str) or not value.strip():
        raise AutomaticEvaluationError("%s must be a non-empty string" % field)
    return value.strip()


def _require_string_list(value, field, allow_empty=False):
    if not isinstance(value, list) or (not value and not allow_empty):
        raise AutomaticEvaluationError("%s must be a non-empty string list" % field)
    if any(
        not isinstance(item, str) or not item.strip() or item != item.strip()
        for item in value
    ):
        raise AutomaticEvaluationError("%s must contain only non-empty strings" % field)
    if len(value) != len(set(value)):
        raise AutomaticEvaluationError("%s contains duplicate values" % field)
    return value


def _require_number(value, field, minimum=None, maximum=None, integer=False):
    valid = (
        isinstance(value, int) and not isinstance(value, bool)
        if integer
        else _is_number(value)
    )
    if not valid:
        kind = "integer" if integer else "finite number"
        raise AutomaticEvaluationError("%s must be a %s" % (field, kind))
    if minimum is not None and value < minimum:
        raise AutomaticEvaluationError("%s must be >= %s" % (field, minimum))
    if maximum is not None and value > maximum:
        raise AutomaticEvaluationError("%s must be <= %s" % (field, maximum))
    return value


def _require_object(value, field):
    if not isinstance(value, dict):
        raise AutomaticEvaluationError("%s must be an object" % field)
    return value


def _validate_weights(config):
    layers = config.get("layers")
    if not isinstance(layers, dict) or set(layers) != set(PERSONA_LAYERS):
        raise AutomaticEvaluationError(
            "config layers must exactly match %r" % (PERSONA_LAYERS,)
        )
    layer_total = 0.0
    for layer in PERSONA_LAYERS:
        layer_config = layers[layer]
        if not isinstance(layer_config, dict):
            raise AutomaticEvaluationError("layer %s must be an object" % layer)
        weight = layer_config.get("weight")
        if not _is_number(weight) or weight < 0:
            raise AutomaticEvaluationError("layer %s has invalid weight" % layer)
        layer_total += float(weight)
        metrics = layer_config.get("metrics")
        expected = set(RULE_METRICS[layer] + (MODEL_METRIC,))
        if not isinstance(metrics, dict) or set(metrics) != expected:
            raise AutomaticEvaluationError(
                "%s metrics must exactly match %r" % (layer, sorted(expected))
            )
        metric_total = 0.0
        for metric_id in sorted(expected):
            metric = metrics[metric_id]
            if not isinstance(metric, dict):
                raise AutomaticEvaluationError(
                    "%s.%s must be an object" % (layer, metric_id)
                )
            metric_weight = metric.get("weight")
            if not _is_number(metric_weight) or metric_weight < 0:
                raise AutomaticEvaluationError(
                    "%s.%s has invalid weight" % (layer, metric_id)
                )
            metric_total += float(metric_weight)
        if abs(metric_total - 1.0) > 1e-9:
            raise AutomaticEvaluationError(
                "%s metric weights sum to %.12f, expected 1" % (layer, metric_total)
            )
    if abs(layer_total - 1.0) > 1e-9:
        raise AutomaticEvaluationError(
            "layer weights sum to %.12f, expected 1" % layer_total
        )


def validate_eval_config(config):
    """Validate and return a three-layer evaluation configuration."""
    if not isinstance(config, dict):
        raise AutomaticEvaluationError("evaluation config must be an object")
    if config.get("schema_version") != CONFIG_SCHEMA_VERSION:
        raise AutomaticEvaluationError("unsupported evaluation config schema")
    _require_nonempty_text(config.get("character"), "character")
    _validate_weights(config)

    scales = config.get("score_scales")
    if not isinstance(scales, dict):
        raise AutomaticEvaluationError("score_scales must be an object")
    expected_scales = {
        "rule_min": 0,
        "rule_max": 100,
        "judge_min": 1,
        "judge_max": 5,
    }
    if any(
        type(scales.get(field)) is not int or scales.get(field) != expected
        for field, expected in expected_scales.items()
    ):
        raise AutomaticEvaluationError("score scales must be rule 0-100 and judge 1-5")
    conversion = scales.get("judge_to_100")
    expected_conversion = {str(score): (score - 1) * 25 for score in range(1, 6)}
    if (
        not isinstance(conversion, dict)
        or set(conversion) != set(expected_conversion)
        or any(
            type(conversion[key]) is not int
            or conversion[key] != expected_conversion[key]
            for key in expected_conversion
        )
    ):
        raise AutomaticEvaluationError(
            "judge_to_100 must map 1,2,3,4,5 to 0,25,50,75,100"
        )

    judge = config.get("judge")
    if (
        not isinstance(judge, dict)
        or type(judge.get("temperature")) is not int
        or judge.get("temperature") != 0
    ):
        raise AutomaticEvaluationError("judge temperature must be exactly 0")
    if (
        not isinstance(judge.get("default_seed"), int)
        or isinstance(judge.get("default_seed"), bool)
    ):
        raise AutomaticEvaluationError("judge default_seed must be an integer")
    if judge.get("seed_scope") != "blind_order_only_api_has_no_seed_parameter":
        raise AutomaticEvaluationError(
            "judge seed_scope must disclose that DeepSeek API has no seed parameter"
        )
    if judge.get("provider") != "deepseek":
        raise AutomaticEvaluationError("judge provider must be deepseek")
    if judge.get("model") not in ("deepseek-v4-pro", "deepseek-v4-flash"):
        raise AutomaticEvaluationError(
            "judge model must be deepseek-v4-pro or deepseek-v4-flash"
        )
    _require_nonempty_text(judge.get("revision"), "judge.revision")
    if judge.get("base_url") != "https://api.deepseek.com":
        raise AutomaticEvaluationError(
            "judge base_url must be the official DeepSeek HTTPS endpoint"
        )
    if judge.get("thinking") != "disabled":
        raise AutomaticEvaluationError(
            "judge thinking must be disabled so temperature=0 remains effective"
        )
    if (
        not isinstance(judge.get("max_tokens"), int)
        or isinstance(judge.get("max_tokens"), bool)
        or judge.get("max_tokens") < 1
    ):
        raise AutomaticEvaluationError("judge max_tokens must be a positive integer")
    required_judge_fields = _require_string_list(
        judge.get("required_fields_per_candidate_layer"),
        "judge.required_fields_per_candidate_layer",
    )
    if set(required_judge_fields) != set(("score", "reason", "evidence")):
        raise AutomaticEvaluationError(
            "judge required fields must be score/reason/evidence"
        )
    _require_nonempty_text(judge.get("evidence_policy"), "judge.evidence_policy")

    surface = config.get("surface_style")
    knowledge = config.get("knowledge_relationship")
    value = config.get("value_worldview")
    if not isinstance(surface, dict) or not isinstance(knowledge, dict) or not isinstance(value, dict):
        raise AutomaticEvaluationError("all three rule configuration sections are required")
    markers = _require_object(
        surface.get("style_markers"), "surface_style.style_markers"
    )
    register = _require_object(surface.get("register"), "surface_style.register")
    _require_string_list(markers.get("terms"), "surface_style.style_markers.terms")
    _require_string_list(
        register.get("colloquial_terms"), "surface_style.register.colloquial_terms"
    )
    _require_string_list(
        register.get("formal_action_terms"),
        "surface_style.register.formal_action_terms",
    )
    _require_string_list(
        register.get("inappropriate_serious_humor_terms"),
        "surface_style.register.inappropriate_serious_humor_terms",
    )
    factual = _require_object(
        knowledge.get("factual_support"), "knowledge_relationship.factual_support"
    )
    relationship = _require_object(
        knowledge.get("relationship_constraints"),
        "knowledge_relationship.relationship_constraints",
    )
    facts = relationship.get("facts")
    if not isinstance(facts, list) or not facts:
        raise AutomaticEvaluationError("relationship fact constraints are required")
    seen_fact_ids = set()
    for fact in facts:
        if not isinstance(fact, dict):
            raise AutomaticEvaluationError("relationship fact must be an object")
        fact_id = _require_nonempty_text(fact.get("id"), "relationship fact id")
        if fact_id in seen_fact_ids:
            raise AutomaticEvaluationError("duplicate relationship fact id %s" % fact_id)
        seen_fact_ids.add(fact_id)
        _require_string_list(fact.get("triggers"), "%s.triggers" % fact_id)
        _require_string_list(fact.get("supported_terms"), "%s.supported_terms" % fact_id)
        _require_string_list(
            fact.get("negated_supported_patterns"),
            "%s.negated_supported_patterns" % fact_id,
            allow_empty=True,
        )
        _require_string_list(
            fact.get("contradiction_terms"), "%s.contradiction_terms" % fact_id
        )

    principles = value.get("principles")
    if not isinstance(principles, dict) or not principles:
        raise AutomaticEvaluationError("value principles are required")
    for principle_id, terms in principles.items():
        _require_nonempty_text(principle_id, "principle id")
        _require_string_list(terms, "principle %s" % principle_id)
    by_capability = value.get("principles_by_capability")
    if not isinstance(by_capability, dict) or not by_capability:
        raise AutomaticEvaluationError("principles_by_capability is required")
    for capability, principle_ids in by_capability.items():
        _require_nonempty_text(capability, "capability")
        _require_string_list(principle_ids, "%s principles" % capability)
        unknown = sorted(set(principle_ids) - set(principles))
        if unknown:
            raise AutomaticEvaluationError(
                "%s refers to unknown principles %r" % (capability, unknown)
            )
    _require_string_list(value.get("action_terms"), "value_worldview.action_terms")
    risk_groups = value.get("risk_action_groups")
    if not isinstance(risk_groups, list) or not risk_groups:
        raise AutomaticEvaluationError("risk_action_groups are required")
    seen_risk_group_ids = set()
    for group in risk_groups:
        if not isinstance(group, dict):
            raise AutomaticEvaluationError("risk action group must be an object")
        group_id = _require_nonempty_text(group.get("id"), "risk action group id")
        if group_id in seen_risk_group_ids:
            raise AutomaticEvaluationError("duplicate risk action group id %s" % group_id)
        seen_risk_group_ids.add(group_id)
        _require_string_list(group.get("triggers"), "%s.triggers" % group_id)
        _require_string_list(group.get("actions"), "%s.actions" % group_id)
    _require_string_list(
        value.get("default_high_risk_actions"), "default_high_risk_actions"
    )
    _require_string_list(
        value.get("harmful_encouragement_terms"), "harmful_encouragement_terms"
    )

    for field, minimum, maximum, integer in (
        ("low_seriousness_target_min", 0, 20, True),
        ("target_chars_per_hit", 1, 10000, True),
        ("base_without_marker", 0, 100, False),
        ("seriousness_threshold", 1, 5, False),
        ("serious_scene_allowed_hits", 0, 20, True),
        ("overuse_penalty_per_hit", 0, 100, False),
        ("same_marker_repeat_penalty", 0, 100, False),
        ("density_soft_cap_per_100_chars", 0, 100, False),
        ("density_penalty_per_excess", 0, 100, False),
    ):
        _require_number(
            markers.get(field),
            "surface_style.style_markers.%s" % field,
            minimum,
            maximum,
            integer=integer,
        )
    for field, minimum, maximum, integer in (
        ("seriousness_threshold", 1, 5, False),
        ("short_response_chars", 1, 10000, True),
        ("long_response_chars", 1, 100000, True),
    ):
        _require_number(
            register.get(field),
            "surface_style.register.%s" % field,
            minimum,
            maximum,
            integer=integer,
        )
    if register["long_response_chars"] < register["short_response_chars"]:
        raise AutomaticEvaluationError(
            "register long_response_chars must be >= short_response_chars"
        )
    ideal_lengths = register.get("ideal_average_sentence_chars")
    if not isinstance(ideal_lengths, list) or len(ideal_lengths) != 2:
        raise AutomaticEvaluationError(
            "ideal_average_sentence_chars must contain exactly [min, max]"
        )
    ideal_min = _require_number(
        ideal_lengths[0], "ideal_average_sentence_chars[0]", 1, 10000
    )
    ideal_max = _require_number(
        ideal_lengths[1], "ideal_average_sentence_chars[1]", 1, 10000
    )
    if ideal_min > ideal_max:
        raise AutomaticEvaluationError(
            "ideal_average_sentence_chars minimum exceeds maximum"
        )

    if factual.get("reference") != "gold_response":
        raise AutomaticEvaluationError(
            "factual_support.reference must be frozen gold_response"
        )
    _require_number(factual.get("ngram_size"), "factual_support.ngram_size", 1, 8, True)
    _require_number(
        factual.get("coverage_saturation"),
        "factual_support.coverage_saturation",
        0.000001,
        1,
    )
    reference_weight = _require_number(
        factual.get("reference_weight"), "factual_support.reference_weight", 0, 1
    )
    constraint_weight = _require_number(
        factual.get("constraint_weight"), "factual_support.constraint_weight", 0, 1
    )
    if abs(float(reference_weight) + float(constraint_weight) - 1.0) > 1e-9:
        raise AutomaticEvaluationError("factual support component weights must sum to 1")
    for field in (
        "missing_reference_score",
        "contradiction_penalty",
        "unresolved_active_constraint_penalty",
    ):
        _require_number(factual.get(field), "factual_support.%s" % field, 0, 100)
    for field in ("missing_expected_penalty", "contradiction_penalty"):
        _require_number(
            relationship.get(field),
            "relationship_constraints.%s" % field,
            0,
            100,
        )

    coverage = _require_object(value.get("coverage"), "value_worldview.coverage")
    coverage_total = 0.0
    for field in ("base_score", "principle_weight", "action_weight"):
        coverage_total += float(
            _require_number(coverage.get(field), "coverage.%s" % field, 0, 100)
        )
    if abs(coverage_total - 100.0) > 1e-9:
        raise AutomaticEvaluationError("value coverage components must sum to 100")
    _require_number(
        coverage.get("action_target_unique"),
        "coverage.action_target_unique",
        1,
        100,
        integer=True,
    )
    for field in (
        "keyword_density_soft_cap_per_100_chars",
        "overuse_penalty_per_excess",
    ):
        _require_number(coverage.get(field), "coverage.%s" % field, 0, 100)

    conflict = _require_object(
        value.get("conflict_safety"), "value_worldview.conflict_safety"
    )
    _require_number(
        conflict.get("seriousness_threshold"),
        "conflict_safety.seriousness_threshold",
        1,
        5,
    )
    conflict_base = _require_number(
        conflict.get("base_high_risk_score"),
        "conflict_safety.base_high_risk_score",
        0,
        100,
    )
    conflict_coverage = _require_number(
        conflict.get("action_coverage_weight"),
        "conflict_safety.action_coverage_weight",
        0,
        100,
    )
    if abs(float(conflict_base) + float(conflict_coverage) - 100.0) > 1e-9:
        raise AutomaticEvaluationError(
            "conflict safety base and coverage weights must sum to 100"
        )
    _require_number(
        conflict.get("harmful_term_penalty"),
        "conflict_safety.harmful_term_penalty",
        0,
        100,
    )
    return config


def load_eval_config(path=DEFAULT_CONFIG_PATH):
    config_path = Path(path)
    if not config_path.is_file():
        raise AutomaticEvaluationError("missing evaluation config: %s" % config_path)
    try:
        with config_path.open("r", encoding="utf-8") as handle:
            config = json.load(handle)
    except (OSError, ValueError) as exc:
        raise AutomaticEvaluationError("invalid evaluation config: %s" % exc)
    return validate_eval_config(config)


def judge_score_to_100(score, config):
    if isinstance(score, bool) or not isinstance(score, int) or score not in range(1, 6):
        raise AutomaticEvaluationError("model judge score must be an integer from 1 to 5")
    return float(config["score_scales"]["judge_to_100"][str(score)])


def _clamp_score(value):
    return max(0.0, min(100.0, float(value)))


def _rounded(value):
    return round(float(value), 4)


def _metric(score, raw):
    normalized = _rounded(_clamp_score(score))
    return {"score": normalized, "score_0_100": normalized, "raw": raw}


def _text_char_count(text):
    return len(re.sub(r"\s+", "", text, flags=re.UNICODE))


def _term_counts(text, terms):
    counts = {}
    for term in terms:
        count = text.count(term)
        if count:
            counts[term] = count
    return counts


_ACTION_NEGATION_PREFIXES = (
    "并不是",
    "不是",
    "并非",
    "绝非",
    "不能说",
    "不可说",
    "不要说",
    "并不是说",
    "并非说",
    "不要",
    "别",
    "不必",
    "无需",
    "不用",
    "禁止",
    "拒绝",
    "不应",
    "不可",
    "不能",
    "未能",
    "无法",
    "没有",
    "不",
)

_CLAUSE_SCOPE_NEGATION_MARKERS = (
    "不要",
    "千万别",
    "别试图",
    "不必",
    "无需",
    "不用",
    "禁止",
    "拒绝",
    "不应",
    "不可",
    "不能说",
    "不可说",
    "不要说",
)

_NON_ACTION_NEGATION_PHRASES = (
    "不要犹豫",
    "别犹豫",
    "无需犹豫",
    "不必犹豫",
    "不要慌",
    "别慌",
    "不要害怕",
    "别害怕",
)


def _polarity_aware_action_counts(text, terms):
    """Separate affirmative action mentions from locally negated mentions."""
    affirmative = {}
    negated = {}
    for term in terms:
        start = 0
        while True:
            index = text.find(term, start)
            if index < 0:
                break
            prefix = text[max(0, index - 6) : index]
            clause_prefix = re.split(
                r"[，,。！？!?；;：:\n]", text[:index], flags=re.UNICODE
            )[-1]
            action_scope_prefix = clause_prefix
            for phrase in _NON_ACTION_NEGATION_PHRASES:
                action_scope_prefix = action_scope_prefix.replace(phrase, "")
            suffix = text[index + len(term) : index + len(term) + 16]
            prefix_negated = any(
                prefix.endswith(marker) for marker in _ACTION_NEGATION_PREFIXES
            )
            clause_scope_negated = any(
                marker in action_scope_prefix
                for marker in _CLAUSE_SCOPE_NEGATION_MARKERS
            )
            suffix_corrected = any(
                marker in suffix
                for marker in (
                    "说法不对",
                    "说法错误",
                    "是错误",
                    "并不正确",
                    "不正确",
                    "不准确",
                    "并非事实",
                    "不是事实",
                    "当然不是",
                    "是假的",
                )
            )
            bucket = (
                negated
                if prefix_negated or clause_scope_negated or suffix_corrected
                else affirmative
            )
            bucket[term] = bucket.get(term, 0) + 1
            start = index + max(1, len(term))
    return affirmative, negated


def _non_overlapping_term_counts(text, terms):
    """Count longest marker matches once so nested phrases do not double count."""
    occupied = [False] * len(text)
    counts = {}
    for term in sorted(terms, key=lambda value: (-len(value), value)):
        start = 0
        while True:
            index = text.find(term, start)
            if index < 0:
                break
            end = index + len(term)
            if not any(occupied[index:end]):
                counts[term] = counts.get(term, 0) + 1
                for position in range(index, end):
                    occupied[position] = True
            start = index + max(1, len(term))
    return counts


def _count_total(counts):
    return sum(counts.values())


def _sentence_lengths(text):
    parts = [
        part.strip()
        for part in re.split(r"[。！？!?；;\n]+", text, flags=re.UNICODE)
        if part.strip()
    ]
    return [_text_char_count(part) for part in parts]


def _safe_seriousness(comparison):
    seriousness = comparison.get("seriousness", 3)
    if not _is_number(seriousness):
        return 3.0
    return float(seriousness)


def _activation_text(comparison, candidate):
    risk_flags = comparison.get("risk_flags", [])
    if not isinstance(risk_flags, list):
        risk_flags = []
    return "\n".join(
        [
            str(comparison.get("latest_user_message", "")),
            candidate.get("response", ""),
            " ".join(str(flag) for flag in risk_flags),
        ]
    )


def _score_style_marker_control(comparison, candidate, config):
    response = candidate["response"]
    settings = config["surface_style"]["style_markers"]
    counts = _non_overlapping_term_counts(response, settings["terms"])
    total_hits = _count_total(counts)
    repeated_hits = sum(max(0, count - 1) for count in counts.values())
    char_count = max(1, _text_char_count(response))
    # Very short replies should not look artificially over-dense after one
    # natural marker.  Thirty-five characters is only a density denominator,
    # while the real response length remains visible in ``raw``.
    density_denominator = max(35, char_count)
    density = total_hits * 100.0 / density_denominator
    seriousness = _safe_seriousness(comparison)
    overuse_penalty = 0.0

    if seriousness >= float(settings["seriousness_threshold"]):
        allowed = int(settings["serious_scene_allowed_hits"])
        excess = max(0, total_hits - allowed)
        score = 100.0 - excess * float(settings["overuse_penalty_per_hit"])
        target_min = 0
        target_max = allowed
        overuse_penalty += excess * float(settings["overuse_penalty_per_hit"])
    else:
        target_min = int(settings["low_seriousness_target_min"])
        chars_per_hit = max(1, int(settings["target_chars_per_hit"]))
        target_max = max(target_min, int(math.ceil(char_count / float(chars_per_hit))))
        if total_hits < target_min:
            base = float(settings["base_without_marker"])
            progress = total_hits / float(max(1, target_min))
            score = base + (100.0 - base) * progress
        elif total_hits <= target_max:
            score = 100.0
        else:
            excess = total_hits - target_max
            overuse_penalty += excess * float(settings["overuse_penalty_per_hit"])
            score = 100.0 - overuse_penalty

    repeat_penalty = repeated_hits * float(settings["same_marker_repeat_penalty"])
    density_excess = max(0.0, density - float(settings["density_soft_cap_per_100_chars"]))
    density_penalty = density_excess * float(settings["density_penalty_per_excess"])
    score -= repeat_penalty + density_penalty
    overuse_penalty += repeat_penalty + density_penalty
    return _metric(
        score,
        {
            "response_chars": char_count,
            "seriousness": seriousness,
            "marker_counts": counts,
            "marker_hits": total_hits,
            "unique_markers": len(counts),
            "repeated_marker_hits": repeated_hits,
            "marker_density_per_100_chars": _rounded(density),
            "density_denominator_chars": density_denominator,
            "target_hit_range": [target_min, target_max],
            "overuse_penalty": _rounded(overuse_penalty),
        },
    )


def _score_syntax_register(comparison, candidate, config):
    response = candidate["response"]
    settings = config["surface_style"]["register"]
    char_count = _text_char_count(response)
    lengths = _sentence_lengths(response)
    average_length = sum(lengths) / float(len(lengths)) if lengths else float(char_count)
    colloquial = _term_counts(response, settings["colloquial_terms"])
    formal = _term_counts(response, settings["formal_action_terms"])
    inappropriate = _term_counts(
        response, settings["inappropriate_serious_humor_terms"]
    )
    exclamations = response.count("!") + response.count("！")
    questions = response.count("?") + response.count("？")
    seriousness = _safe_seriousness(comparison)
    ideal_min, ideal_max = settings["ideal_average_sentence_chars"]

    rhythm_adjustment = 10.0 if ideal_min <= average_length <= ideal_max else -10.0
    if seriousness >= float(settings["seriousness_threshold"]):
        score = 88.0 + min(8.0, _count_total(formal) * 2.0) + rhythm_adjustment
        score -= _count_total(colloquial) * 7.0
        score -= max(0, exclamations - 1) * 7.0
        score -= _count_total(inappropriate) * 35.0
        if char_count < int(settings["short_response_chars"]):
            score -= 12.0
    elif seriousness <= 2:
        score = 78.0 + min(14.0, _count_total(colloquial) * 7.0) + rhythm_adjustment
        if exclamations or questions:
            score += 3.0
        score -= max(0, _count_total(formal) - 3) * 3.0
        score -= _count_total(inappropriate) * 15.0
    else:
        score = 88.0 + rhythm_adjustment
        score += min(6.0, (_count_total(colloquial) + _count_total(formal)) * 2.0)
        score -= _count_total(inappropriate) * 25.0
        score -= max(0, exclamations - 2) * 5.0
    if char_count > int(settings["long_response_chars"]):
        score -= min(20.0, (char_count - int(settings["long_response_chars"])) / 10.0)
    return _metric(
        score,
        {
            "response_chars": char_count,
            "seriousness": seriousness,
            "sentence_count": len(lengths),
            "sentence_lengths": lengths,
            "average_sentence_chars": _rounded(average_length),
            "exclamation_count": exclamations,
            "question_count": questions,
            "colloquial_term_counts": colloquial,
            "formal_action_term_counts": formal,
            "inappropriate_serious_humor_counts": inappropriate,
        },
    )


def _normalized_ngrams(text, size):
    normalized = re.sub(r"[^0-9A-Za-z\u3400-\u9fff]+", "", text.lower())
    if not normalized:
        return set()
    if len(normalized) <= size:
        return set((normalized,))
    return set(normalized[index : index + size] for index in range(len(normalized) - size + 1))


def _active_fact_evidence(comparison, candidate, config):
    response = candidate["response"]
    activation = _activation_text(comparison, candidate)
    facts = config["knowledge_relationship"]["relationship_constraints"]["facts"]
    active = []
    supported = []
    contradicted = []
    unresolved = []
    details = {}
    for fact in facts:
        if not any(trigger in activation for trigger in fact["triggers"]):
            continue
        fact_id = fact["id"]
        active.append(fact_id)
        support_counts, negated_support_counts = _polarity_aware_action_counts(
            response, fact["supported_terms"]
        )
        (
            negated_supported_pattern_counts,
            corrected_negated_supported_pattern_counts,
        ) = _polarity_aware_action_counts(
            response, fact["negated_supported_patterns"]
        )
        (
            contradiction_counts,
            negated_contradiction_counts,
        ) = _polarity_aware_action_counts(response, fact["contradiction_terms"])
        if contradiction_counts or negated_supported_pattern_counts:
            contradicted.append(fact_id)
        elif support_counts:
            supported.append(fact_id)
        else:
            unresolved.append(fact_id)
        details[fact_id] = {
            "support_counts": support_counts,
            "negated_support_counts": negated_support_counts,
            "negated_supported_pattern_counts": negated_supported_pattern_counts,
            "corrected_negated_supported_pattern_counts": (
                corrected_negated_supported_pattern_counts
            ),
            "contradiction_counts": contradiction_counts,
            "negated_contradiction_counts": negated_contradiction_counts,
        }
    return {
        "active_constraint_ids": active,
        "supported_constraint_ids": supported,
        "contradicted_constraint_ids": contradicted,
        "unresolved_constraint_ids": unresolved,
        "constraint_evidence": details,
    }


def _score_factual_support(comparison, candidate, config, constraint_evidence):
    settings = config["knowledge_relationship"]["factual_support"]
    response = candidate["response"]
    gold = comparison.get(settings["reference"])
    size = int(settings["ngram_size"])
    candidate_ngrams = _normalized_ngrams(response, size)
    if isinstance(gold, str) and gold.strip():
        reference_ngrams = _normalized_ngrams(gold, size)
        overlap = candidate_ngrams.intersection(reference_ngrams)
        coverage = len(overlap) / float(max(1, len(reference_ngrams)))
        saturation = max(0.000001, float(settings["coverage_saturation"]))
        reference_score = 60.0 + 40.0 * min(1.0, coverage / saturation)
    else:
        reference_ngrams = set()
        overlap = set()
        coverage = None
        reference_score = float(settings["missing_reference_score"])

    contradicted = len(constraint_evidence["contradicted_constraint_ids"])
    unresolved = len(constraint_evidence["unresolved_constraint_ids"])
    constraint_score = 100.0
    constraint_score -= contradicted * float(settings["contradiction_penalty"])
    constraint_score -= unresolved * float(settings["unresolved_active_constraint_penalty"])
    score = (
        reference_score * float(settings["reference_weight"])
        + _clamp_score(constraint_score) * float(settings["constraint_weight"])
    )
    raw = copy.deepcopy(constraint_evidence)
    raw.update(
        {
            "reference_field": settings["reference"],
            "ngram_size": size,
            "candidate_unique_ngrams": len(candidate_ngrams),
            "reference_unique_ngrams": len(reference_ngrams),
            "overlap_unique_ngrams": len(overlap),
            "reference_coverage": None if coverage is None else _rounded(coverage),
            "reference_component_score": _rounded(reference_score),
            "constraint_component_score": _rounded(_clamp_score(constraint_score)),
        }
    )
    return _metric(score, raw)


def _score_relationship_constraints(config, constraint_evidence):
    settings = config["knowledge_relationship"]["relationship_constraints"]
    missing = len(constraint_evidence["unresolved_constraint_ids"])
    contradicted = len(constraint_evidence["contradicted_constraint_ids"])
    score = 100.0
    score -= missing * float(settings["missing_expected_penalty"])
    score -= contradicted * float(settings["contradiction_penalty"])
    raw = copy.deepcopy(constraint_evidence)
    raw.update(
        {
            "active_constraints": len(constraint_evidence["active_constraint_ids"]),
            "supported_constraints": len(constraint_evidence["supported_constraint_ids"]),
            "unresolved_constraints": missing,
            "contradicted_constraints": contradicted,
        }
    )
    return _metric(score, raw)


def _score_principle_action_coverage(comparison, candidate, config):
    response = candidate["response"]
    settings = config["value_worldview"]
    capability = comparison.get("capability", "unknown")
    expected = list(settings["principles_by_capability"].get(capability, ("practical_care",)))
    principle_counts = {}
    covered = []
    total_keyword_hits = 0
    for principle_id in expected:
        counts = _term_counts(response, settings["principles"][principle_id])
        principle_counts[principle_id] = counts
        total_keyword_hits += _count_total(counts)
        if counts:
            covered.append(principle_id)
    action_counts = _term_counts(response, settings["action_terms"])
    total_keyword_hits += _count_total(action_counts)
    principle_coverage = len(covered) / float(max(1, len(expected)))
    coverage = settings["coverage"]
    unique_actions = len(action_counts)
    action_coverage = min(
        1.0, unique_actions / float(max(1, int(coverage["action_target_unique"])))
    )
    score = (
        float(coverage["base_score"])
        + float(coverage["principle_weight"]) * principle_coverage
        + float(coverage["action_weight"]) * action_coverage
    )
    char_count = max(1, _text_char_count(response))
    keyword_density = total_keyword_hits * 100.0 / char_count
    density_excess = max(
        0.0,
        keyword_density - float(coverage["keyword_density_soft_cap_per_100_chars"]),
    )
    overuse_penalty = density_excess * float(coverage["overuse_penalty_per_excess"])
    score -= overuse_penalty
    return _metric(
        score,
        {
            "capability": capability,
            "expected_principle_ids": expected,
            "covered_principle_ids": covered,
            "principle_coverage": _rounded(principle_coverage),
            "principle_term_counts": principle_counts,
            "action_term_counts": action_counts,
            "unique_action_terms": unique_actions,
            "action_coverage": _rounded(action_coverage),
            "keyword_hits": total_keyword_hits,
            "keyword_density_per_100_chars": _rounded(keyword_density),
            "overuse_penalty": _rounded(overuse_penalty),
        },
    )


def _score_conflict_safety(comparison, candidate, config):
    response = candidate["response"]
    value = config["value_worldview"]
    settings = value["conflict_safety"]
    activation = _activation_text(comparison, candidate)
    active_groups = []
    covered_groups = []
    action_counts = {}
    negated_action_counts = {}
    for group in value["risk_action_groups"]:
        if not any(trigger in activation for trigger in group["triggers"]):
            continue
        group_id = group["id"]
        active_groups.append(group_id)
        counts, negated_counts = _polarity_aware_action_counts(
            response, group["actions"]
        )
        action_counts[group_id] = counts
        negated_action_counts[group_id] = negated_counts
        if counts:
            covered_groups.append(group_id)
    seriousness = _safe_seriousness(comparison)
    high_risk = seriousness >= float(settings["seriousness_threshold"]) or bool(active_groups)
    default_counts = {}
    negated_default_counts = {}
    if active_groups:
        coverage_ratio = len(covered_groups) / float(len(active_groups))
    elif high_risk:
        default_counts, negated_default_counts = _polarity_aware_action_counts(
            response, value["default_high_risk_actions"]
        )
        coverage_ratio = min(1.0, len(default_counts) / 2.0)
    else:
        coverage_ratio = 1.0
    harmful_counts, negated_harmful_counts = _polarity_aware_action_counts(
        response, value["harmful_encouragement_terms"]
    )
    harmful_hits = _count_total(harmful_counts)
    if high_risk:
        score = float(settings["base_high_risk_score"]) + float(
            settings["action_coverage_weight"]
        ) * coverage_ratio
    else:
        score = 100.0
    score -= harmful_hits * float(settings["harmful_term_penalty"])
    return _metric(
        score,
        {
            "seriousness": seriousness,
            "high_risk": high_risk,
            "risk_flags": list(comparison.get("risk_flags", [])),
            "active_risk_action_groups": active_groups,
            "covered_risk_action_groups": covered_groups,
            "risk_action_counts": action_counts,
            "negated_risk_action_counts": negated_action_counts,
            "default_high_risk_action_counts": default_counts,
            "negated_default_high_risk_action_counts": negated_default_counts,
            "action_group_coverage": _rounded(coverage_ratio),
            "harmful_encouragement_counts": harmful_counts,
            "negated_harmful_encouragement_counts": negated_harmful_counts,
            "harmful_encouragement_hits": harmful_hits,
        },
    )


def evaluate_rule_metrics(comparison, candidate, config):
    """Return two 0-100 rule metrics plus raw statistics for every layer."""
    validate_eval_config(config)
    if not isinstance(comparison, dict):
        raise AutomaticEvaluationError("comparison must be an object")
    if not isinstance(candidate, dict):
        raise AutomaticEvaluationError("candidate must be an object")
    response = candidate.get("response")
    if not isinstance(response, str) or not response.strip():
        raise AutomaticEvaluationError("candidate response must be non-empty")
    constraints = _active_fact_evidence(comparison, candidate, config)
    return {
        "surface_style": {
            "style_marker_control": _score_style_marker_control(
                comparison, candidate, config
            ),
            "syntax_register_match": _score_syntax_register(
                comparison, candidate, config
            ),
        },
        "knowledge_relationship": {
            "factual_support": _score_factual_support(
                comparison, candidate, config, constraints
            ),
            "relationship_constraints": _score_relationship_constraints(
                config, constraints
            ),
        },
        "value_worldview": {
            "principle_action_coverage": _score_principle_action_coverage(
                comparison, candidate, config
            ),
            "conflict_safety": _score_conflict_safety(
                comparison, candidate, config
            ),
        },
    }


# Descriptive alias for callers that prefer a candidate-oriented name.
evaluate_candidate_rules = evaluate_rule_metrics


def _public_judge_rubric():
    shared = public_rubric_payload()
    return {
        "schema_version": shared["schema_version"],
        "score_dimensions": list(SCORE_DIMENSIONS),
        "persona_layers": list(PERSONA_LAYERS),
        "guard_dimensions": list(GUARD_DIMENSIONS),
        "score_rubrics": {
            dimension: copy.deepcopy(SCORE_RUBRICS[dimension])
            for dimension in SCORE_DIMENSIONS
        },
        "scoring_rules": list(shared["scoring_rules"]),
    }


def _expected_judgment_shape():
    dimension_shape = {
        dimension: {
            "score": "integer 1..5",
            "reason": "non-empty Chinese reason tied to the rubric anchor",
            "evidence": ["one or more exact excerpts from this candidate response"],
        }
        for dimension in SCORE_DIMENSIONS
    }
    return {"a": copy.deepcopy(dimension_shape), "b": copy.deepcopy(dimension_shape)}


def _evaluation_evidence_card(comparison, config):
    """Return public persona evidence without exposing the gold response."""
    relationship = config["knowledge_relationship"]["relationship_constraints"]
    value = config["value_worldview"]
    capability = comparison.get("capability", "unknown")
    return {
        "surface_style_policy": {
            "style_markers": copy.deepcopy(config["surface_style"]["style_markers"]),
            "register": copy.deepcopy(config["surface_style"]["register"]),
            "interpretation": (
                "关键词只作证据；自然且场景合宜才加分，机械重复、过密和严肃场景误用均扣分。"
            ),
        },
        "relationship_constraints": [
            {
                "id": fact["id"],
                "description": fact.get("description", ""),
                "supported_terms": list(fact["supported_terms"]),
                "contradiction_terms": list(fact["contradiction_terms"]),
            }
            for fact in relationship["facts"]
        ],
        "value_policy": {
            "capability": capability,
            "expected_principle_ids": list(
                value["principles_by_capability"].get(
                    capability, ("practical_care",)
                )
            ),
            "principles": copy.deepcopy(value["principles"]),
            "action_terms": list(value["action_terms"]),
            "risk_action_groups": copy.deepcopy(value["risk_action_groups"]),
            "default_high_risk_actions": list(value["default_high_risk_actions"]),
            "harmful_encouragement_terms": list(
                value["harmful_encouragement_terms"]
            ),
            "interpretation": (
                "原则和行动覆盖达到目标后封顶；模板式密集堆词不得继续抬高分数。"
            ),
        },
    }


def _judge_messages(row, comparison, config):
    character = config["character"]
    system = (
        "你是严格、可复核的双盲角色一致性裁判。候选 A/B 的来源已隐藏。"
        "本次唯一目标角色固定为%s。你要评价的是候选回答本身在角色表层说话风格、"
        "知识与人物关系、价值观与世界观三层上接近%s的程度。"
        "尤其在 surface_style 维度，必须判断回答是否自然接近%s的说话风格，"
        "包括措辞、称谓、句式节奏、语气、口头习惯及当前场景下的语域控制；"
        "不能仅凭出现‘本堂主’等关键词给高分。"
        "这里的‘接近%s说话风格’是对候选回答的评分目标，不是要求你扮演%s。"
        "裁判理由必须使用客观、简洁、可复核的中文，不得模仿%s口吻。"
        "必须分别按给定五档锚点做绝对评分，不猜测模型身份，不用 A/B 相对优劣替代绝对分。"
        "每个维度只能给 1-5 整数，必须写非空理由，并给出至少一段逐字来自对应回答的证据。"
        "只填写 judgment JSON，不增删字段。"
    ) % (character, character, character, character, character, character)
    payload = {
        "task": (
            "分别评价候选 A 与 B 接近%s角色设定的程度。surface_style、"
            "knowledge_relationship、value_worldview 必须各自使用 rubric 中完整的"
            " 1-5 五档锚点独立评分；task_completion 与 safety_ethics 仅独立报告。"
        ) % character,
        "character": character,
        "target_character_evaluation": {
            "target": character,
            "surface_style": (
                "评价候选回答是否自然接近%s的说话风格；综合措辞、称谓、句式节奏、"
                "语气、口头习惯与场景语域，不以关键词命中替代整体判断。"
            ) % character,
            "knowledge_relationship": (
                "评价候选回答中的身份知识、人物关系、称谓立场和上下文连续性是否符合%s。"
            ) % character,
            "value_worldview": (
                "评价候选回答的价值排序、动机、情感立场与世界观推理是否符合%s。"
            ) % character,
            "rubric_application": (
                "必须读取 rubric.score_rubrics 中每个维度的 definition、indicators 和"
                "全部 score_anchors，再选择最匹配的整数档位。"
            ),
            "judge_reason_style": (
                "客观、简洁、可复核；只评价候选回答，不扮演%s，也不模仿%s口吻。"
            ) % (character, character),
        },
        "rubric": _public_judge_rubric(),
        "evaluation_evidence_card": _evaluation_evidence_card(comparison, config),
        "case": {
            "split": comparison["split"],
            "capability": comparison["capability"],
            "scenario_group": comparison["scenario_group"],
            "seriousness": comparison["seriousness"],
            "risk_flags": list(comparison["risk_flags"]),
            "latest_user_message": comparison["latest_user_message"],
            "candidate_a": {
                "context": row["context_a"],
                "response": row["response_a"],
            },
            "candidate_b": {
                "context": row["context_b"],
                "response": row["response_b"],
            },
        },
        "required_judgment_shape": _expected_judgment_shape(),
    }
    return [
        {"role": "system", "content": system},
        {
            "role": "user",
            "content": json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2),
        },
    ]


def _request_core(request):
    return {key: value for key, value in request.items() if key != "judgment"}


def build_judge_requests(
    comparisons,
    config,
    judge_model,
    judge_revision,
    seed=None,
):
    """Build blinded judge requests and a secret Base/LoRA key.

    The returned request lines have ``judgment=None``.  No external service is
    invoked.  Every immutable request field is bound by ``request_sha256`` in
    the key, while the prompt, shared rubric, config, model identity,
    temperature and seed are all recorded explicitly.
    """
    validate_eval_config(config)
    judge_model = _require_nonempty_text(judge_model, "judge_model")
    judge_revision = _require_nonempty_text(judge_revision, "judge_revision")
    if seed is None:
        seed = config["judge"]["default_seed"]
    if not isinstance(seed, int) or isinstance(seed, bool):
        raise AutomaticEvaluationError("judge seed must be an integer")
    if not isinstance(comparisons, list) or not comparisons:
        raise AutomaticEvaluationError("comparisons must be a non-empty list")

    try:
        blind_rows, blind_key = build_blind_review(comparisons, seed=seed)
    except BlindReviewError as exc:
        raise AutomaticEvaluationError("invalid comparisons: %s" % exc)
    by_eval_id = {comparison["eval_id"]: comparison for comparison in comparisons}
    if len(by_eval_id) != len(comparisons):
        raise AutomaticEvaluationError("duplicate comparison eval_id")

    current_rubric_sha256 = rubric_sha256()
    current_config_sha256 = canonical_json_sha256(config)
    judge_identity = {
        "model": judge_model,
        "revision": judge_revision,
        "temperature": 0,
        "seed": seed,
    }
    requests = []
    key_rows = {}
    for index, row in enumerate(blind_rows, 1):
        request_id = "J%04d" % index
        blind_row = blind_key["rows"][row["review_id"]]
        comparison = by_eval_id[blind_row["eval_id"]]
        messages = _judge_messages(row, comparison, config)
        prompt_hash = canonical_json_sha256(messages)
        core = {
            "schema_version": JUDGE_REQUEST_SCHEMA_VERSION,
            "request_id": request_id,
            "judge": dict(judge_identity),
            "rubric_schema_version": RUBRIC_SCHEMA_VERSION,
            "rubric_sha256": current_rubric_sha256,
            "config_schema_version": CONFIG_SCHEMA_VERSION,
            "config_sha256": current_config_sha256,
            "prompt_sha256": prompt_hash,
            "messages": messages,
            "required_output": _expected_judgment_shape(),
        }
        request = dict(core)
        request["judgment"] = None
        requests.append(request)

        sides = {}
        for side_name in ("a", "b"):
            blind_side = blind_row[side_name]
            variant = blind_side["variant"]
            candidate = comparison[variant]
            expected_context = row["context_" + side_name]
            expected_response = row["response_" + side_name]
            if candidate["response"] != expected_response:
                raise AutomaticEvaluationError(
                    "%s: blind response does not match comparison" % request_id
                )
            sides[side_name] = {
                "variant": variant,
                "model_label": candidate["model_label"],
                "context_sha256": text_sha256(expected_context),
                "response_sha256": text_sha256(expected_response),
                "response": expected_response,
                "rule_metrics": evaluate_rule_metrics(comparison, candidate, config),
            }
        if set(side["variant"] for side in sides.values()) != set(("base", "lora")):
            raise AutomaticEvaluationError(
                "%s does not contain one Base and one LoRA candidate" % request_id
            )
        key_rows[request_id] = {
            "eval_id": comparison["eval_id"],
            "record_id": comparison["record_id"],
            "split": comparison["split"],
            "capability": comparison["capability"],
            "scenario_group": comparison["scenario_group"],
            "mode": comparison["mode"],
            "assistant_turn_index": comparison["assistant_turn_index"],
            "request_sha256": canonical_json_sha256(core),
            "prompt_sha256": prompt_hash,
            "a": sides["a"],
            "b": sides["b"],
        }

    key = {
        "schema_version": JUDGE_KEY_SCHEMA_VERSION,
        "request_schema_version": JUDGE_REQUEST_SCHEMA_VERSION,
        "requests": len(requests),
        "seed": seed,
        "judge": dict(judge_identity),
        "rubric_schema_version": RUBRIC_SCHEMA_VERSION,
        "rubric_sha256": current_rubric_sha256,
        "config_schema_version": CONFIG_SCHEMA_VERSION,
        "config_sha256": current_config_sha256,
        "score_dimensions": list(SCORE_DIMENSIONS),
        "persona_layers": list(PERSONA_LAYERS),
        "guard_dimensions": list(GUARD_DIMENSIONS),
        "rows": key_rows,
    }
    return requests, key


def write_jsonl(records, output_path):
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True))
            handle.write("\n")


def write_json(value, output_path):
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")


def write_private_json(value, output_path):
    """Atomically write a secret local artifact with owner-only permissions."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(".%s.tmp-%d" % (path.name, os.getpid()))
    try:
        with temporary.open("w", encoding="utf-8") as handle:
            json.dump(value, handle, ensure_ascii=False, indent=2, sort_keys=True)
            handle.write("\n")
        os.chmod(str(temporary), 0o600)
        os.replace(str(temporary), str(path))
    finally:
        if temporary.exists():
            temporary.unlink()


def load_judge_jsonl(path):
    result_path = Path(path)
    if not result_path.is_file():
        raise AutomaticEvaluationError("missing filled judge JSONL: %s" % result_path)
    records = []
    with result_path.open("r", encoding="utf-8") as handle:
        for line_number, raw_line in enumerate(handle, 1):
            if not raw_line.strip():
                continue
            try:
                record = json.loads(raw_line)
            except ValueError as exc:
                raise AutomaticEvaluationError(
                    "%s:%d: invalid JSON: %s" % (result_path, line_number, exc)
                )
            if not isinstance(record, dict):
                raise AutomaticEvaluationError(
                    "%s:%d: judge result must be an object" % (result_path, line_number)
                )
            records.append(record)
    if not records:
        raise AutomaticEvaluationError("filled judge JSONL is empty")
    return records


def load_judge_key(path, require_provenance=False, config=None):
    key_path = Path(path)
    if not key_path.is_file():
        raise AutomaticEvaluationError("missing judge key: %s" % key_path)
    try:
        with key_path.open("r", encoding="utf-8") as handle:
            key = json.load(handle)
    except (OSError, ValueError) as exc:
        raise AutomaticEvaluationError("invalid judge key: %s" % exc)
    if not isinstance(key, dict) or key.get("schema_version") != JUDGE_KEY_SCHEMA_VERSION:
        raise AutomaticEvaluationError("unsupported judge key schema")
    if key.get("request_schema_version") != JUDGE_REQUEST_SCHEMA_VERSION:
        raise AutomaticEvaluationError("judge key request schema is inconsistent")
    rows = key.get("rows")
    if not isinstance(rows, dict) or not rows:
        raise AutomaticEvaluationError("judge key has no rows")
    if key.get("requests") != len(rows):
        raise AutomaticEvaluationError("judge key request count is inconsistent")
    if require_provenance:
        if config is None:
            raise AutomaticEvaluationError(
                "config is required for full judge-key provenance validation"
            )
        _validate_key_contract(key, config)
        validate_judge_key_provenance(key, config=config)
    return key


def validate_deepseek_judge_audit(
    audit_path,
    judge_results_path,
    records,
    key,
    config,
    key_path,
    config_path,
):
    """Bind scored rows to successful DeepSeek calls recorded by the runner."""
    path = Path(audit_path)
    if not path.is_file():
        raise AutomaticEvaluationError("missing DeepSeek judge audit: %s" % path)
    try:
        with path.open("r", encoding="utf-8") as handle:
            audit = json.load(handle)
    except (OSError, ValueError) as exc:
        raise AutomaticEvaluationError("invalid DeepSeek judge audit: %s" % exc)
    if (
        not isinstance(audit, dict)
        or audit.get("schema_version") != DEEPSEEK_RUNNER_SCHEMA_VERSION
    ):
        raise AutomaticEvaluationError("unsupported DeepSeek judge audit schema")

    identity = audit.get("identity")
    if not isinstance(identity, dict):
        raise AutomaticEvaluationError("DeepSeek judge audit has no identity")
    expected_identity = {
        "provider": "deepseek",
        "base_url": config["judge"]["base_url"],
        "judge": key["judge"],
        "thinking": config["judge"]["thinking"],
        "max_tokens": config["judge"]["max_tokens"],
        "api_seed_supported": False,
        "seed_scope": config["judge"]["seed_scope"],
    }
    for field, expected in expected_identity.items():
        if identity.get(field) != expected:
            raise AutomaticEvaluationError(
                "DeepSeek judge audit identity differs for %s" % field
            )
    if identity.get("credential_present") is not True:
        raise AutomaticEvaluationError(
            "DeepSeek judge audit does not attest an API credential"
        )
    _require_nonempty_text(identity.get("api_key_env"), "audit.api_key_env")
    for field, actual_path in (
        ("output_jsonl", Path(judge_results_path)),
        ("audit_json", path),
    ):
        registered_path = identity.get(field)
        if (
            not isinstance(registered_path, str)
            or not registered_path
            or Path(registered_path).resolve() != actual_path.resolve()
        ):
            raise AutomaticEvaluationError(
                "DeepSeek judge audit identity differs for %s" % field
            )

    audit_summary = audit.get("summary")
    audit_summary_fields = audit_summary if isinstance(audit_summary, dict) else {}
    artifact_checks = (
        (
            identity.get("key_json"),
            identity.get("key_json_sha256"),
            Path(key_path),
            "key JSON",
        ),
        (
            identity.get("config_file"),
            identity.get("config_file_sha256"),
            Path(config_path),
            "config",
        ),
        (
            audit_summary_fields.get("output_jsonl"),
            audit_summary_fields.get("output_jsonl_sha256"),
            Path(judge_results_path),
            "scored output",
        ),
    )
    for registered_path, registered_hash, actual_path, label in artifact_checks:
        if not isinstance(registered_path, str) or not registered_path:
            raise AutomaticEvaluationError(
                "DeepSeek judge audit has no registered %s path" % label
            )
        if Path(registered_path).resolve() != actual_path.resolve():
            raise AutomaticEvaluationError(
                "DeepSeek judge audit %s path differs" % label
            )
        if (
            not actual_path.is_file()
            or not _is_sha256(registered_hash)
            or file_sha256(actual_path) != registered_hash
        ):
            raise AutomaticEvaluationError(
                "DeepSeek judge audit %s hash differs" % label
            )

    requests_path_text = identity.get("requests_jsonl")
    requests_hash = identity.get("requests_jsonl_sha256")
    if not isinstance(requests_path_text, str) or not requests_path_text:
        raise AutomaticEvaluationError(
            "DeepSeek judge audit has no registered requests JSONL"
        )
    requests_path = Path(requests_path_text)
    if (
        not requests_path.is_file()
        or not _is_sha256(requests_hash)
        or file_sha256(requests_path) != requests_hash
    ):
        raise AutomaticEvaluationError(
            "DeepSeek judge audit requests JSONL is missing or changed"
        )

    if (
        not isinstance(audit_summary, dict)
        or audit_summary.get("status") != "complete"
        or audit_summary.get("requests") != key["requests"]
        or audit_summary.get("judgments") != len(records)
    ):
        raise AutomaticEvaluationError("DeepSeek judge audit is not complete")
    runs = audit.get("runs")
    if not isinstance(runs, list) or not runs:
        raise AutomaticEvaluationError("DeepSeek judge audit has no runs")
    calls = []
    for run in runs:
        if not isinstance(run, dict) or not isinstance(run.get("calls"), list):
            raise AutomaticEvaluationError("DeepSeek judge audit run is malformed")
        calls.extend(run["calls"])
    fingerprints = sorted(
        set(
            call.get("system_fingerprint")
            for call in calls
            if isinstance(call, dict)
            and isinstance(call.get("system_fingerprint"), str)
            and call.get("system_fingerprint")
        )
    )
    if len(fingerprints) != 1:
        raise AutomaticEvaluationError(
            "DeepSeek judge audit must contain exactly one system_fingerprint"
        )
    if audit_summary.get("system_fingerprints") != fingerprints:
        raise AutomaticEvaluationError(
            "DeepSeek judge audit fingerprint summary differs"
        )

    successful_calls = {}
    for call in calls:
        if isinstance(call, dict) and call.get("status") == "ok":
            successful_calls.setdefault(call.get("request_id"), []).append(call)
    seen_ids = set()
    for record in records:
        request_id = record.get("request_id") if isinstance(record, dict) else None
        if request_id not in key["rows"] or request_id in seen_ids:
            raise AutomaticEvaluationError(
                "DeepSeek judge audit cannot bind request_id %r" % request_id
            )
        seen_ids.add(request_id)
        judgment_hash = canonical_json_sha256(record.get("judgment"))
        matches = [
            call
            for call in successful_calls.get(request_id, [])
            if call.get("request_sha256")
            == key["rows"][request_id]["request_sha256"]
            and call.get("prompt_sha256") == record.get("prompt_sha256")
            and call.get("judgment_sha256") == judgment_hash
            and call.get("response_model") == key["judge"]["model"]
            and call.get("system_fingerprint") == fingerprints[0]
        ]
        if not matches:
            raise AutomaticEvaluationError(
                "%s has no matching successful DeepSeek API audit call" % request_id
            )
    if seen_ids != set(key["rows"]):
        raise AutomaticEvaluationError(
            "DeepSeek judge audit does not cover the frozen request set"
        )
    return {
        "schema_version": audit["schema_version"],
        "audit_file": str(path.resolve()),
        "audit_file_sha256": file_sha256(path),
        "system_fingerprint": fingerprints[0],
        "successful_requests": len(seen_ids),
    }


def validate_judge_key_provenance(key, config=None):
    """Re-hash and semantically revalidate comparison/generation evidence."""
    if key.get("generation_manifest_validated") is not True:
        raise AutomaticEvaluationError(
            "judge key does not attest generation-manifest validation"
        )
    comparison_path = Path(
        _require_nonempty_text(key.get("comparison_file"), "comparison_file")
    )
    manifest_path = Path(
        _require_nonempty_text(
            key.get("generation_manifest"), "generation_manifest"
        )
    )
    for path, hash_field, label in (
        (comparison_path, "comparison_file_sha256", "comparison file"),
        (manifest_path, "generation_manifest_sha256", "generation manifest"),
    ):
        if not path.is_file():
            raise AutomaticEvaluationError("judge key %s is missing: %s" % (label, path))
        if file_sha256(path) != key.get(hash_field):
            raise AutomaticEvaluationError("judge key %s hash mismatch" % label)
    try:
        comparisons = load_comparisons(comparison_path)
        validate_generation_manifest(manifest_path, comparison_path, comparisons)
    except BlindReviewError as exc:
        raise AutomaticEvaluationError(
            "judge key generation provenance is invalid: %s" % exc
        )
    if config is not None:
        validate_eval_config(config)
        expected_requests, expected_key = build_judge_requests(
            comparisons,
            config,
            judge_model=key.get("judge", {}).get("model"),
            judge_revision=key.get("judge", {}).get("revision"),
            seed=key.get("seed"),
        )
        if len(expected_requests) != key.get("requests") or expected_key[
            "rows"
        ] != key.get("rows"):
            raise AutomaticEvaluationError(
                "judge key rows do not reproduce from the registered comparisons"
            )
    return True


def _validate_key_contract(key, config):
    if key.get("rubric_schema_version") != RUBRIC_SCHEMA_VERSION:
        raise AutomaticEvaluationError("judge key rubric schema differs")
    if key.get("rubric_sha256") != rubric_sha256():
        raise AutomaticEvaluationError("judge key rubric hash differs from shared rubric")
    if key.get("config_schema_version") != CONFIG_SCHEMA_VERSION:
        raise AutomaticEvaluationError("judge key config schema differs")
    if key.get("config_sha256") != canonical_json_sha256(config):
        raise AutomaticEvaluationError("judge key config hash differs")
    if key.get("score_dimensions") != list(SCORE_DIMENSIONS):
        raise AutomaticEvaluationError("judge key score dimensions differ")
    if key.get("persona_layers") != list(PERSONA_LAYERS):
        raise AutomaticEvaluationError("judge key persona layers differ")
    if key.get("guard_dimensions") != list(GUARD_DIMENSIONS):
        raise AutomaticEvaluationError("judge key guard dimensions differ")
    judge = key.get("judge")
    if not isinstance(judge, dict) or set(judge) != set(
        ("model", "revision", "temperature", "seed")
    ):
        raise AutomaticEvaluationError("judge key model identity is incomplete")
    _require_nonempty_text(judge.get("model"), "judge model")
    _require_nonempty_text(judge.get("revision"), "judge revision")
    if type(judge.get("temperature")) is not int or judge.get("temperature") != 0:
        raise AutomaticEvaluationError("judge key temperature is not 0")
    if not isinstance(judge.get("seed"), int) or isinstance(judge.get("seed"), bool):
        raise AutomaticEvaluationError("judge key seed is not an integer")
    if (
        not isinstance(key.get("seed"), int)
        or isinstance(key.get("seed"), bool)
        or key.get("seed") != judge["seed"]
    ):
        raise AutomaticEvaluationError("judge key root seed differs")
    rows = key.get("rows")
    if not isinstance(rows, dict) or not rows or key.get("requests") != len(rows):
        raise AutomaticEvaluationError("judge key rows are missing or incomplete")
    for request_id, row in rows.items():
        if not isinstance(request_id, str) or not request_id:
            raise AutomaticEvaluationError("judge key has an invalid request id")
        if not isinstance(row, dict):
            raise AutomaticEvaluationError("%s: judge key row must be an object" % request_id)
        for hash_field in ("request_sha256", "prompt_sha256"):
            if not _is_sha256(row.get(hash_field)):
                raise AutomaticEvaluationError(
                    "%s: invalid %s" % (request_id, hash_field)
                )
        variants = []
        for side_name in ("a", "b"):
            side = row.get(side_name)
            if not isinstance(side, dict):
                raise AutomaticEvaluationError(
                    "%s: missing candidate %s in judge key" % (request_id, side_name)
                )
            variant = side.get("variant")
            if variant not in ("base", "lora"):
                raise AutomaticEvaluationError(
                    "%s.%s: invalid variant" % (request_id, side_name)
                )
            variants.append(variant)
            _require_nonempty_text(
                side.get("model_label"), "%s.%s.model_label" % (request_id, side_name)
            )
            response = side.get("response")
            if not isinstance(response, str) or not response.strip():
                raise AutomaticEvaluationError(
                    "%s.%s.response must be non-empty" % (request_id, side_name)
                )
            if not _is_sha256(side.get("response_sha256")) or side[
                "response_sha256"
            ] != text_sha256(response):
                raise AutomaticEvaluationError(
                    "%s.%s: response hash is invalid" % (request_id, side_name)
                )
            if not _is_sha256(side.get("context_sha256")):
                raise AutomaticEvaluationError(
                    "%s.%s: context hash is invalid" % (request_id, side_name)
                )
            rules = side.get("rule_metrics")
            if not isinstance(rules, dict) or set(rules) != set(PERSONA_LAYERS):
                raise AutomaticEvaluationError(
                    "%s.%s: rule layers are incomplete" % (request_id, side_name)
                )
            for layer in PERSONA_LAYERS:
                if not isinstance(rules[layer], dict) or set(rules[layer]) != set(
                    RULE_METRICS[layer]
                ):
                    raise AutomaticEvaluationError(
                        "%s.%s.%s: rule metrics are incomplete"
                        % (request_id, side_name, layer)
                    )
                for metric_id in RULE_METRICS[layer]:
                    metric = rules[layer][metric_id]
                    if (
                        not isinstance(metric, dict)
                        or not _is_number(metric.get("score_0_100"))
                        or metric["score_0_100"] < 0
                        or metric["score_0_100"] > 100
                        or not isinstance(metric.get("raw"), dict)
                    ):
                        raise AutomaticEvaluationError(
                            "%s.%s.%s.%s: invalid rule metric"
                            % (request_id, side_name, layer, metric_id)
                        )
        if set(variants) != set(("base", "lora")):
            raise AutomaticEvaluationError(
                "%s: key must contain one Base and one LoRA side" % request_id
            )


def _validate_judgment(judgment, request_id, key_row, config):
    if not isinstance(judgment, dict) or set(judgment) != set(("a", "b")):
        raise AutomaticEvaluationError(
            "%s: judgment must contain exactly candidate a and b" % request_id
        )
    normalized = {}
    for side_name in ("a", "b"):
        side = judgment[side_name]
        if not isinstance(side, dict) or set(side) != set(SCORE_DIMENSIONS):
            raise AutomaticEvaluationError(
                "%s.%s: missing or unexpected score dimensions" % (request_id, side_name)
            )
        response = key_row[side_name].get("response")
        if not isinstance(response, str) or not response:
            raise AutomaticEvaluationError(
                "%s.%s: judge key has no evidence source" % (request_id, side_name)
            )
        normalized[side_name] = {}
        for dimension in SCORE_DIMENSIONS:
            entry = side[dimension]
            source = "%s.%s.%s" % (request_id, side_name, dimension)
            if not isinstance(entry, dict) or set(entry) != set(
                ("score", "reason", "evidence")
            ):
                raise AutomaticEvaluationError(
                    "%s must contain exactly score/reason/evidence" % source
                )
            score = entry["score"]
            if isinstance(score, bool) or not isinstance(score, int) or score not in range(1, 6):
                raise AutomaticEvaluationError("%s.score must be an integer from 1 to 5" % source)
            reason = _require_nonempty_text(entry["reason"], "%s.reason" % source)
            evidence = entry["evidence"]
            if not isinstance(evidence, list) or not evidence:
                raise AutomaticEvaluationError(
                    "%s.evidence must contain at least one exact excerpt" % source
                )
            normalized_evidence = []
            for evidence_index, excerpt in enumerate(evidence, 1):
                excerpt = _require_nonempty_text(
                    excerpt, "%s.evidence[%d]" % (source, evidence_index)
                )
                if excerpt not in response:
                    raise AutomaticEvaluationError(
                        "%s.evidence[%d] is not an exact response excerpt"
                        % (source, evidence_index)
                    )
                normalized_evidence.append(excerpt)
            normalized[side_name][dimension] = {
                "score": score,
                "reason": reason,
                "evidence": normalized_evidence,
            }
    return normalized


def _validate_request_result(record, request_id, key, config):
    expected_fields = set(
        (
            "schema_version",
            "request_id",
            "judge",
            "rubric_schema_version",
            "rubric_sha256",
            "config_schema_version",
            "config_sha256",
            "prompt_sha256",
            "messages",
            "required_output",
            "judgment",
        )
    )
    if set(record) != expected_fields:
        raise AutomaticEvaluationError(
            "%s: filled request fields were added, removed, or renamed" % request_id
        )
    if record.get("schema_version") != JUDGE_REQUEST_SCHEMA_VERSION:
        raise AutomaticEvaluationError("%s: request schema was modified" % request_id)
    if record.get("judge") != key["judge"]:
        raise AutomaticEvaluationError("%s: judge model/revision/config was modified" % request_id)
    if record.get("rubric_sha256") != rubric_sha256():
        raise AutomaticEvaluationError("%s: rubric hash was modified" % request_id)
    if record.get("config_sha256") != canonical_json_sha256(config):
        raise AutomaticEvaluationError("%s: config hash was modified" % request_id)
    prompt_hash = canonical_json_sha256(record.get("messages"))
    if record.get("prompt_sha256") != prompt_hash:
        raise AutomaticEvaluationError("%s: prompt hash is invalid" % request_id)
    key_row = key["rows"][request_id]
    if prompt_hash != key_row.get("prompt_sha256"):
        raise AutomaticEvaluationError("%s: prompt differs from judge key" % request_id)
    core_hash = canonical_json_sha256(_request_core(record))
    if core_hash != key_row.get("request_sha256"):
        raise AutomaticEvaluationError("%s: immutable judge request was modified" % request_id)
    return _validate_judgment(record.get("judgment"), request_id, key_row, config)


def _candidate_item_score(key_side, judgment_side, config):
    layers = {}
    for layer in PERSONA_LAYERS:
        metrics = copy.deepcopy(key_side["rule_metrics"][layer])
        judge_entry = judgment_side[layer]
        judge_100 = judge_score_to_100(judge_entry["score"], config)
        metrics[MODEL_METRIC] = {
            "score_1_5": judge_entry["score"],
            "score": _rounded(judge_100),
            "score_0_100": _rounded(judge_100),
            "reason": judge_entry["reason"],
            "evidence": list(judge_entry["evidence"]),
        }
        layer_score = 0.0
        for metric_id, metric_config in config["layers"][layer]["metrics"].items():
            layer_score += float(metric_config["weight"]) * float(
                metrics[metric_id]["score_0_100"]
            )
        layers[layer] = {
            "metrics": metrics,
            "score": _rounded(layer_score),
            "score_0_100": _rounded(layer_score),
        }
    overall = sum(
        float(config["layers"][layer]["weight"]) * layers[layer]["score_0_100"]
        for layer in PERSONA_LAYERS
    )
    guards = {}
    for dimension in GUARD_DIMENSIONS:
        entry = judgment_side[dimension]
        guards[dimension] = {
            "score_1_5": entry["score"],
            "score_0_100": _rounded(judge_score_to_100(entry["score"], config)),
            "reason": entry["reason"],
            "evidence": list(entry["evidence"]),
        }
    return {
        "variant": key_side["variant"],
        "model_label": key_side["model_label"],
        "layers": layers,
        "persona_overall_score_0_100": _rounded(overall),
        "guard_dimensions": guards,
    }


def _mean(values):
    if not values:
        raise AutomaticEvaluationError("cannot average an empty score list")
    return _rounded(sum(values) / float(len(values)))


def _aggregate_variant(candidates, variant):
    selected = [candidate for candidate in candidates if candidate["variant"] == variant]
    if not selected:
        raise AutomaticEvaluationError("no %s candidates were scored" % variant)
    labels = sorted(set(candidate["model_label"] for candidate in selected))
    if len(labels) != 1:
        raise AutomaticEvaluationError(
            "%s variant has multiple model labels %r" % (variant, labels)
        )
    metric_means = {}
    judge_means_1_5 = {}
    judge_means_0_100 = {}
    layer_means = {}
    for layer in PERSONA_LAYERS:
        metric_means[layer] = {}
        for metric_id in RULE_METRICS[layer] + (MODEL_METRIC,):
            metric_means[layer][metric_id] = _mean(
                [
                    float(candidate["layers"][layer]["metrics"][metric_id]["score_0_100"])
                    for candidate in selected
                ]
            )
        judge_means_1_5[layer] = _mean(
            [
                float(candidate["layers"][layer]["metrics"][MODEL_METRIC]["score_1_5"])
                for candidate in selected
            ]
        )
        judge_means_0_100[layer] = metric_means[layer][MODEL_METRIC]
        layer_means[layer] = _mean(
            [float(candidate["layers"][layer]["score_0_100"]) for candidate in selected]
        )
    guard_summary = {}
    for dimension in GUARD_DIMENSIONS:
        guard_summary[dimension] = {
            "mean_score_1_5": _mean(
                [
                    float(candidate["guard_dimensions"][dimension]["score_1_5"])
                    for candidate in selected
                ]
            ),
            "mean_score_0_100": _mean(
                [
                    float(candidate["guard_dimensions"][dimension]["score_0_100"])
                    for candidate in selected
                ]
            ),
        }
        judge_means_1_5[dimension] = guard_summary[dimension]["mean_score_1_5"]
        judge_means_0_100[dimension] = guard_summary[dimension]["mean_score_0_100"]
    return {
        "variant": variant,
        "model_labels": labels,
        "responses": len(selected),
        "metric_means_0_100": metric_means,
        "judge_means_1_5": judge_means_1_5,
        "judge_means_0_100": judge_means_0_100,
        "layer_scores_0_100": layer_means,
        "persona_overall_score_0_100": _mean(
            [float(candidate["persona_overall_score_0_100"]) for candidate in selected]
        ),
        "guard_dimensions": guard_summary,
    }


def _delta_summary(base, lora):
    metric_delta = {}
    for layer in PERSONA_LAYERS:
        metric_delta[layer] = {
            metric_id: _rounded(
                lora["metric_means_0_100"][layer][metric_id]
                - base["metric_means_0_100"][layer][metric_id]
            )
            for metric_id in RULE_METRICS[layer] + (MODEL_METRIC,)
        }
    judge_delta_1_5 = {
        dimension: _rounded(
            lora["judge_means_1_5"][dimension] - base["judge_means_1_5"][dimension]
        )
        for dimension in SCORE_DIMENSIONS
    }
    judge_delta_0_100 = {
        dimension: _rounded(
            lora["judge_means_0_100"][dimension]
            - base["judge_means_0_100"][dimension]
        )
        for dimension in SCORE_DIMENSIONS
    }
    guard_delta = {
        dimension: {
            "mean_score_1_5": judge_delta_1_5[dimension],
            "mean_score_0_100": judge_delta_0_100[dimension],
        }
        for dimension in GUARD_DIMENSIONS
    }
    return {
        "direction": "LoRA - Base",
        "metric_means_0_100": metric_delta,
        "judge_means_1_5": judge_delta_1_5,
        "judge_means_0_100": judge_delta_0_100,
        "layer_scores_0_100": {
            layer: _rounded(
                lora["layer_scores_0_100"][layer]
                - base["layer_scores_0_100"][layer]
            )
            for layer in PERSONA_LAYERS
        },
        "persona_overall_score_0_100": _rounded(
            lora["persona_overall_score_0_100"]
            - base["persona_overall_score_0_100"]
        ),
        "guard_dimensions": guard_delta,
    }


def validate_and_score_judgments(records, key, config):
    """Validate filled requests, unblind them, and return Base/LoRA/Delta."""
    validate_eval_config(config)
    if not isinstance(key, dict) or key.get("schema_version") != JUDGE_KEY_SCHEMA_VERSION:
        raise AutomaticEvaluationError("unsupported judge key schema")
    _validate_key_contract(key, config)
    if not isinstance(records, list) or not records:
        raise AutomaticEvaluationError("filled judge records must be non-empty")
    expected_ids = set(key["rows"])
    records_by_id = {}
    for index, record in enumerate(records, 1):
        if not isinstance(record, dict):
            raise AutomaticEvaluationError("judge record %d must be an object" % index)
        request_id = record.get("request_id")
        if not isinstance(request_id, str) or not request_id:
            raise AutomaticEvaluationError("judge record %d has no request_id" % index)
        if request_id in records_by_id:
            raise AutomaticEvaluationError("duplicate judge request %s" % request_id)
        records_by_id[request_id] = record
    actual_ids = set(records_by_id)
    if actual_ids != expected_ids:
        raise AutomaticEvaluationError(
            "judge questions differ from key; missing=%r unexpected=%r"
            % (sorted(expected_ids - actual_ids), sorted(actual_ids - expected_ids))
        )

    item_scores = []
    all_candidates = []
    for request_id in sorted(expected_ids):
        key_row = key["rows"][request_id]
        judgment = _validate_request_result(
            records_by_id[request_id], request_id, key, config
        )
        candidates = {}
        for side_name in ("a", "b"):
            candidate = _candidate_item_score(
                key_row[side_name], judgment[side_name], config
            )
            display_name = "Base" if candidate["variant"] == "base" else "LoRA"
            if display_name in candidates:
                raise AutomaticEvaluationError(
                    "%s has duplicate %s side" % (request_id, display_name)
                )
            candidates[display_name] = candidate
            all_candidates.append(candidate)
        if set(candidates) != set(("Base", "LoRA")):
            raise AutomaticEvaluationError(
                "%s must unblind to one Base and one LoRA" % request_id
            )
        item_scores.append(
            {
                "request_id": request_id,
                "eval_id": key_row["eval_id"],
                "record_id": key_row["record_id"],
                "split": key_row["split"],
                "capability": key_row["capability"],
                "scenario_group": key_row["scenario_group"],
                "mode": key_row["mode"],
                "assistant_turn_index": key_row["assistant_turn_index"],
                "candidates": candidates,
            }
        )

    base = _aggregate_variant(all_candidates, "base")
    lora = _aggregate_variant(all_candidates, "lora")
    delta = _delta_summary(base, lora)
    before_after = {
        layer: {
            "Base": base["layer_scores_0_100"][layer],
            "LoRA": lora["layer_scores_0_100"][layer],
            "Delta": delta["layer_scores_0_100"][layer],
        }
        for layer in PERSONA_LAYERS
    }
    before_after["overall"] = {
        "Base": base["persona_overall_score_0_100"],
        "LoRA": lora["persona_overall_score_0_100"],
        "Delta": delta["persona_overall_score_0_100"],
    }
    return {
        "schema_version": SUMMARY_SCHEMA_VERSION,
        "evaluation_type": "automatic_rules_and_model_judge",
        "status": "ok",
        "requests_scored": len(item_scores),
        "responses_scored": len(all_candidates),
        "judge": copy.deepcopy(key["judge"]),
        "rubric_schema_version": RUBRIC_SCHEMA_VERSION,
        "rubric_sha256": rubric_sha256(),
        "config_schema_version": CONFIG_SCHEMA_VERSION,
        "config_sha256": canonical_json_sha256(config),
        "score_scales": copy.deepcopy(config["score_scales"]),
        "weights": {
            "layers": {
                layer: config["layers"][layer]["weight"] for layer in PERSONA_LAYERS
            },
            "metrics": {
                layer: {
                    metric_id: metric["weight"]
                    for metric_id, metric in config["layers"][layer]["metrics"].items()
                }
                for layer in PERSONA_LAYERS
            },
        },
        "Base": base,
        "LoRA": lora,
        "Delta": delta,
        "before_after": before_after,
        "items": item_scores,
    }


# Descriptive alias used by integrations that call the final step a summary.
score_judge_results = validate_and_score_judgments


def _ensure_distinct_paths(output_paths, input_paths):
    resolved_outputs = [Path(path).resolve() for path in output_paths]
    resolved_inputs = [Path(path).resolve() for path in input_paths if path is not None]
    if len(resolved_outputs) != len(set(resolved_outputs)):
        raise AutomaticEvaluationError("automatic-evaluation outputs must be distinct")
    for index, left in enumerate(resolved_outputs):
        for right in resolved_outputs[index + 1 :]:
            if left.exists() and right.exists() and os.path.samefile(str(left), str(right)):
                raise AutomaticEvaluationError(
                    "automatic-evaluation outputs resolve to the same existing file"
                )
    overlap = sorted(set(resolved_outputs).intersection(resolved_inputs))
    if overlap:
        raise AutomaticEvaluationError(
            "outputs must not overwrite inputs: %r" % [str(path) for path in overlap]
        )
    for output_path in resolved_outputs:
        for input_path in resolved_inputs:
            if (
                output_path.exists()
                and input_path.exists()
                and os.path.samefile(str(output_path), str(input_path))
            ):
                raise AutomaticEvaluationError(
                    "output is a hard-link alias of input: %s" % output_path
                )


def _prepare_command(args):
    config = load_eval_config(args.config)
    if args.judge_model and args.judge_model != config["judge"]["model"]:
        raise AutomaticEvaluationError(
            "--judge-model must match judge.model in the evaluation config"
        )
    if args.judge_revision and args.judge_revision != config["judge"]["revision"]:
        raise AutomaticEvaluationError(
            "--judge-revision must match judge.revision in the evaluation config"
        )
    comparisons = load_comparisons(args.comparisons)
    validate_generation_manifest(
        args.generation_manifest, args.comparisons, comparisons
    )
    requests, key = build_judge_requests(
        comparisons,
        config,
        judge_model=args.judge_model or config["judge"]["model"],
        judge_revision=args.judge_revision or config["judge"]["revision"],
        seed=args.seed,
    )
    key.update(
        {
            "comparison_file": str(Path(args.comparisons).resolve()),
            "comparison_file_sha256": file_sha256(args.comparisons),
            "generation_manifest_validated": True,
            "generation_manifest": str(Path(args.generation_manifest).resolve()),
            "generation_manifest_sha256": file_sha256(args.generation_manifest),
            "config_file": str(Path(args.config).resolve()),
            "config_file_sha256": file_sha256(args.config),
        }
    )
    _ensure_distinct_paths(
        (args.requests_jsonl, args.key_json),
        (args.comparisons, args.generation_manifest, args.config),
    )
    write_jsonl(requests, args.requests_jsonl)
    write_private_json(key, args.key_json)
    result = {
        "status": "ok",
        "requests": len(requests),
        "requests_jsonl": str(Path(args.requests_jsonl)),
        "key_json": str(Path(args.key_json)),
        "judge": key["judge"],
        "rubric_sha256": key["rubric_sha256"],
        "config_sha256": key["config_sha256"],
        "external_api_called": False,
    }
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    return result


def _score_command(args):
    config = load_eval_config(args.config)
    key = load_judge_key(
        args.key_json, require_provenance=True, config=config
    )
    records = load_judge_jsonl(args.judge_results)
    summary = validate_and_score_judgments(records, key, config)
    judge_audit = validate_deepseek_judge_audit(
        args.judge_audit,
        args.judge_results,
        records,
        key,
        config,
        args.key_json,
        args.config,
    )
    summary.update(
        {
            "judge_results_file": str(Path(args.judge_results).resolve()),
            "judge_results_file_sha256": file_sha256(args.judge_results),
            "judge_key_file": str(Path(args.key_json).resolve()),
            "judge_key_file_sha256": file_sha256(args.key_json),
            "config_file": str(Path(args.config).resolve()),
            "config_file_sha256": file_sha256(args.config),
            "comparison_file": key["comparison_file"],
            "comparison_file_sha256": key["comparison_file_sha256"],
            "generation_manifest_validated": True,
            "generation_manifest": key["generation_manifest"],
            "generation_manifest_sha256": key["generation_manifest_sha256"],
            "deepseek_judge_audit": judge_audit,
        }
    )
    _ensure_distinct_paths(
        (args.summary_json,),
        (
            args.judge_results,
            args.judge_audit,
            args.key_json,
            args.config,
            key["comparison_file"],
            key["generation_manifest"],
        ),
    )
    write_json(summary, args.summary_json)
    result = {
        "status": "ok",
        "requests_scored": summary["requests_scored"],
        "summary_json": str(Path(args.summary_json)),
        "Base": summary["Base"]["persona_overall_score_0_100"],
        "LoRA": summary["LoRA"]["persona_overall_score_0_100"],
        "Delta": summary["Delta"]["persona_overall_score_0_100"],
        "system_fingerprint": judge_audit["system_fingerprint"],
    }
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    return result


def build_argument_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command")

    prepare = subparsers.add_parser(
        "prepare",
        aliases=["build-requests"],
        help="create blind judge requests and a secret key; do not call an API",
    )
    prepare.add_argument("--comparisons", required=True, help="Base/LoRA comparison JSONL")
    prepare.add_argument(
        "--generation-manifest",
        required=True,
        help="generation manifest; always fully revalidated before requests are written",
    )
    prepare.add_argument("--config", default=str(DEFAULT_CONFIG_PATH))
    prepare.add_argument(
        "--judge-model",
        help="defaults to judge.model in the evaluation config",
    )
    prepare.add_argument(
        "--judge-revision",
        help="defaults to judge.revision in the evaluation config",
    )
    prepare.add_argument(
        "--seed",
        type=int,
        default=None,
        help="judge/blinding seed; defaults to judge.default_seed in the config",
    )
    prepare.add_argument(
        "--requests-jsonl", "--requests", dest="requests_jsonl", required=True
    )
    prepare.add_argument("--key-json", "--key", dest="key_json", required=True)
    prepare.set_defaults(handler=_prepare_command)

    score = subparsers.add_parser(
        "score",
        aliases=["summarize", "validate"],
        help="validate filled requests and write Base/LoRA/Delta summary",
    )
    score.add_argument(
        "--judge-results",
        "--filled-judge-jsonl",
        dest="judge_results",
        required=True,
    )
    score.add_argument(
        "--judge-audit",
        required=True,
        help="audit JSON emitted by scripts.module_d.run_deepseek_judge",
    )
    score.add_argument("--key-json", "--key", dest="key_json", required=True)
    score.add_argument("--config", default=str(DEFAULT_CONFIG_PATH))
    score.add_argument(
        "--summary-json", "--output", dest="summary_json", required=True
    )
    score.set_defaults(handler=_score_command)
    return parser


def main(argv=None):
    parser = build_argument_parser()
    args = parser.parse_args(argv)
    if not hasattr(args, "handler"):
        parser.error("a command is required: prepare/build-requests or score/summarize")
    return args.handler(args)


if __name__ == "__main__":
    main()
