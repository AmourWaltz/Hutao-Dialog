#!/usr/bin/env python3
"""Build deterministic, contextual prompt/completion views for Module C."""

from __future__ import annotations

import argparse
import os
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Tuple

from scripts.module_c.common import (
    ExperimentError,
    SPLITS,
    expand_assistant_turns,
    load_json,
    load_jsonl,
    sha256_file,
    summarize_derived,
    verify_sha256,
    workspace_path,
    write_json,
    write_jsonl,
)


DEFAULT_CONFIG = "configs/module_c/hutao_qwen3_1p7b_lora_bf16.json"


def build(config_path: Path) -> Dict[str, Any]:
    config = load_json(config_path)
    data_config = config.get("data")
    if not isinstance(data_config, dict):
        raise ExperimentError("Config is missing a data object")

    source_dir = workspace_path(data_config["source_dir"])
    derived_dir = workspace_path(data_config["derived_dir"])
    expected_hashes = data_config["expected_sha256"]
    expected_derived_hashes = data_config["expected_derived_sha256"]
    expected_source_counts = data_config["expected_source_records"]
    expected_derived_counts = data_config["expected_derived_examples"]

    source_ids = set()
    derived_ids = set()
    manifest_splits: Dict[str, Any] = {}
    prepared_outputs: List[Tuple[Path, Path]] = []

    derived_dir.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix="module-c-data-", dir=str(derived_dir.parent)
    ) as temporary_directory:
        temporary_root = Path(temporary_directory)
        for split in SPLITS:
            source_path = source_dir / "{}.jsonl".format(split)
            verify_sha256(source_path, expected_hashes[split])
            source_records = load_jsonl(source_path)
            if len(source_records) != expected_source_counts[split]:
                raise ExperimentError(
                    "{} has {} records; expected {}".format(
                        source_path, len(source_records), expected_source_counts[split],
                    )
                )

            examples: List[Dict[str, Any]] = []
            for record in source_records:
                record_id = record.get("id")
                if record_id in source_ids:
                    raise ExperimentError(
                        "Duplicate source id across splits: {}".format(record_id)
                    )
                source_ids.add(record_id)
                examples.extend(expand_assistant_turns(record, split))

            for example in examples:
                if example["id"] in derived_ids:
                    raise ExperimentError(
                        "Duplicate derived id: {}".format(example["id"])
                    )
                derived_ids.add(example["id"])

            if len(examples) != expected_derived_counts[split]:
                raise ExperimentError(
                    "Derived {} examples for {}; expected {}".format(
                        len(examples), split, expected_derived_counts[split]
                    )
                )

            output_path = derived_dir / "{}.jsonl".format(split)
            temporary_output = temporary_root / "{}.jsonl".format(split)
            write_jsonl(temporary_output, examples)
            derived_sha256 = sha256_file(temporary_output)
            registered_derived_sha256 = expected_derived_hashes[split]
            if derived_sha256 != registered_derived_sha256:
                raise ExperimentError(
                    "Derived {} SHA-256 is {}, expected {}".format(
                        split, derived_sha256, registered_derived_sha256
                    )
                )
            summary = summarize_derived(source_records, examples)
            summary.update(
                {
                    "source_file": str(
                        source_path.relative_to(source_dir.parent.parent)
                    ),
                    "source_sha256": sha256_file(source_path),
                    "derived_file": str(
                        output_path.relative_to(derived_dir.parent.parent)
                    ),
                    "derived_sha256": derived_sha256,
                }
            )
            manifest_splits[split] = summary
            prepared_outputs.append((temporary_output, output_path))

        manifest: Dict[str, Any] = {
            "schema_version": "1.0",
            "dataset_name": "hutao_persona_grounded_sft_turn_view",
            "source_dataset_version": "2.0",
            "construction": (
                "one_contextual_prompt_completion_per_supervised_assistant_turn"
            ),
            "ordering": "source_file_order_then_assistant_turn_order",
            "metadata_in_model_input": False,
            "splits": manifest_splits,
        }
        temporary_manifest = temporary_root / "manifest.json"
        write_json(temporary_manifest, manifest)
        registered_manifest_sha256 = data_config["expected_manifest_sha256"]
        if sha256_file(temporary_manifest) != registered_manifest_sha256:
            raise ExperimentError(
                "Derived manifest SHA-256 differs from the registered config"
            )

        derived_dir.mkdir(parents=True, exist_ok=True)
        for temporary_output, output_path in prepared_outputs:
            os.replace(str(temporary_output), str(output_path))
        os.replace(str(temporary_manifest), str(derived_dir / "manifest.json"))
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        default=DEFAULT_CONFIG,
        help="Workspace-relative or absolute experiment config path.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    manifest = build(workspace_path(args.config))
    counts = {split: manifest["splits"][split]["derived_examples"] for split in SPLITS}
    print("Prepared Module C data: {}".format(counts))


if __name__ == "__main__":
    main()
