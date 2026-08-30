#!/usr/bin/env python3
"""Export Trainer log_history to an auditable CSV and dependency-free SVG curve."""

from __future__ import annotations

import argparse
import csv
import html
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence

from scripts.module_c.common import ExperimentError, load_json, workspace_path


def extract_points(log_history: Sequence[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    points: List[Dict[str, Any]] = []
    for entry in log_history:
        step = entry.get("step")
        epoch = entry.get("epoch")
        if step is None:
            continue
        if "loss" in entry:
            points.append(
                {
                    "series": "train_loss",
                    "step": int(step),
                    "epoch": epoch,
                    "value": float(entry["loss"]),
                    "learning_rate": entry.get("learning_rate"),
                    "grad_norm": entry.get("grad_norm"),
                }
            )
        if "eval_loss" in entry:
            points.append(
                {
                    "series": "validation_loss",
                    "step": int(step),
                    "epoch": epoch,
                    "value": float(entry["eval_loss"]),
                    "learning_rate": entry.get("learning_rate"),
                    "grad_norm": entry.get("grad_norm"),
                }
            )
    if not points:
        raise ExperimentError("No train/eval loss points found in log_history")
    return points


def write_csv(path: Path, points: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=(
                "series",
                "step",
                "epoch",
                "value",
                "learning_rate",
                "grad_norm",
            ),
        )
        writer.writeheader()
        writer.writerows(points)


def _polyline(
    points: Sequence[Mapping[str, Any]],
    x_min: float,
    x_max: float,
    y_min: float,
    y_max: float,
    left: float,
    top: float,
    width: float,
    height: float,
) -> str:
    coordinates = []
    for point in points:
        x = left + (float(point["step"]) - x_min) / max(x_max - x_min, 1.0) * width
        y = (
            top
            + height
            - (float(point["value"]) - y_min) / max(y_max - y_min, 1e-12) * height
        )
        coordinates.append("{:.2f},{:.2f}".format(x, y))
    return " ".join(coordinates)


def write_svg(path: Path, points: Sequence[Mapping[str, Any]], title: str) -> None:
    width, height = 960, 540
    left, right, top, bottom = 80.0, 30.0, 55.0, 65.0
    plot_width = width - left - right
    plot_height = height - top - bottom
    x_values = [float(point["step"]) for point in points]
    y_values = [float(point["value"]) for point in points]
    x_min, x_max = min(x_values), max(x_values)
    y_min, y_max = min(y_values), max(y_values)
    margin = max((y_max - y_min) * 0.08, 0.01)
    y_min -= margin
    y_max += margin
    colours = {"train_loss": "#2f6fdd", "validation_loss": "#d45a3a"}
    fragments = [
        '<svg xmlns="http://www.w3.org/2000/svg" width="{}" height="{}" viewBox="0 0 {} {}">'.format(
            width, height, width, height
        ),
        '<rect width="100%" height="100%" fill="white"/>',
        '<text x="{}" y="32" font-family="sans-serif" font-size="22" font-weight="600">{}</text>'.format(
            left, html.escape(title)
        ),
        '<line x1="{0}" y1="{1}" x2="{0}" y2="{2}" stroke="#444"/>'.format(
            left, top, top + plot_height
        ),
        '<line x1="{0}" y1="{1}" x2="{2}" y2="{1}" stroke="#444"/>'.format(
            left, top + plot_height, left + plot_width
        ),
        '<text x="{}" y="{}" font-family="sans-serif" font-size="14">optimizer step</text>'.format(
            left + plot_width / 2 - 45, height - 18
        ),
        '<text transform="translate(22,{}) rotate(-90)" font-family="sans-serif" font-size="14">completion-only NLL</text>'.format(
            top + plot_height / 2 + 55
        ),
    ]
    for series in ("train_loss", "validation_loss"):
        series_points = sorted(
            (point for point in points if point["series"] == series),
            key=lambda point: point["step"],
        )
        if not series_points:
            continue
        coordinates = _polyline(
            series_points,
            x_min,
            x_max,
            y_min,
            y_max,
            left,
            top,
            plot_width,
            plot_height,
        )
        fragments.append(
            '<polyline fill="none" stroke="{}" stroke-width="2.5" points="{}"/>'.format(
                colours[series], coordinates
            )
        )
    fragments.extend(
        [
            '<text x="{}" y="{}" font-family="sans-serif" font-size="13">train loss</text>'.format(
                width - 190, 28
            ),
            '<line x1="{}" y1="24" x2="{}" y2="24" stroke="{}" stroke-width="3"/>'.format(
                width - 225, width - 198, colours["train_loss"]
            ),
            '<text x="{}" y="{}" font-family="sans-serif" font-size="13">validation loss</text>'.format(
                width - 190, 47
            ),
            '<line x1="{}" y1="43" x2="{}" y2="43" stroke="{}" stroke-width="3"/>'.format(
                width - 225, width - 198, colours["validation_loss"]
            ),
            "</svg>",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(fragments) + "\n", encoding="utf-8")


def export(
    log_path: Path, csv_path: Path, svg_path: Path, title: str
) -> List[Dict[str, Any]]:
    payload = load_json(log_path)
    history = payload.get("log_history")
    if not isinstance(history, list):
        raise ExperimentError("{} has no log_history array".format(log_path))
    points = extract_points(history)
    write_csv(csv_path, points)
    write_svg(svg_path, points, title)
    return points


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--log-history", required=True)
    parser.add_argument("--csv", required=True)
    parser.add_argument("--svg", required=True)
    parser.add_argument("--title", default="Hu Tao LoRA training curve")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    points = export(
        workspace_path(args.log_history),
        workspace_path(args.csv),
        workspace_path(args.svg),
        args.title,
    )
    print("Exported {} loss points".format(len(points)))


if __name__ == "__main__":
    main()
