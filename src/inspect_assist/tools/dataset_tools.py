"""Dataset tools — scan, summarize, and sample the image dataset."""

from __future__ import annotations

import json
import statistics
from dataclasses import asdict

from PIL import Image

from inspect_assist.tools import ToolParam, tool

# These will be wired to real instances by the app factory
_dataset_adapter = None


def set_dataset_adapter(adapter) -> None:
    global _dataset_adapter
    _dataset_adapter = adapter


def _get_adapter():
    if _dataset_adapter is None:
        raise RuntimeError("Dataset adapter not initialized")
    return _dataset_adapter


@tool(
    name="get_dataset_summary",
    description=(
        "Get a summary of the inspection image dataset including total images, "
        "count per label (PASS/FAULT), class balance ratios, and available labels. "
        "Use this when the user asks about their dataset, data distribution, or class balance."
    ),
    params=[],
)
async def get_dataset_summary() -> str:
    adapter = _get_adapter()
    summary = adapter.get_summary()
    return json.dumps(asdict(summary), indent=2)


@tool(
    name="get_sample_images",
    description=(
        "Get a random sample of image filenames from a specific label folder. "
        "Use this to show the user example images or to pick images for analysis."
    ),
    params=[
        ToolParam(name="label", type="string", description="Label folder name, e.g. 'PASS' or 'FAULT'"),
        ToolParam(name="count", type="integer", description="Number of samples to return (default 8)", required=False),
    ],
)
async def get_sample_images(label: str, count: int = 8) -> str:
    adapter = _get_adapter()
    samples = adapter.get_sample(label, count)
    return json.dumps([s.to_dict() for s in samples], indent=2)


@tool(
    name="get_dataset_statistics",
    description=(
        "Get detailed statistics about the inspection image dataset including "
        "image dimensions (width/height), file sizes, per-label breakdowns, and "
        "class balance metrics. Use when the user asks for deeper analysis of "
        "their data quality, image properties, or dataset health."
    ),
    params=[],
)
async def get_dataset_statistics() -> str:
    adapter = _get_adapter()
    all_images = adapter.get_images()

    if not all_images:
        return json.dumps({"error": "No images found in dataset"})

    # Per-label breakdown
    labels: dict[str, list] = {}
    for img in all_images:
        labels.setdefault(img.label, []).append(img)

    # Gather image dimensions and file sizes
    widths: list[int] = []
    heights: list[int] = []
    sizes: list[int] = [img.size_bytes for img in all_images]

    for img in all_images:
        try:
            with Image.open(img.path) as pil_img:
                w, h = pil_img.size
                widths.append(w)
                heights.append(h)
        except Exception:
            continue  # skip unreadable images

    def _stats(values: list[int | float]) -> dict:
        if not values:
            return {}
        return {
            "min": min(values),
            "max": max(values),
            "mean": round(statistics.mean(values), 1),
            "median": round(statistics.median(values), 1),
            "stdev": round(statistics.stdev(values), 1) if len(values) > 1 else 0.0,
        }

    # Class balance analysis
    label_counts = {lbl: len(imgs) for lbl, imgs in sorted(labels.items())}
    total = sum(label_counts.values())
    max_count = max(label_counts.values())
    min_count = min(label_counts.values())
    imbalance_ratio = round(max_count / min_count, 2) if min_count > 0 else float("inf")

    per_label: list[dict] = []
    for lbl, imgs in sorted(labels.items()):
        lbl_sizes = [i.size_bytes for i in imgs]
        per_label.append({
            "label": lbl,
            "count": len(imgs),
            "percentage": round(len(imgs) / total * 100, 1),
            "file_size": _stats(lbl_sizes),
        })

    result = {
        "total_images": total,
        "labels": per_label,
        "class_balance": {
            "imbalance_ratio": imbalance_ratio,
            "balanced": imbalance_ratio < 1.5,
            "recommendation": (
                "Dataset is well-balanced."
                if imbalance_ratio < 1.5
                else f"Dataset is imbalanced ({imbalance_ratio}:1). "
                "Consider augmenting the minority class or collecting more samples."
            ),
        },
        "image_dimensions": {
            "width": _stats(widths),
            "height": _stats(heights),
            "images_measured": len(widths),
        },
        "file_size_bytes": _stats(sizes),
        "total_size_mb": round(sum(sizes) / (1024 * 1024), 2),
    }

    return json.dumps(result, indent=2)
