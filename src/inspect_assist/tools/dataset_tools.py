"""Dataset tools — scan, summarize, and sample the image dataset."""

from __future__ import annotations

import json
from dataclasses import asdict

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
