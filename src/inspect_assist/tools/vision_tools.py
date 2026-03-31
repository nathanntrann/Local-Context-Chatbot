"""Vision tools — analyze and compare thermal inspection images using LLM vision."""

from __future__ import annotations

import json
from pathlib import Path

from inspect_assist.llm import ImageContent, LLMResponse, Message, Role
from inspect_assist.tools import ToolParam, tool

# Wired by app factory
_llm_provider = None
_dataset_adapter = None
_settings = None


def set_vision_deps(llm_provider, dataset_adapter, settings) -> None:
    global _llm_provider, _dataset_adapter, _settings
    _llm_provider = llm_provider
    _dataset_adapter = dataset_adapter
    _settings = settings


def _get_deps():
    if _llm_provider is None or _dataset_adapter is None or _settings is None:
        raise RuntimeError("Vision deps not initialized")
    return _llm_provider, _dataset_adapter, _settings


_ANALYZE_PROMPT = """You are an expert thermal inspection image analyst for industrial seal inspection.

Analyze this thermal image and provide:
1. What you observe in the image (thermal patterns, seal integrity, anomalies)
2. Whether this appears to be a PASS (good seal) or FAULT (defective seal)
3. Your confidence level (high/medium/low) and reasoning
4. Any notable features or concerns

The image is currently labeled as: {label}
Filename: {filename}

If your assessment disagrees with the label, explicitly flag this as a potential mislabel and explain why."""

_COMPARE_PROMPT = """You are an expert thermal inspection image analyst.

Compare these two thermal inspection images side by side:

Image 1: {file1} (labeled: {label1})
Image 2: {file2} (labeled: {label2})

Provide:
1. Key visual differences between the two images
2. Whether the labels seem correct for each
3. What distinguishes a PASS from a FAULT in these images
4. Any observations that could help operators identify issues"""


@tool(
    name="analyze_image",
    description=(
        "Analyze a single thermal inspection image using GPT-4o vision. "
        "Provides expert interpretation of thermal patterns, seal quality assessment, "
        "and flags potential mislabels. Use when a user asks about a specific image."
    ),
    params=[
        ToolParam(
            name="image_path",
            type="string",
            description="Path to the image, e.g. 'PASS/img_001.png' or a filename",
        ),
    ],
)
async def analyze_image(image_path: str) -> str:
    llm, dataset, settings = _get_deps()

    img_info = dataset.get_image_by_path(image_path)
    if img_info is None:
        img_info = dataset.get_image_by_name(image_path)
    if img_info is None:
        return json.dumps({"error": f"Image not found: {image_path}"})

    image_content = ImageContent.from_path(
        img_info.path, max_size_px=settings.vision_max_image_size_px
    )
    prompt = _ANALYZE_PROMPT.format(label=img_info.label, filename=img_info.filename)

    response: LLMResponse = await llm.chat(
        messages=[Message(role=Role.USER, content=prompt, images=[image_content])],
        temperature=0.2,
    )

    return json.dumps(
        {
            "image": img_info.to_dict(),
            "analysis": response.content,
            "tokens_used": response.usage,
        },
        indent=2,
    )


@tool(
    name="compare_images",
    description=(
        "Compare two thermal inspection images side by side using GPT-4o vision. "
        "Highlights differences, validates labels, and teaches what distinguishes "
        "PASS from FAULT. Use when users want to understand differences between images."
    ),
    params=[
        ToolParam(name="image_path_1", type="string", description="Path to first image"),
        ToolParam(name="image_path_2", type="string", description="Path to second image"),
    ],
)
async def compare_images(image_path_1: str, image_path_2: str) -> str:
    llm, dataset, settings = _get_deps()

    img1 = dataset.get_image_by_path(image_path_1) or dataset.get_image_by_name(image_path_1)
    img2 = dataset.get_image_by_path(image_path_2) or dataset.get_image_by_name(image_path_2)

    if not img1:
        return json.dumps({"error": f"Image not found: {image_path_1}"})
    if not img2:
        return json.dumps({"error": f"Image not found: {image_path_2}"})

    ic1 = ImageContent.from_path(img1.path, max_size_px=settings.vision_max_image_size_px)
    ic2 = ImageContent.from_path(img2.path, max_size_px=settings.vision_max_image_size_px)

    prompt = _COMPARE_PROMPT.format(
        file1=img1.filename, label1=img1.label,
        file2=img2.filename, label2=img2.label,
    )

    response: LLMResponse = await llm.chat(
        messages=[Message(role=Role.USER, content=prompt, images=[ic1, ic2])],
        temperature=0.2,
    )

    return json.dumps(
        {
            "image_1": img1.to_dict(),
            "image_2": img2.to_dict(),
            "comparison": response.content,
            "tokens_used": response.usage,
        },
        indent=2,
    )


@tool(
    name="find_suspicious_labels",
    description=(
        "Sample images from a label folder and use vision to flag potential mislabels. "
        "Useful for dataset quality assurance. Analyzes a batch of images and reports "
        "which ones might be incorrectly labeled."
    ),
    params=[
        ToolParam(name="label", type="string", description="Label to audit, e.g. 'PASS' or 'FAULT'"),
        ToolParam(
            name="sample_size",
            type="integer",
            description="Number of images to sample and check (default 8)",
            required=False,
        ),
    ],
)
async def find_suspicious_labels(label: str, sample_size: int = 8) -> str:
    llm, dataset, settings = _get_deps()

    samples = dataset.get_sample(label, sample_size)
    if not samples:
        return json.dumps({"error": f"No images found for label: {label}"})

    results = []
    for img_info in samples:
        image_content = ImageContent.from_path(
            img_info.path, max_size_px=settings.vision_max_image_size_px
        )
        prompt = (
            f"You are auditing thermal inspection image labels.\n"
            f"This image is labeled '{img_info.label}'.\n"
            f"Filename: {img_info.filename}\n\n"
            f"Briefly assess: does this label appear correct? "
            f"Reply with a JSON object: "
            f'{{"correct": true/false, "confidence": "high/medium/low", '
            f'"reason": "brief explanation"}}'
        )

        try:
            response: LLMResponse = await llm.chat(
                messages=[Message(role=Role.USER, content=prompt, images=[image_content])],
                temperature=0.1,
            )
            results.append({
                "filename": img_info.filename,
                "label": img_info.label,
                "assessment": response.content,
            })
        except Exception as e:
            results.append({
                "filename": img_info.filename,
                "label": img_info.label,
                "assessment": f"Error: {str(e)}",
            })

    return json.dumps({"label_audited": label, "samples_checked": len(results), "results": results}, indent=2)
