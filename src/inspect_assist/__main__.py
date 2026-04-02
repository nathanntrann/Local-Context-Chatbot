"""Entry point for running with `python -m inspect_assist` or uvicorn."""

import argparse
import asyncio
import json
import sys
from pathlib import Path

import uvicorn

from inspect_assist.config import get_settings


def main() -> None:
    settings = get_settings()
    app = create_app()
    uvicorn.run(app, host=settings.app_host, port=settings.app_port)


def create_app():
    from inspect_assist.app import create_app as _create_app
    return _create_app()


async def batch_audit(labels: list[str], sample_size: int, output: str | None) -> None:
    """Run find_suspicious_labels on the given labels and write results."""
    settings = get_settings()

    from inspect_assist.adapters.dataset import ImageDatasetAdapter
    from inspect_assist.llm.providers import create_llm_provider
    from inspect_assist.tools import vision_tools

    llm = create_llm_provider(settings)
    dataset = ImageDatasetAdapter(settings.dataset_path)
    vision_tools.set_vision_deps(llm, dataset, settings)

    summary = dataset.get_summary()
    print(f"Dataset: {summary.total_images} images ({summary.pass_count} PASS, {summary.fault_count} FAULT)")

    if not labels:
        labels = summary.labels or ["PASS", "FAULT"]

    all_results = {}
    for label in labels:
        print(f"\nAuditing {label} (sample_size={sample_size})...")
        raw = await vision_tools.find_suspicious_labels(label=label, sample_size=sample_size)
        result = json.loads(raw)
        all_results[label] = result

        if "error" in result:
            print(f"  Error: {result['error']}")
            continue

        suspicious = 0
        for r in result.get("results", []):
            assessment = r.get("assessment", "")
            flag = "⚠ SUSPICIOUS" if '"correct": false' in assessment.lower() or '"correct":false' in assessment.lower() else "  OK"
            if flag.startswith("⚠"):
                suspicious += 1
            print(f"  {flag}  {r['filename']}: {assessment[:80]}")

        print(f"  {result['samples_checked']} checked, {suspicious} suspicious")

    report = {
        "batch_audit": True,
        "labels_audited": labels,
        "sample_size": sample_size,
        "results": all_results,
    }

    if output:
        out_path = Path(output)
    else:
        out_path = Path(settings.dataset_path).parent / "reports" / "batch_audit.json"

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2))
    print(f"\nReport saved to {out_path}")


def cli() -> None:
    parser = argparse.ArgumentParser(prog="inspect_assist", description="InspectAssist thermal inspection assistant")
    sub = parser.add_subparsers(dest="command")

    # Default: run server
    sub.add_parser("serve", help="Start the InspectAssist API server (default)")

    # Batch audit
    batch_parser = sub.add_parser("batch", help="Run batch mislabel audit on dataset")
    batch_parser.add_argument("--labels", nargs="*", default=[], help="Labels to audit (default: all)")
    batch_parser.add_argument("--sample-size", type=int, default=8, help="Images to sample per label (default: 8)")
    batch_parser.add_argument("--output", type=str, default=None, help="Output JSON path (default: data/reports/batch_audit.json)")

    args = parser.parse_args()

    if args.command == "batch":
        asyncio.run(batch_audit(args.labels, args.sample_size, args.output))
    else:
        main()


if __name__ == "__main__":
    cli()
