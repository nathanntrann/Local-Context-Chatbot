"""Image dataset adapter — scans PASS/FAULT folder structure."""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from pathlib import Path

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


@dataclass
class DatasetSummary:
    total_images: int
    pass_count: int
    fault_count: int
    pass_ratio: float
    fault_ratio: float
    labels: list[str]
    path: str


@dataclass
class ImageInfo:
    path: Path
    label: str
    filename: str
    size_bytes: int

    def to_dict(self) -> dict:
        return {
            "path": str(self.path),
            "label": self.label,
            "filename": self.filename,
            "size_bytes": self.size_bytes,
        }


class ImageDatasetAdapter:
    """Reads a directory of labeled images organized as label-named subdirectories."""

    def __init__(self, dataset_path: Path) -> None:
        self._root = dataset_path
        self._cache: dict[str, list[ImageInfo]] | None = None

    def _scan(self) -> dict[str, list[ImageInfo]]:
        """Scan the dataset directory for label folders and image files."""
        if self._cache is not None:
            return self._cache

        result: dict[str, list[ImageInfo]] = {}
        if not self._root.exists():
            self._cache = result
            return result

        for label_dir in sorted(self._root.iterdir()):
            if not label_dir.is_dir():
                continue
            label = label_dir.name
            images: list[ImageInfo] = []
            for f in sorted(label_dir.iterdir()):
                if f.is_file() and f.suffix.lower() in IMAGE_EXTENSIONS:
                    images.append(
                        ImageInfo(
                            path=f,
                            label=label,
                            filename=f.name,
                            size_bytes=f.stat().st_size,
                        )
                    )
            if images:
                result[label] = images

        self._cache = result
        return result

    def invalidate_cache(self) -> None:
        self._cache = None

    def get_summary(self) -> DatasetSummary:
        data = self._scan()
        pass_count = len(data.get("PASS", []))
        fault_count = len(data.get("FAULT", []))
        total = pass_count + fault_count
        # Also count any other labels
        for label, imgs in data.items():
            if label not in ("PASS", "FAULT"):
                total += len(imgs)

        return DatasetSummary(
            total_images=total,
            pass_count=pass_count,
            fault_count=fault_count,
            pass_ratio=pass_count / total if total > 0 else 0.0,
            fault_ratio=fault_count / total if total > 0 else 0.0,
            labels=list(data.keys()),
            path=str(self._root),
        )

    def get_images(self, label: str | None = None) -> list[ImageInfo]:
        data = self._scan()
        if label:
            return data.get(label, [])
        all_images: list[ImageInfo] = []
        for imgs in data.values():
            all_images.extend(imgs)
        return all_images

    def get_sample(self, label: str, count: int = 8) -> list[ImageInfo]:
        images = self.get_images(label)
        if len(images) <= count:
            return images
        return random.sample(images, count)

    def get_image_by_name(self, filename: str) -> ImageInfo | None:
        for imgs in self._scan().values():
            for img in imgs:
                if img.filename == filename:
                    return img
        return None

    def get_image_by_path(self, path_str: str) -> ImageInfo | None:
        """Find image by relative or absolute path."""
        target = Path(path_str)
        for imgs in self._scan().values():
            for img in imgs:
                if img.path == target or img.path.name == target.name:
                    return img
                # Check relative path like "PASS/img_001.png"
                try:
                    rel = img.path.relative_to(self._root)
                    if str(rel) == path_str:
                        return img
                except ValueError:
                    continue
        return None
