"""Tests for core components — dataset adapter, knowledge engine, tool registry."""

import json
import tempfile
from pathlib import Path

import pytest

from inspect_assist.adapters.dataset import ImageDatasetAdapter
from inspect_assist.knowledge import KnowledgeEngine
from inspect_assist.tools import ToolDef, ToolParam, ToolRegistry


# --- Dataset Adapter Tests ---

class TestImageDatasetAdapter:
    def test_empty_directory(self, tmp_path: Path):
        adapter = ImageDatasetAdapter(tmp_path)
        summary = adapter.get_summary()
        assert summary.total_images == 0
        assert summary.pass_count == 0
        assert summary.fault_count == 0

    def test_nonexistent_directory(self, tmp_path: Path):
        adapter = ImageDatasetAdapter(tmp_path / "nonexistent")
        summary = adapter.get_summary()
        assert summary.total_images == 0

    def test_counts_images(self, tmp_path: Path):
        pass_dir = tmp_path / "PASS"
        fault_dir = tmp_path / "FAULT"
        pass_dir.mkdir()
        fault_dir.mkdir()

        for i in range(5):
            (pass_dir / f"img_{i:03d}.png").write_bytes(b"fake png")
        for i in range(3):
            (fault_dir / f"img_{i:03d}.png").write_bytes(b"fake png")

        adapter = ImageDatasetAdapter(tmp_path)
        summary = adapter.get_summary()
        assert summary.total_images == 8
        assert summary.pass_count == 5
        assert summary.fault_count == 3
        assert abs(summary.pass_ratio - 5 / 8) < 0.01

    def test_ignores_non_image_files(self, tmp_path: Path):
        pass_dir = tmp_path / "PASS"
        pass_dir.mkdir()
        (pass_dir / "img_001.png").write_bytes(b"fake")
        (pass_dir / "readme.txt").write_text("not an image")
        (pass_dir / "data.csv").write_text("1,2,3")

        adapter = ImageDatasetAdapter(tmp_path)
        assert adapter.get_summary().pass_count == 1

    def test_get_images_by_label(self, tmp_path: Path):
        pass_dir = tmp_path / "PASS"
        pass_dir.mkdir()
        (pass_dir / "img_001.png").write_bytes(b"fake")
        (pass_dir / "img_002.png").write_bytes(b"fake")

        adapter = ImageDatasetAdapter(tmp_path)
        images = adapter.get_images("PASS")
        assert len(images) == 2
        assert all(img.label == "PASS" for img in images)

    def test_get_image_by_name(self, tmp_path: Path):
        pass_dir = tmp_path / "PASS"
        pass_dir.mkdir()
        (pass_dir / "target.png").write_bytes(b"fake")

        adapter = ImageDatasetAdapter(tmp_path)
        img = adapter.get_image_by_name("target.png")
        assert img is not None
        assert img.filename == "target.png"

    def test_get_image_by_path(self, tmp_path: Path):
        fault_dir = tmp_path / "FAULT"
        fault_dir.mkdir()
        (fault_dir / "defect_001.png").write_bytes(b"fake")

        adapter = ImageDatasetAdapter(tmp_path)
        img = adapter.get_image_by_path("FAULT/defect_001.png")
        assert img is not None
        assert img.label == "FAULT"

    def test_cache_invalidation(self, tmp_path: Path):
        pass_dir = tmp_path / "PASS"
        pass_dir.mkdir()
        (pass_dir / "img_001.png").write_bytes(b"fake")

        adapter = ImageDatasetAdapter(tmp_path)
        assert adapter.get_summary().pass_count == 1

        (pass_dir / "img_002.png").write_bytes(b"fake")
        # Still cached
        assert adapter.get_summary().pass_count == 1
        # After invalidation
        adapter.invalidate_cache()
        assert adapter.get_summary().pass_count == 2


# --- Knowledge Engine Tests ---

class TestKnowledgeEngine:
    def _make_article(self, path: Path, title: str, category: str, tags: list[str], content: str):
        tag_str = ", ".join(tags)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            f"---\ntitle: {title}\ncategory: {category}\ntags: [{tag_str}]\n---\n\n{content}",
            encoding="utf-8",
        )

    def test_empty_directory(self, tmp_path: Path):
        engine = KnowledgeEngine(tmp_path)
        assert engine.search("anything") == []

    def test_nonexistent_directory(self, tmp_path: Path):
        engine = KnowledgeEngine(tmp_path / "nope")
        assert engine.list_all() == []

    def test_loads_articles(self, tmp_path: Path):
        self._make_article(
            tmp_path / "test.md",
            "Test Article", "concepts", ["test", "demo"],
            "This is test content about thermal seals.",
        )
        engine = KnowledgeEngine(tmp_path)
        articles = engine.list_all()
        assert len(articles) == 1
        assert articles[0]["title"] == "Test Article"

    def test_search_by_title(self, tmp_path: Path):
        self._make_article(
            tmp_path / "thresholds.md",
            "Understanding Thresholds", "parameters", ["threshold"],
            "Content about threshold parameters.",
        )
        self._make_article(
            tmp_path / "calibration.md",
            "Calibration Guide", "procedures", ["calibration"],
            "How to calibrate the system.",
        )

        engine = KnowledgeEngine(tmp_path)
        results = engine.search("threshold")
        assert len(results) >= 1
        assert results[0].title == "Understanding Thresholds"

    def test_search_by_tag(self, tmp_path: Path):
        self._make_article(
            tmp_path / "fp.md",
            "False Positives", "troubleshooting", ["false-positive", "rejection"],
            "Troubleshooting guide for false positives.",
        )
        engine = KnowledgeEngine(tmp_path)
        results = engine.search("false-positive")
        assert len(results) >= 1

    def test_get_by_slug(self, tmp_path: Path):
        self._make_article(
            tmp_path / "my-article.md",
            "My Article", "concepts", ["test"],
            "Content here.",
        )
        engine = KnowledgeEngine(tmp_path)
        article = engine.get_by_slug("my-article")
        assert article is not None
        assert article.title == "My Article"

    def test_get_by_category(self, tmp_path: Path):
        self._make_article(tmp_path / "a.md", "A", "troubleshooting", [], "content")
        self._make_article(tmp_path / "b.md", "B", "troubleshooting", [], "content")
        self._make_article(tmp_path / "c.md", "C", "concepts", [], "content")

        engine = KnowledgeEngine(tmp_path)
        results = engine.get_by_category("troubleshooting")
        assert len(results) == 2


# --- Tool Registry Tests ---

class TestToolRegistry:
    def test_register_and_get(self):
        registry = ToolRegistry()
        td = ToolDef(name="test_tool", description="A test tool")
        registry.register(td)
        assert registry.get("test_tool") is not None
        assert registry.get("nonexistent") is None

    def test_openai_schemas(self):
        registry = ToolRegistry()
        td = ToolDef(
            name="my_tool",
            description="Does something",
            params=[
                ToolParam(name="query", type="string", description="The query"),
                ToolParam(name="limit", type="integer", description="Max results", required=False),
            ],
        )
        registry.register(td)
        schemas = registry.openai_schemas()
        assert len(schemas) == 1
        schema = schemas[0]
        assert schema["type"] == "function"
        assert schema["function"]["name"] == "my_tool"
        assert "query" in schema["function"]["parameters"]["properties"]
        assert "query" in schema["function"]["parameters"]["required"]
        assert "limit" not in schema["function"]["parameters"]["required"]

    @pytest.mark.asyncio
    async def test_call_tool(self):
        async def echo_handler(text: str) -> str:
            return json.dumps({"echo": text})

        registry = ToolRegistry()
        td = ToolDef(
            name="echo",
            description="Echoes text",
            params=[ToolParam(name="text", type="string", description="Text to echo")],
            handler=echo_handler,
        )
        registry.register(td)

        result = await registry.call("echo", '{"text": "hello"}')
        data = json.loads(result)
        assert data["echo"] == "hello"

    @pytest.mark.asyncio
    async def test_call_unknown_tool(self):
        registry = ToolRegistry()
        result = await registry.call("nonexistent", "{}")
        data = json.loads(result)
        assert "error" in data
