"""FastAPI application factory — wires all components together."""

from __future__ import annotations

from pathlib import Path

import structlog
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from inspect_assist.adapters.dataset import ImageDatasetAdapter
from inspect_assist.api.routes import router
from inspect_assist.config import get_settings
from inspect_assist.knowledge import KnowledgeEngine
from inspect_assist.llm.providers import create_llm_provider
from inspect_assist.orchestrator import Orchestrator
from inspect_assist.tools import ToolRegistry
from inspect_assist.tools import dataset_tools, knowledge_tools, vision_tools


def create_app() -> FastAPI:
    settings = get_settings()

    # Configure structured logging
    import logging

    log_level = getattr(logging, settings.log_level.upper(), logging.INFO)
    structlog.configure(
        processors=[
            structlog.contextvars.merge_contextvars,
            structlog.processors.add_log_level,
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.dev.ConsoleRenderer(),
        ],
        wrapper_class=structlog.make_filtering_bound_logger(log_level),
    )
    logger = structlog.get_logger()

    app = FastAPI(
        title="InspectAssist",
        description="Context-aware LLM assistant for industrial thermal inspection",
        version="0.1.0",
    )

    # --- Initialize components ---
    llm = create_llm_provider(settings)
    dataset_adapter = ImageDatasetAdapter(settings.dataset_path)
    knowledge_engine = KnowledgeEngine(settings.knowledge_path)

    # Wire tool dependencies
    dataset_tools.set_dataset_adapter(dataset_adapter)
    vision_tools.set_vision_deps(llm, dataset_adapter, settings)
    knowledge_tools.set_knowledge_engine(knowledge_engine)

    # Build tool registry
    registry = ToolRegistry()
    registry.register_module(dataset_tools)
    registry.register_module(vision_tools)
    registry.register_module(knowledge_tools)

    logger.info("tools_registered", tools=[t.name for t in registry.all_tools])

    # Build orchestrator
    orchestrator = Orchestrator(
        llm=llm,
        tool_registry=registry,
        max_turns=settings.max_conversation_turns,
        max_tool_calls_per_turn=settings.max_tool_calls_per_turn,
    )

    # Store on app state for route access
    app.state.orchestrator = orchestrator
    app.state.tool_registry = registry
    app.state.settings = settings

    # --- Templates & static files ---
    templates_dir = Path(__file__).parent / "templates"
    static_dir = Path(__file__).parent / "static"
    templates_dir.mkdir(exist_ok=True)
    static_dir.mkdir(exist_ok=True)

    app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")
    templates = Jinja2Templates(directory=str(templates_dir))
    app.state.templates = templates

    # --- Routes ---
    app.include_router(router)

    # Chat UI route
    @app.get("/", response_class=HTMLResponse, tags=["ui"])
    async def chat_ui(request: Request):
        return templates.TemplateResponse(request, "chat.html")

    logger.info(
        "app_started",
        provider=settings.llm_provider.value,
        dataset=str(settings.dataset_path),
        knowledge=str(settings.knowledge_path),
    )

    return app
