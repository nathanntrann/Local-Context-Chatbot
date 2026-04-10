"""FastAPI application factory — wires all components together."""

from __future__ import annotations

import time
from collections import defaultdict
from pathlib import Path

import structlog
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from inspect_assist.adapters.dataset import ImageDatasetAdapter
from inspect_assist.api.routes import router
from inspect_assist.config import get_settings
from inspect_assist.knowledge import KnowledgeEngine
from inspect_assist.llm.providers import create_llm_provider, create_provider_for
from inspect_assist.orchestrator import Orchestrator
from inspect_assist.storage import ConversationStore
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

    # Allow cross-origin requests so the widget can be embedded in other apps
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # --- API key authentication ---
    api_key = settings.api_key
    if api_key:
        @app.middleware("http")
        async def api_key_auth(request: Request, call_next):
            if request.url.path.startswith("/api/"):
                key = request.headers.get("X-API-Key", "")
                if key != api_key:
                    return JSONResponse(status_code=401, content={"detail": "Invalid or missing API key"})
            return await call_next(request)

        logger.info("api_key_auth_enabled")

    # --- Rate limiting on chat endpoints ---
    rate_limit = settings.rate_limit_per_minute
    if rate_limit > 0:
        _rate_buckets: dict[str, list[float]] = defaultdict(list)

        @app.middleware("http")
        async def rate_limiter(request: Request, call_next):
            if request.url.path.startswith("/api/v1/chat"):
                client_ip = request.client.host if request.client else "unknown"
                now = time.monotonic()
                window = 60.0
                hits = _rate_buckets[client_ip]
                # Prune old entries
                _rate_buckets[client_ip] = [t for t in hits if now - t < window]
                if len(_rate_buckets[client_ip]) >= rate_limit:
                    return JSONResponse(
                        status_code=429,
                        content={"detail": f"Rate limit exceeded ({rate_limit}/min). Try again shortly."},
                    )
                _rate_buckets[client_ip].append(now)
            return await call_next(request)

        logger.info("rate_limiting_enabled", limit=rate_limit)

    # --- Initialize components ---
    llm = create_llm_provider(settings)
    dataset_adapter = ImageDatasetAdapter(settings.dataset_path)
    knowledge_engine = KnowledgeEngine(
        settings.knowledge_path,
        vectorstore_path=Path(settings.dataset_path).parent / "vectorstore",
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
        parent_chunk_size=settings.parent_chunk_size,
        parent_chunk_overlap=settings.parent_chunk_overlap,
        embed_model=settings.embedding_model,
        contextual_retrieval=settings.contextual_retrieval_enabled,
        hybrid_search=settings.hybrid_search_enabled,
        rrf_k=settings.rrf_k,
        reranker_enabled=settings.reranker_enabled,
        reranker_type=settings.reranker_type,
        reranker_model=settings.reranker_model,
        rerank_top_n=settings.rerank_top_n,
        hyde_enabled=settings.hyde_enabled,
        cache_enabled=settings.semantic_cache_enabled,
        cache_similarity_threshold=settings.cache_similarity_threshold,
        cache_ttl_seconds=settings.cache_ttl_seconds,
        cache_max_size=settings.cache_max_size,
        max_context_tokens=settings.max_context_tokens,
    )

    # Wire embedding client for semantic search (uses same OpenAI client)
    knowledge_engine.set_embed_client(llm._client)

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

    # Build conversation store
    db_path = Path(settings.dataset_path).parent / "conversations.db"
    conversation_store = ConversationStore(str(db_path))

    # Build smart-routing providers (if enabled)
    llm_fast = None
    routing_enabled = settings.routing_enabled
    if routing_enabled:
        llm_fast = create_provider_for(settings.fast_provider, settings.fast_model, settings)
        llm_strong = create_provider_for(settings.strong_provider, settings.strong_model, settings)
        logger.info(
            "smart_routing_enabled",
            fast=f"{settings.fast_provider.value}/{settings.fast_model}",
            strong=f"{settings.strong_provider.value}/{settings.strong_model}",
        )
    else:
        llm_strong = llm

    # Build orchestrator
    orchestrator = Orchestrator(
        llm=llm_strong,
        tool_registry=registry,
        max_turns=settings.max_conversation_turns,
        max_tool_calls_per_turn=settings.max_tool_calls_per_turn,
        store=conversation_store,
        llm_fast=llm_fast,
        routing_enabled=routing_enabled,
    )

    # Store on app state for route access
    app.state.orchestrator = orchestrator
    app.state.tool_registry = registry
    app.state.settings = settings
    app.state.conversation_store = conversation_store

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

    # Widget demo — shows how the embed looks in a host app
    @app.get("/widget-demo", response_class=HTMLResponse, tags=["ui"])
    async def widget_demo(request: Request):
        return templates.TemplateResponse(request, "widget_demo.html")

    # Build embeddings for knowledge base on startup
    @app.on_event("startup")
    async def _build_knowledge_embeddings():
        await knowledge_engine.build_embeddings()

    logger.info(
        "app_started",
        provider=settings.llm_provider.value,
        dataset=str(settings.dataset_path),
        knowledge=str(settings.knowledge_path),
    )

    return app
