"""Application settings loaded from environment / .env file."""

from __future__ import annotations

from enum import Enum
from pathlib import Path
from typing import Optional

from pydantic_settings import BaseSettings


class LLMProvider(str, Enum):
    AZURE_OPENAI = "azure_openai"
    OPENAI = "openai"
    OLLAMA = "ollama"


class Settings(BaseSettings):
    model_config = {"env_file": ".env", "env_file_encoding": "utf-8", "extra": "ignore"}

    # --- LLM ---
    llm_provider: LLMProvider = LLMProvider.OLLAMA

    # Azure OpenAI
    azure_openai_endpoint: str = ""
    azure_openai_api_key: str = ""
    azure_openai_deployment: str = "gpt-4o"
    azure_openai_api_version: str = "2024-10-21"

    # OpenAI
    openai_api_key: str = ""
    openai_model: str = "gpt-4o"

    # Ollama
    ollama_base_url: str = "http://localhost:11434/v1"
    ollama_model: str = "llama3.1:8b"

    # --- App ---
    app_host: str = "0.0.0.0"
    app_port: int = 8000
    log_level: str = "INFO"
    api_key: str = ""  # If set, requires X-API-Key header on all /api/ routes
    rate_limit_per_minute: int = 30  # Max requests per minute on chat endpoints (0 = unlimited)

    # --- Data ---
    dataset_path: Path = Path("./data/images")
    knowledge_path: Path = Path("./knowledge")

    # --- Limits ---
    max_conversation_turns: int = 50
    max_tool_calls_per_turn: int = 5
    vision_max_image_size_px: int = 1024
    vision_sample_size: int = 8

    @property
    def dataset_pass_dir(self) -> Path:
        return self.dataset_path / "PASS"

    @property
    def dataset_fault_dir(self) -> Path:
        return self.dataset_path / "FAULT"


_settings: Optional[Settings] = None


def get_settings() -> Settings:
    global _settings
    if _settings is None:
        _settings = Settings()
    return _settings
