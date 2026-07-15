from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, Field


PROJECT_ROOT = Path(__file__).resolve().parent.parent


class ASRConfig(BaseModel):
    engine: str = "faster-whisper"
    model: str = "large-v3"
    fallback_model: str = "medium"
    device: str = "cuda"
    compute_type: str = "float16"
    fallback_compute_type: str = "int8_float16"
    vad_filter: bool = True
    beam_size: int = 5
    best_of: int = 5
    word_timestamps: bool = True
    condition_on_previous_text: bool = True


class YouTubeConfig(BaseModel):
    cookies_file: Path | None = None
    cookies_from_browser: str | None = None
    browser_profile: str | None = None
    js_runtimes: list[str] = Field(default_factory=lambda: ["node"])


class TranslationConfig(BaseModel):
    primary_provider: Literal["ollama", "openai_compatible", "passthrough"] = "ollama"
    ollama_base_url: str = "http://127.0.0.1:11434"
    openai_compatible_base_url: str = "http://127.0.0.1:8080/v1"
    openai_compatible_api_key: str = "local"
    model: str = "qwen3:8b"
    batch_enabled: bool = True
    batch_size: int = 12
    batch_token_limit: int = 5000
    semantic_context_enabled: bool = True
    semantic_context_max_chars: int = 1200
    context_previous_segments: int = 3
    context_next_segments: int = 1
    glossary_enabled: bool = True
    subtitle_style: str = "concise"
    request_timeout_seconds: float = 180
    memory_enabled: bool = True
    memory_max_items: int = 8
    name_memory_max_items: int = 30
    proofread_enabled: bool = True
    batch_proofread_enabled: bool = True
    proofread_only_low_confidence: bool = True
    topic_summary_enabled: bool = True
    topic_summary_interval_batches: int = 8
    topic_summary_max_chars: int = 500
    cache_enabled: bool = True
    cache_path: Path = Path("data/translation_cache.sqlite3")


class SubtitleConfig(BaseModel):
    formats: list[str] = Field(default_factory=lambda: ["srt", "vtt", "ass"])
    max_lines: int = 2
    zh_tw_chars_per_line: int = 18
    en_chars_per_line: int = 42
    ja_chars_per_line: int = 18
    min_cue_ms: int = 1200
    max_cue_ms: int = 6000


class ServerConfig(BaseModel):
    host: str = "127.0.0.1"
    port: int = 8765


class StorageConfig(BaseModel):
    data_dir: Path = Path("data")
    work_dir: Path = Path("work")


class AppConfig(BaseModel):
    asr: ASRConfig = Field(default_factory=ASRConfig)
    youtube: YouTubeConfig = Field(default_factory=YouTubeConfig)
    translation: TranslationConfig = Field(default_factory=TranslationConfig)
    subtitle: SubtitleConfig = Field(default_factory=SubtitleConfig)
    server: ServerConfig = Field(default_factory=ServerConfig)
    storage: StorageConfig = Field(default_factory=StorageConfig)


@lru_cache
def load_config(path: str = "config/app.yaml") -> AppConfig:
    config_path = Path(path)
    if not config_path.is_absolute():
        config_path = PROJECT_ROOT / config_path
    raw = yaml.safe_load(config_path.read_text(encoding="utf-8")) if config_path.exists() else {}
    config = AppConfig.model_validate(raw or {})
    config = resolve_config_paths(config)
    config.storage.data_dir.mkdir(parents=True, exist_ok=True)
    config.storage.work_dir.mkdir(parents=True, exist_ok=True)
    for child in ["audio", "transcripts", "translations", "subtitles"]:
        (config.storage.work_dir / child).mkdir(parents=True, exist_ok=True)
    return config


def resolve_config_paths(config: AppConfig) -> AppConfig:
    if not config.storage.data_dir.is_absolute():
        config.storage.data_dir = PROJECT_ROOT / config.storage.data_dir
    if not config.storage.work_dir.is_absolute():
        config.storage.work_dir = PROJECT_ROOT / config.storage.work_dir
    if config.youtube.cookies_file and not config.youtube.cookies_file.is_absolute():
        config.youtube.cookies_file = PROJECT_ROOT / config.youtube.cookies_file
    if not config.translation.cache_path.is_absolute():
        config.translation.cache_path = PROJECT_ROOT / config.translation.cache_path
    config.translation.cache_path.parent.mkdir(parents=True, exist_ok=True)
    return config
