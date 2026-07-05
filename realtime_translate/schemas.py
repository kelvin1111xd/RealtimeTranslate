from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field, HttpUrl


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


class JobStatus(str, Enum):
    queued = "queued"
    downloading = "downloading"
    transcribing = "transcribing"
    translating = "translating"
    exporting = "exporting"
    done = "done"
    failed = "failed"


class CreateJobRequest(BaseModel):
    youtubeUrl: HttpUrl
    sourceLanguage: str = "auto"
    targetLanguages: list[str] = Field(default_factory=lambda: ["zh-TW"])
    qualityMode: str = "quality"


class VideoJob(BaseModel):
    id: str
    youtubeUrl: str
    videoId: str
    title: str | None = None
    duration: float | None = None
    channel: str | None = None
    thumbnail: str | None = None
    audioPath: str | None = None
    status: JobStatus = JobStatus.queued
    progressPercent: int = 0
    progressStage: str = "queued"
    progressMessage: str = "Queued"
    progressDetail: dict[str, Any] = Field(default_factory=dict)
    sourceLanguage: str = "auto"
    detectedLanguage: str | None = None
    targetLanguages: list[str] = Field(default_factory=list)
    error: str | None = None
    createdAt: str = Field(default_factory=utc_now_iso)
    updatedAt: str = Field(default_factory=utc_now_iso)


class TranscriptSegment(BaseModel):
    id: str
    jobId: str
    index: int
    startMs: int
    endMs: int
    sourceText: str
    normalizedText: str | None = None
    speaker: str | None = None
    confidence: float | None = None


class TranslationSegment(BaseModel):
    id: str
    jobId: str
    segmentId: str
    targetLanguage: str
    translatedText: str
    polishedText: str | None = None
    model: str
    warnings: list[str] = Field(default_factory=list)


class SubtitleCue(BaseModel):
    startMs: int
    endMs: int
    text: str
    language: str


class GlossaryEntry(BaseModel):
    source: str
    target: str
    languages: list[str]
    caseSensitive: bool = False


class JobResponse(BaseModel):
    job: VideoJob
    links: dict[str, Any] = Field(default_factory=dict)


class RetranslateRequest(BaseModel):
    targetLanguages: list[str] | None = None
    provider: str | None = None
