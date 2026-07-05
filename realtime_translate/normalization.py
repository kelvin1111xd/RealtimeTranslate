from __future__ import annotations

import re
from uuid import uuid4

from .schemas import TranscriptSegment


FILLER_PATTERNS = [
    re.compile(r"\b(um|uh|er|ah)\b", re.IGNORECASE),
    re.compile(r"(えーと|えっと|あのー|そのー)"),
]


def normalize_segments(segments: list[TranscriptSegment]) -> list[TranscriptSegment]:
    cleaned: list[TranscriptSegment] = []
    for segment in segments:
        text = segment.sourceText.strip()
        for pattern in FILLER_PATTERNS:
            text = pattern.sub("", text)
        text = re.sub(r"\s+", " ", text).strip()
        text = re.sub(r"([。！？!?]){2,}", r"\1", text)
        if text:
            segment.normalizedText = text
            cleaned.append(segment)

    merged = merge_short_segments(cleaned)
    return split_long_segments(merged)


def merge_short_segments(segments: list[TranscriptSegment]) -> list[TranscriptSegment]:
    if not segments:
        return []
    result: list[TranscriptSegment] = []
    buffer = segments[0].model_copy(deep=True)
    for segment in segments[1:]:
        duration = buffer.endMs - buffer.startMs
        text = buffer.normalizedText or buffer.sourceText
        next_text = segment.normalizedText or segment.sourceText
        should_merge = duration < 5000 or len(text) < 18
        would_be_too_long = (segment.endMs - buffer.startMs) > 20000 or len(text + next_text) > 220
        if should_merge and not would_be_too_long:
            buffer.endMs = segment.endMs
            buffer.sourceText = f"{buffer.sourceText} {segment.sourceText}".strip()
            buffer.normalizedText = f"{text} {next_text}".strip()
        else:
            result.append(buffer)
            buffer = segment.model_copy(deep=True)
    result.append(buffer)
    for index, segment in enumerate(result):
        segment.index = index
    return result


def split_long_segments(segments: list[TranscriptSegment]) -> list[TranscriptSegment]:
    result: list[TranscriptSegment] = []
    for segment in segments:
        text = segment.normalizedText or segment.sourceText
        duration = segment.endMs - segment.startMs
        if duration <= 20000 and len(text) <= 220:
            result.append(segment)
            continue
        parts = re.split(r"(?<=[。！？.!?])\s*", text)
        parts = [p for p in parts if p]
        if len(parts) <= 1:
            parts = [text[i : i + 90] for i in range(0, len(text), 90)]
        total_chars = sum(len(p) for p in parts) or 1
        cursor = segment.startMs
        for part in parts:
            part_duration = max(1200, round(duration * len(part) / total_chars))
            child = segment.model_copy(deep=True)
            child.id = str(uuid4())
            child.startMs = cursor
            child.endMs = min(segment.endMs, cursor + part_duration)
            child.sourceText = part
            child.normalizedText = part
            result.append(child)
            cursor = child.endMs
    for index, segment in enumerate(result):
        segment.index = index
    return result

