from __future__ import annotations

import html
import re
from pathlib import Path

from .config import SubtitleConfig
from .schemas import SubtitleCue, TranscriptSegment, TranslationSegment


def build_cues(
    segments: list[TranscriptSegment],
    translations: list[TranslationSegment],
    language: str,
    config: SubtitleConfig,
) -> list[SubtitleCue]:
    by_segment = {translation.segmentId: translation for translation in translations}
    cues: list[SubtitleCue] = []
    for segment in segments:
        translation = by_segment.get(segment.id)
        if not translation:
            continue
        text = translation.polishedText or translation.translatedText
        text = normalize_subtitle_text(text)
        if not text or re.fullmatch(r"[\W_]+", text):
            continue
        lines = wrap_text(text, language, config)
        cues.append(
            SubtitleCue(
                startMs=segment.startMs,
                endMs=segment.endMs,
                text="\n".join(lines),
                language=language,
            )
        )
    return cues


def normalize_subtitle_text(text: str) -> str:
    text = re.sub(r"\s+", " ", text).strip()
    text = text.replace("。 ", "。").replace("、 ", "、")
    return text


def wrap_text(text: str, language: str, config: SubtitleConfig) -> list[str]:
    limit = {
        "zh-TW": config.zh_tw_chars_per_line,
        "zh": config.zh_tw_chars_per_line,
        "ja": config.ja_chars_per_line,
        "en": config.en_chars_per_line,
    }.get(language, config.en_chars_per_line)
    if language.startswith("en"):
        words = text.split()
        lines: list[str] = []
        current = ""
        for word in words:
            candidate = f"{current} {word}".strip()
            if len(candidate) <= limit:
                current = candidate
            else:
                if current:
                    lines.append(current)
                current = word
        if current:
            lines.append(current)
    else:
        chunks = split_cjk_text(text, limit)
        lines = chunks
    if len(lines) <= config.max_lines:
        return lines
    return compact_lines(lines, config.max_lines, limit, cjk=not language.startswith("en"))


def split_cjk_text(text: str, limit: int) -> list[str]:
    hard_breakpoints = "，、。！？!?；;：:"
    soft_breakpoints = "はがをにへでとものやかねよわ"
    lines: list[str] = []
    remaining = text.strip()
    while len(remaining) > limit:
        window_end = min(len(remaining), limit + 6)
        search_window = remaining[:window_end]
        split_at = -1
        for index, char in enumerate(search_window):
            if char in hard_breakpoints:
                split_at = index + 1
        if split_at < max(1, limit - 8):
            for index in range(min(len(search_window), limit + 4) - 1, max(0, limit - 8) - 1, -1):
                if search_window[index] in soft_breakpoints:
                    split_at = index + 1
                    break
        if split_at < max(1, limit - 8):
            split_at = avoid_katakana_split(remaining, limit)
        lines.append(remaining[:split_at].strip())
        remaining = remaining[split_at:].strip()
    if remaining:
        lines.append(remaining)
    return lines


def avoid_katakana_split(text: str, limit: int) -> int:
    split_at = min(limit, len(text))
    katakana = re.compile(r"[\u30a0-\u30ffー]")
    if split_at < len(text) and katakana.match(text[split_at - 1]) and katakana.match(text[split_at]):
        cursor = split_at
        while cursor < min(len(text), limit + 8) and katakana.match(text[cursor]):
            cursor += 1
        if cursor < len(text):
            return cursor
        cursor = split_at
        while cursor > max(1, limit - 8) and katakana.match(text[cursor - 1]):
            cursor -= 1
        return max(1, cursor)
    return split_at


def compact_lines(lines: list[str], max_lines: int, limit: int, *, cjk: bool = False) -> list[str]:
    text = "".join(lines) if cjk else " ".join(lines)
    if max_lines <= 1:
        return [text[: limit * 2]]
    target = max(1, len(text) // max_lines)
    result = []
    cursor = 0
    for index in range(max_lines):
        if index == max_lines - 1:
            result.append(text[cursor:].strip())
        else:
            result.append(text[cursor : cursor + target].strip())
            cursor += target
    return [line for line in result if line]


def split_long_cues(cues: list[SubtitleCue], config: SubtitleConfig) -> list[SubtitleCue]:
    result: list[SubtitleCue] = []
    for cue in cues:
        duration = cue.endMs - cue.startMs
        if duration <= config.max_cue_ms:
            result.append(cue)
            continue
        parts = [part.strip() for part in re.split(r"(?<=[。！？.!?])", cue.text) if part.strip()]
        if len(parts) <= 1:
            result.append(cue)
            continue
        part_ms = duration // len(parts)
        cursor = cue.startMs
        for part in parts:
            end = min(cue.endMs, cursor + part_ms)
            result.append(SubtitleCue(startMs=cursor, endMs=end, text=part, language=cue.language))
            cursor = end
    return result


def ms_to_srt(ms: int) -> str:
    hours, remainder = divmod(ms, 3_600_000)
    minutes, remainder = divmod(remainder, 60_000)
    seconds, millis = divmod(remainder, 1000)
    return f"{hours:02}:{minutes:02}:{seconds:02},{millis:03}"


def ms_to_vtt(ms: int) -> str:
    return ms_to_srt(ms).replace(",", ".")


def ms_to_ass(ms: int) -> str:
    centis = ms // 10
    hours, remainder = divmod(centis, 360_000)
    minutes, remainder = divmod(remainder, 6_000)
    seconds, centiseconds = divmod(remainder, 100)
    return f"{hours}:{minutes:02}:{seconds:02}.{centiseconds:02}"


def export_srt(cues: list[SubtitleCue]) -> str:
    blocks = []
    for index, cue in enumerate(cues, 1):
        blocks.append(f"{index}\n{ms_to_srt(cue.startMs)} --> {ms_to_srt(cue.endMs)}\n{cue.text}")
    return "\n\n".join(blocks) + ("\n" if blocks else "")


def export_vtt(cues: list[SubtitleCue]) -> str:
    blocks = ["WEBVTT\n"]
    for cue in cues:
        blocks.append(f"{ms_to_vtt(cue.startMs)} --> {ms_to_vtt(cue.endMs)}\n{cue.text}\n")
    return "\n".join(blocks)


def export_ass(cues: list[SubtitleCue]) -> str:
    header = """[Script Info]
ScriptType: v4.00+
PlayResX: 1920
PlayResY: 1080

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: Default,Microsoft JhengHei,56,&H00FFFFFF,&H00000000,&H80000000,0,0,0,0,100,100,0,0,1,3,1,2,80,80,70,1

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text"""
    events = []
    for cue in cues:
        text = html.escape(cue.text).replace("\n", r"\N")
        events.append(
            f"Dialogue: 0,{ms_to_ass(cue.startMs)},{ms_to_ass(cue.endMs)},Default,,0,0,0,,{text}"
        )
    return header + "\n" + "\n".join(events) + ("\n" if events else "")


def write_subtitles(output_dir: Path, video_id: str, language: str, cues: list[SubtitleCue], formats: list[str]) -> dict[str, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    writers = {"srt": export_srt, "vtt": export_vtt, "ass": export_ass}
    paths: dict[str, Path] = {}
    for fmt in formats:
        writer = writers.get(fmt)
        if not writer:
            continue
        path = output_dir / f"{video_id}.{language}.{fmt}"
        path.write_text(writer(cues), encoding="utf-8")
        paths[fmt] = path
    return paths
