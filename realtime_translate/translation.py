from __future__ import annotations

import json
import hashlib
import re
import sqlite3
from abc import ABC, abstractmethod
from collections.abc import Callable
from uuid import uuid4

import httpx
from pydantic import BaseModel, Field

from .config import TranslationConfig
from .schemas import GlossaryEntry, TranscriptSegment, TranslationSegment


class TranslationInput(BaseModel):
    sourceLanguage: str
    targetLanguage: str
    currentText: str
    previousContext: list[str]
    nextContext: list[str]
    glossary: list[GlossaryEntry] = Field(default_factory=list)
    memoryNotes: list[str] = Field(default_factory=list)
    nameMemory: list[str] = Field(default_factory=list)
    recentTranslations: list[str] = Field(default_factory=list)
    topicSummary: str = ""
    draftTranslation: str | None = None
    styleGuide: str | None = None


class TranslationOutput(BaseModel):
    translatedText: str
    warnings: list[str] = Field(default_factory=list)


class BatchTranslationSegment(BaseModel):
    id: str
    text: str


class TranslationBatchInput(BaseModel):
    sourceLanguage: str
    targetLanguage: str
    segments: list[BatchTranslationSegment]
    previousContext: list[str]
    nextContext: list[str]
    glossary: list[GlossaryEntry] = Field(default_factory=list)
    memoryNotes: list[str] = Field(default_factory=list)
    nameMemory: list[str] = Field(default_factory=list)
    recentTranslations: list[str] = Field(default_factory=list)
    topicSummary: str = ""
    draftTranslations: dict[str, str] = Field(default_factory=dict)
    styleGuide: str | None = None


class TranslationBatchOutput(BaseModel):
    translations: dict[str, str]
    warnings: list[str] = Field(default_factory=list)


class TranslationMemory(BaseModel):
    notes: list[str] = Field(default_factory=list)
    names: list[str] = Field(default_factory=list)
    recent_translations: list[str] = Field(default_factory=list)
    topic_summary: str = ""


class TranslationProvider(ABC):
    name: str
    supportedLanguages: list[str]

    @abstractmethod
    def translate(self, input: TranslationInput) -> TranslationOutput:
        raise NotImplementedError

    def translate_batch(self, input: TranslationBatchInput) -> TranslationBatchOutput:
        translations = {}
        warnings = []
        for segment in input.segments:
            output = self.translate(
                TranslationInput(
                    sourceLanguage=input.sourceLanguage,
                    targetLanguage=input.targetLanguage,
                    currentText=segment.text,
                    previousContext=input.previousContext,
                    nextContext=input.nextContext,
                    glossary=input.glossary,
                    memoryNotes=input.memoryNotes,
                    nameMemory=input.nameMemory,
                    recentTranslations=input.recentTranslations,
                    topicSummary=input.topicSummary,
                    draftTranslation=input.draftTranslations.get(segment.id),
                    styleGuide=input.styleGuide,
                )
            )
            translations[segment.id] = output.translatedText
            warnings.extend(output.warnings)
        return TranslationBatchOutput(translations=translations, warnings=warnings)

    def summarize_memory(self, memory: TranslationMemory, max_chars: int) -> str:
        return ""


class PassthroughProvider(TranslationProvider):
    name = "passthrough"
    supportedLanguages = ["zh-TW", "en", "ja"]

    def translate(self, input: TranslationInput) -> TranslationOutput:
        return TranslationOutput(
            translatedText=input.currentText,
            warnings=["passthrough provider did not translate text"],
        )

    def translate_batch(self, input: TranslationBatchInput) -> TranslationBatchOutput:
        return TranslationBatchOutput(
            translations={segment.id: segment.text for segment in input.segments},
            warnings=["passthrough provider did not translate text"],
        )


class OllamaProvider(TranslationProvider):
    name = "ollama"
    supportedLanguages = ["zh-TW", "en", "ja"]

    def __init__(self, config: TranslationConfig):
        self.config = config
        self.client = httpx.Client(timeout=config.request_timeout_seconds)

    def _post(self, payload: dict) -> dict:
        response = self.client.post(
            f"{self.config.ollama_base_url.rstrip('/')}/api/chat",
            json=payload,
        )
        response.raise_for_status()
        return response.json()

    def translate(self, input: TranslationInput) -> TranslationOutput:
        prompt = build_prompt(input)
        if self.config.model.startswith("qwen3"):
            prompt = f"/no_think\n{prompt}"
        response = self._post(
                {
                    "model": self.config.model,
                    "messages": [
                        {
                            "role": "system",
                            "content": (
                                "You are a subtitle translation engine. "
                                "Return only the translation of CURRENT_SEGMENT. "
                                "Never translate context, never explain, never add markup. "
                                "Use memory only for consistency."
                            ),
                        },
                        {"role": "user", "content": prompt},
                    ],
                    "stream": False,
                    "keep_alive": "10m",
                    "options": {
                        "temperature": 0.0,
                        "top_p": 0.8,
                        "repeat_penalty": 1.08,
                        "num_ctx": 8192,
                    },
                }
            )
        text = response.get("message", {}).get("content", "").strip()
        return TranslationOutput(translatedText=clean_model_output(text))

    def translate_batch(self, input: TranslationBatchInput) -> TranslationBatchOutput:
        prompt = build_batch_prompt(input)
        if self.config.model.startswith("qwen3"):
            prompt = f"/no_think\n{prompt}"
        response = self._post(
                {
                    "model": self.config.model,
                    "messages": [
                        {
                            "role": "system",
                            "content": (
                                "You are a subtitle translation engine. "
                                "Return only a JSON array. "
                                "Every input id must appear exactly once. "
                                "Use memory only for consistency."
                            ),
                        },
                        {"role": "user", "content": prompt},
                    ],
                    "stream": False,
                    "keep_alive": "10m",
                    "format": "json",
                    "options": {
                        "temperature": 0.0,
                        "top_p": 0.8,
                        "repeat_penalty": 1.08,
                        "num_ctx": 8192,
                    },
                }
            )
        text = response.get("message", {}).get("content", "").strip()
        return parse_batch_translation_output(text, [segment.id for segment in input.segments])

    def summarize_memory(self, memory: TranslationMemory, max_chars: int) -> str:
        prompt = build_memory_summary_prompt(memory, max_chars)
        response = self._post(
            {
                "model": self.config.model,
                "messages": [
                    {
                        "role": "system",
                        "content": "Summarize subtitle context only. Return plain text, no bullets.",
                    },
                    {"role": "user", "content": prompt},
                ],
                "stream": False,
                "keep_alive": "10m",
                "options": {"temperature": 0.0, "num_ctx": 4096},
            }
        )
        return compact_text(response.get("message", {}).get("content", ""), max_chars)


class OpenAICompatibleProvider(TranslationProvider):
    name = "openai_compatible"
    supportedLanguages = ["zh-TW", "en", "ja"]

    def __init__(self, config: TranslationConfig):
        self.config = config
        self.client = httpx.Client(timeout=config.request_timeout_seconds)

    def _post(self, payload: dict) -> dict:
        response = self.client.post(
            f"{self.config.openai_compatible_base_url.rstrip('/')}/chat/completions",
            headers={"Authorization": f"Bearer {self.config.openai_compatible_api_key}"},
            json=payload,
        )
        response.raise_for_status()
        return response.json()

    def translate(self, input: TranslationInput) -> TranslationOutput:
        prompt = build_prompt(input)
        response = self._post(
                {
                    "model": self.config.model,
                    "messages": [
                        {
                            "role": "system",
                            "content": (
                                "You are a professional subtitle translator. "
                                "Return only the final subtitle text."
                            ),
                        },
                        {"role": "user", "content": prompt},
                    ],
                    "temperature": 0.1,
                }
            )
        text = response["choices"][0]["message"]["content"].strip()
        return TranslationOutput(translatedText=clean_model_output(text))

    def translate_batch(self, input: TranslationBatchInput) -> TranslationBatchOutput:
        prompt = build_batch_prompt(input)
        response = self._post(
                {
                    "model": self.config.model,
                    "messages": [
                        {
                            "role": "system",
                            "content": (
                                "You are a professional subtitle translator. "
                                "Return only a JSON array with id and translation fields."
                            ),
                        },
                        {"role": "user", "content": prompt},
                    ],
                    "temperature": 0.1,
                }
            )
        text = response["choices"][0]["message"]["content"].strip()
        return parse_batch_translation_output(text, [segment.id for segment in input.segments])


def build_provider(config: TranslationConfig, override: str | None = None) -> TranslationProvider:
    provider = override or config.primary_provider
    if provider == "ollama":
        return OllamaProvider(config)
    if provider == "openai_compatible":
        return OpenAICompatibleProvider(config)
    if provider == "passthrough":
        return PassthroughProvider()
    raise ValueError(f"Unknown translation provider: {provider}")


def build_prompt(input: TranslationInput) -> str:
    if input.styleGuide == "proofread":
        return build_proofread_prompt(input)
    glossary = "\n".join(
        f"- {entry.source} => {entry.target}"
        for entry in input.glossary
        if input.targetLanguage in entry.languages
    )
    previous = "\n".join(input.previousContext) or "(none)"
    next_context = "\n".join(input.nextContext) or "(none)"
    memory_notes = "\n".join(f"- {note}" for note in input.memoryNotes) or "(none)"
    name_memory = "\n".join(f"- {name}" for name in input.nameMemory) or "(none)"
    recent_translations = "\n".join(f"- {text}" for text in input.recentTranslations) or "(none)"
    topic_summary = input.topicSummary or "(none)"
    if input.targetLanguage == "zh-TW":
        if input.styleGuide == "strict_retry":
            task = """你是台灣繁體中文字幕翻譯器。只翻譯 <CURRENT_SEGMENT> 內的文字。

硬性規則：
- 只輸出台灣繁體中文譯文。
- 不要輸出英文單字、羅馬拼音、簡體中文、說明、標籤或引號。
- 除非是人名、歌名、頻道名，否則不要保留日文假名。
- 原文如果像直播口語、點歌或閒聊，請用自然台灣中文翻譯，不要自行補充。
- 不確定的專有名詞可音譯或保留原文，但不要亂翻成無關詞。
- 輸出必須簡短，適合字幕。"""
        else:
            task = """Translate ONLY the text inside <CURRENT_SEGMENT> into Traditional Chinese subtitles.

Hard rules:
- Output Traditional Chinese used in Taiwan.
- Output ONLY the translated subtitle text.
- Do NOT translate <PREVIOUS_CONTEXT> or <NEXT_CONTEXT>; they are reference only.
- Do NOT copy Japanese kana unless it is a proper noun that must stay Japanese.
- Do NOT mix in English unless it is an unavoidable product name or song title.
- Do NOT add explanations, quotes, labels, markdown, or extra sentences.
- Keep names and song titles consistent with the glossary.
- Keep names, recurring topics, and pronouns consistent with <MEMORY>.
- Keep it concise and natural for subtitles."""
    elif input.targetLanguage == "ja":
        task = """Translate ONLY the text inside <CURRENT_SEGMENT> into natural Japanese subtitles.
Use context only as reference. Output only Japanese text."""
    else:
        task = """Translate ONLY the text inside <CURRENT_SEGMENT> into natural English subtitles.
Use context only as reference. Output only the translated subtitle text."""
    return f"""{task}

Source language: {input.sourceLanguage}
Target language: {input.targetLanguage}

<PREVIOUS_CONTEXT>
{previous}
</PREVIOUS_CONTEXT>

<CURRENT_SEGMENT>
{input.currentText}
</CURRENT_SEGMENT>

<NEXT_CONTEXT>
{next_context}
</NEXT_CONTEXT>

<MEMORY>
Names and recurring terms:
{name_memory}

Recent topic notes:
{memory_notes}

Episode topic summary:
{topic_summary}

Recent approved translations:
{recent_translations}
</MEMORY>

<GLOSSARY>
{glossary or "(none)"}
</GLOSSARY>

Translation:"""


def build_batch_prompt(input: TranslationBatchInput) -> str:
    if input.styleGuide == "proofread":
        return build_batch_proofread_prompt(input)
    glossary = "\n".join(
        f"- {entry.source} => {entry.target}"
        for entry in input.glossary
        if input.targetLanguage in entry.languages
    )
    previous = "\n".join(input.previousContext) or "(none)"
    next_context = "\n".join(input.nextContext) or "(none)"
    memory_notes = "\n".join(f"- {note}" for note in input.memoryNotes) or "(none)"
    name_memory = "\n".join(f"- {name}" for name in input.nameMemory) or "(none)"
    recent_translations = "\n".join(f"- {text}" for text in input.recentTranslations) or "(none)"
    topic_summary = input.topicSummary or "(none)"
    segments_json = json.dumps(
        [{"id": segment.id, "text": segment.text} for segment in input.segments],
        ensure_ascii=False,
        indent=2,
    )
    language_rules = build_language_rules(input.targetLanguage)
    return f"""You are a professional subtitle translator.

Task:
Translate every item in <SEGMENTS_TO_TRANSLATE> into {input.targetLanguage} subtitles.

Hard rules:
- Return ONLY valid JSON. No markdown, no explanation, no code fence.
- Return a JSON array with exactly {len(input.segments)} objects.
- Each object must be: {{"id": "...", "translation": "..."}}
- Preserve every input id exactly.
- Do not merge, split, reorder, omit, or add segments.
- Translate only the segment text. Context and memory are reference only.
- Keep translations concise, natural, and suitable for subtitles.
- Use glossary and memory for consistent names, works, pronouns, and recurring topics.
- Do not add information that is not in the source.
{language_rules}

Source language: {input.sourceLanguage}
Target language: {input.targetLanguage}

<PREVIOUS_CONTEXT>
{previous}
</PREVIOUS_CONTEXT>

<SEGMENTS_TO_TRANSLATE>
{segments_json}
</SEGMENTS_TO_TRANSLATE>

<NEXT_CONTEXT>
{next_context}
</NEXT_CONTEXT>

<MEMORY>
Names and recurring terms:
{name_memory}

Recent topic notes:
{memory_notes}

Episode topic summary:
{topic_summary}

Recent approved translations:
{recent_translations}
</MEMORY>

<GLOSSARY>
{glossary or "(none)"}
</GLOSSARY>

JSON:"""


def build_batch_proofread_prompt(input: TranslationBatchInput) -> str:
    glossary = "\n".join(
        f"- {entry.source} => {entry.target}"
        for entry in input.glossary
        if input.targetLanguage in entry.languages
    )
    memory_notes = "\n".join(f"- {note}" for note in input.memoryNotes) or "(none)"
    name_memory = "\n".join(f"- {name}" for name in input.nameMemory) or "(none)"
    recent_translations = "\n".join(f"- {text}" for text in input.recentTranslations) or "(none)"
    topic_summary = input.topicSummary or "(none)"
    drafts_json = json.dumps(
        [
            {
                "id": segment.id,
                "source": segment.text,
                "draft_translation": input.draftTranslations.get(segment.id, ""),
            }
            for segment in input.segments
        ],
        ensure_ascii=False,
        indent=2,
    )
    language_rules = build_language_rules(input.targetLanguage)
    return f"""You are a professional subtitle proofreader.

Task:
Proofread each draft translation against its source segment.

Hard rules:
- Return ONLY valid JSON. No markdown, no explanation, no code fence.
- Return a JSON array with exactly {len(input.segments)} objects.
- Each object must be: {{"id": "...", "translation": "..."}}
- Preserve every input id exactly.
- If a draft is already natural and accurate, return it unchanged.
- Fix unnatural literal translation, wrong pronouns, inconsistent names, and subtitle awkwardness.
- Do not merge, split, reorder, omit, or add segments.
- Do not add information that is not in the source.
{language_rules}

Source language: {input.sourceLanguage}
Target language: {input.targetLanguage}

<DRAFTS_TO_PROOFREAD>
{drafts_json}
</DRAFTS_TO_PROOFREAD>

<MEMORY>
Names and recurring terms:
{name_memory}

Recent topic notes:
{memory_notes}

Episode topic summary:
{topic_summary}

Recent approved translations:
{recent_translations}
</MEMORY>

<GLOSSARY>
{glossary or "(none)"}
</GLOSSARY>

JSON:"""


def build_language_rules(target_language: str) -> str:
    if target_language == "zh-TW":
        return """- Output Traditional Chinese used in Taiwan.
- Avoid Simplified Chinese.
- Do not keep Japanese kana unless it is a proper noun that must stay Japanese.
- Do not mix in English unless it is an unavoidable name, product, song title, or glossary term."""
    if target_language == "ja":
        return "- Output natural Japanese subtitle text."
    return "- Output natural English subtitle text."


def build_memory_summary_prompt(memory: TranslationMemory, max_chars: int) -> str:
    notes = "\n".join(f"- {item}" for item in memory.notes[-12:]) or "(none)"
    names = "\n".join(f"- {item}" for item in memory.names[-30:]) or "(none)"
    recent = "\n".join(f"- {item}" for item in memory.recent_translations[-8:]) or "(none)"
    return f"""Create a compact factual context note for future subtitle translation.
Keep it under {max_chars} characters. Mention only recurring topics, entities, relationships,
or terminology that help resolve later references. Do not invent facts.

Names and terms:
{names}

Recent translation notes:
{notes}

Recent approved translations:
{recent}

Context note:"""


def build_proofread_prompt(input: TranslationInput) -> str:
    glossary = "\n".join(
        f"- {entry.source} => {entry.target}"
        for entry in input.glossary
        if input.targetLanguage in entry.languages
    )
    memory_notes = "\n".join(f"- {note}" for note in input.memoryNotes) or "(none)"
    name_memory = "\n".join(f"- {name}" for name in input.nameMemory) or "(none)"
    recent_translations = "\n".join(f"- {text}" for text in input.recentTranslations) or "(none)"
    topic_summary = input.topicSummary or "(none)"
    if input.targetLanguage == "zh-TW":
        task = """你是台灣繁體中文字幕校對。請根據原文檢查初稿是否自然、準確、簡潔。

規則：
- 只輸出修正後的繁體中文字幕。
- 不要輸出解釋、標籤、引號、Markdown 或多餘文字。
- 如果初稿已經自然準確，就原樣輸出。
- 修正不自然的直譯、錯誤代名詞、前後不一致的人名/作品名。
- 避免簡體中文、英文雜訊、日文假名殘留。
- 不要新增原文沒有的資訊。"""
    elif input.targetLanguage == "ja":
        task = """Proofread the draft as natural Japanese subtitles.
Output only the corrected Japanese subtitle text. If the draft is already good, output it unchanged."""
    else:
        task = """Proofread the draft as natural English subtitles.
Output only the corrected English subtitle text. If the draft is already good, output it unchanged."""
    return f"""{task}

Source language: {input.sourceLanguage}
Target language: {input.targetLanguage}

<SOURCE_SEGMENT>
{input.currentText}
</SOURCE_SEGMENT>

<DRAFT_TRANSLATION>
{input.draftTranslation or ""}
</DRAFT_TRANSLATION>

<MEMORY>
Names and recurring terms:
{name_memory}

Recent topic notes:
{memory_notes}

Episode topic summary:
{topic_summary}

Recent approved translations:
{recent_translations}
</MEMORY>

<GLOSSARY>
{glossary or "(none)"}
</GLOSSARY>

Corrected subtitle:"""


def clean_model_output(text: str) -> str:
    text = text.strip().strip('"').strip()
    text = re.sub(r"^(Translation|翻譯|訳)[:：]\s*", "", text, flags=re.IGNORECASE)
    return text.strip()


def parse_batch_translation_output(text: str, expected_ids: list[str]) -> TranslationBatchOutput:
    raw = text.strip()
    raw = re.sub(r"^```(?:json)?\s*", "", raw, flags=re.IGNORECASE)
    raw = re.sub(r"\s*```$", "", raw)
    if not raw.startswith("["):
        start = raw.find("[")
        end = raw.rfind("]")
        if start >= 0 and end > start:
            raw = raw[start : end + 1]
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ValueError("translation batch response was not valid JSON") from exc
    if not isinstance(parsed, list):
        raise ValueError("translation batch response must be a JSON array")
    translations: dict[str, str] = {}
    for item in parsed:
        if not isinstance(item, dict):
            raise ValueError("translation batch item must be an object")
        segment_id = str(item.get("id", "")).strip()
        translation = str(item.get("translation", "")).strip()
        if not segment_id or segment_id not in expected_ids:
            raise ValueError(f"translation batch returned unexpected id: {segment_id}")
        translations[segment_id] = clean_model_output(translation)
    missing_ids = [segment_id for segment_id in expected_ids if segment_id not in translations]
    if missing_ids:
        raise ValueError(f"translation batch missed segment ids: {', '.join(missing_ids)}")
    return TranslationBatchOutput(translations=translations)


class TranslationCache:
    def __init__(self, path):
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with sqlite3.connect(self.path) as conn:
            conn.execute(
                """CREATE TABLE IF NOT EXISTS translation_cache (
                   cache_key TEXT PRIMARY KEY,
                   translated_text TEXT NOT NULL,
                   model TEXT NOT NULL,
                   created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
                )"""
            )

    def get(self, key: str) -> str | None:
        with sqlite3.connect(self.path) as conn:
            row = conn.execute(
                "SELECT translated_text FROM translation_cache WHERE cache_key = ?", (key,)
            ).fetchone()
        return row[0] if row else None

    def put(self, key: str, translated_text: str, model: str) -> None:
        with sqlite3.connect(self.path) as conn:
            conn.execute(
                """INSERT INTO translation_cache(cache_key, translated_text, model)
                   VALUES (?, ?, ?)
                   ON CONFLICT(cache_key) DO UPDATE SET
                     translated_text = excluded.translated_text,
                     model = excluded.model,
                     created_at = CURRENT_TIMESTAMP""",
                (key, translated_text, model),
            )


def translation_cache_key(
    *, source_text: str, source_language: str, target_language: str,
    glossary: list[GlossaryEntry], config: TranslationConfig,
) -> str:
    payload = json.dumps(
        {
            "prompt_version": "batch-memory-v2",
            "source": source_text,
            "source_language": source_language,
            "target_language": target_language,
            "glossary": [entry.model_dump() for entry in glossary],
            "model": config.model,
            "style": config.subtitle_style,
        },
        ensure_ascii=False,
        sort_keys=True,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def estimate_translation_tokens(text: str) -> int:
    return max(1, len(text) // 3)


def build_translation_batches(
    indices: list[int], segments: list[TranscriptSegment], config: TranslationConfig
) -> list[list[int]]:
    if not indices:
        return []
    max_size = max(1, config.batch_size if config.batch_enabled else 1)
    token_limit = max(1000, config.batch_token_limit)
    batches: list[list[int]] = []
    current: list[int] = []
    current_tokens = 700
    for index in indices:
        text = segments[index].normalizedText or segments[index].sourceText
        segment_tokens = estimate_translation_tokens(text) + 16
        if current and (
            len(current) >= max_size or current_tokens + segment_tokens > token_limit
        ):
            batches.append(current)
            current = []
            current_tokens = 700
        current.append(index)
        current_tokens += segment_tokens
    if current:
        batches.append(current)
    return batches


def build_semantic_context(
    segments: list[TranscriptSegment], start: int, end: int, max_chars: int
) -> list[str]:
    """Keep neighboring subtitle fragments together without changing their IDs."""
    selected: list[str] = []
    total = 0
    for segment in segments[start:end]:
        text = compact_text(segment.normalizedText or segment.sourceText, max_chars)
        if not text:
            continue
        if total + len(text) > max_chars and selected:
            break
        selected.append(text)
        total += len(text) + 1
    return selected or ["(none)"]


class TranslationOrchestrator:
    def __init__(self, config: TranslationConfig):
        self.config = config
        self.cache = None

    def translate_segments(
        self,
        *,
        job_id: str,
        source_language: str,
        target_language: str,
        segments: list[TranscriptSegment],
        glossary: list[GlossaryEntry],
        provider_override: str | None = None,
        progress_callback: Callable[[int, int], None] | None = None,
        segment_callback: Callable[[TranslationSegment], None] | None = None,
        completed_segment_ids: set[str] | None = None,
    ) -> list[TranslationSegment]:
        provider = build_provider(self.config, provider_override)
        if self.config.cache_enabled and provider.name != "passthrough":
            self.cache = TranslationCache(self.config.cache_path)
        translations: list[TranslationSegment] = []
        completed = completed_segment_ids or set()
        memory = TranslationMemory()
        if self.config.memory_enabled:
            seed_translation_memory(
                memory,
                segments=segments,
                glossary=glossary,
                target_language=target_language,
                max_names=self.config.name_memory_max_items,
            )
        completed_count = sum(1 for segment in segments if segment.id in completed)
        if progress_callback:
            progress_callback(completed_count, len(segments))
        pending_indices = []
        for index, segment in enumerate(segments):
            if segment.id in completed:
                continue
            source_text = apply_source_corrections(
                segment.normalizedText or segment.sourceText, source_language
            )
            cache_key = translation_cache_key(
                source_text=source_text,
                source_language=source_language,
                target_language=target_language,
                glossary=glossary,
                config=self.config,
            )
            cached = self.cache.get(cache_key) if self.cache else None
            if cached is not None:
                translation = TranslationSegment(
                    id=str(uuid4()),
                    jobId=job_id,
                    segmentId=segment.id,
                    targetLanguage=target_language,
                    translatedText=cached,
                    model=f"cache:{self.config.model}",
                    warnings=["translation restored from cache"],
                )
                translations.append(translation)
                if segment_callback:
                    segment_callback(translation)
                completed_count += 1
                if progress_callback:
                    progress_callback(completed_count, len(segments))
                if self.config.memory_enabled:
                    update_translation_memory(
                        memory, source_text=source_text, translated_text=cached,
                        target_language=target_language,
                        max_notes=self.config.memory_max_items,
                        max_names=self.config.name_memory_max_items,
                    )
                continue
            pending_indices.append(index)

        for batch_indices in build_translation_batches(
            pending_indices, segments, self.config
        ):
            batch_translations = self._translate_batch(
                provider=provider,
                job_id=job_id,
                source_language=source_language,
                target_language=target_language,
                segments=segments,
                indices=batch_indices,
                glossary=glossary,
                memory=memory,
            )
            for index, translation in batch_translations:
                translations.append(translation)
                if segment_callback:
                    segment_callback(translation)
                if self.config.memory_enabled:
                    segment = segments[index]
                    current_text = apply_source_corrections(
                        segment.normalizedText or segment.sourceText, source_language
                    )
                    update_translation_memory(
                        memory,
                        source_text=current_text,
                        translated_text=translation.translatedText,
                        target_language=target_language,
                        max_notes=self.config.memory_max_items,
                        max_names=self.config.name_memory_max_items,
                    )
                else:
                    segment = segments[index]
                    current_text = apply_source_corrections(
                        segment.normalizedText or segment.sourceText, source_language
                    )
                if self.cache:
                    cache_key = translation_cache_key(
                        source_text=current_text,
                        source_language=source_language,
                        target_language=target_language,
                        glossary=glossary,
                        config=self.config,
                    )
                    self.cache.put(cache_key, translation.translatedText, self.config.model)
                completed_count += 1
                if progress_callback:
                    progress_callback(completed_count, len(segments))
            if (
                self.config.memory_enabled
                and self.config.topic_summary_enabled
                and self.config.topic_summary_interval_batches > 0
                and completed_count // max(1, self.config.batch_size)
                % self.config.topic_summary_interval_batches == 0
            ):
                try:
                    summary = provider.summarize_memory(
                        memory, self.config.topic_summary_max_chars
                    )
                except Exception:
                    summary = ""
                if summary:
                    memory.topic_summary = summary
        order = {segment.id: index for index, segment in enumerate(segments)}
        translations.sort(key=lambda item: order.get(item.segmentId, len(segments)))
        return translations

    def _translate_batch(
        self,
        *,
        provider: TranslationProvider,
        job_id: str,
        source_language: str,
        target_language: str,
        segments: list[TranscriptSegment],
        indices: list[int],
        glossary: list[GlossaryEntry],
        memory: TranslationMemory,
    ) -> list[tuple[int, TranslationSegment]]:
        if len(indices) == 1 or not self.config.batch_enabled:
            return [
                (
                    indices[0],
                    self._translate_single(
                        provider=provider,
                        job_id=job_id,
                        source_language=source_language,
                        target_language=target_language,
                        segments=segments,
                        index=indices[0],
                        glossary=glossary,
                        memory=memory,
                    ),
                )
            ]
        batch_segments = [
            BatchTranslationSegment(
                id=segments[index].id,
                text=apply_source_corrections(
                    segments[index].normalizedText or segments[index].sourceText,
                    source_language,
                ),
            )
            for index in indices
        ]
        first_index = indices[0]
        last_index = indices[-1]
        if self.config.semantic_context_enabled:
            previous = build_semantic_context(
                segments,
                max(0, first_index - self.config.context_previous_segments),
                first_index,
                self.config.semantic_context_max_chars,
            )
            next_segments = build_semantic_context(
                segments,
                last_index + 1,
                last_index + 1 + self.config.context_next_segments,
                self.config.semantic_context_max_chars,
            )
        else:
            previous = [
                apply_source_corrections(s.normalizedText or s.sourceText, source_language)
                for s in segments[
                    max(0, first_index - self.config.context_previous_segments) : first_index
                ]
            ]
            next_segments = [
                apply_source_corrections(s.normalizedText or s.sourceText, source_language)
                for s in segments[
                    last_index + 1 : last_index + 1 + self.config.context_next_segments
                ]
            ]
        try:
            output = provider.translate_batch(
                TranslationBatchInput(
                    sourceLanguage=source_language,
                    targetLanguage=target_language,
                    segments=batch_segments,
                    previousContext=previous,
                    nextContext=next_segments,
                    glossary=glossary,
                    memoryNotes=memory.notes,
                    nameMemory=memory.names,
                    recentTranslations=memory.recent_translations,
                    topicSummary=memory.topic_summary,
                    styleGuide=self.config.subtitle_style,
                )
            )
        except Exception:
            return [
                (
                    index,
                    self._translate_single(
                        provider=provider,
                        job_id=job_id,
                        source_language=source_language,
                        target_language=target_language,
                        segments=segments,
                        index=index,
                        glossary=glossary,
                        memory=memory,
                    ),
                )
                for index in indices
            ]
        prepared: dict[int, tuple[str, list[str], list[str]]] = {}
        for index, batch_segment in zip(indices, batch_segments, strict=True):
            raw_translation = output.translations[batch_segment.id]
            translated = strip_source_echo(raw_translation, batch_segment.text)
            translated = enforce_glossary(translated, glossary, target_language)
            translated = polish_translation_text(translated, target_language)
            warnings = list(output.warnings)
            quality_warnings = validate_translation_quality(
                source_text=batch_segment.text,
                translated_text=translated,
                target_language=target_language,
            )
            prepared[index] = (translated, warnings, quality_warnings)
        proofread_indices = [
            index for index in indices
            if (
                not self.config.proofread_only_low_confidence
                or bool(prepared[index][2])
            )
        ]
        if (
            self.config.proofread_enabled
            and self.config.batch_proofread_enabled
            and provider.name != "passthrough"
            and proofread_indices
        ):
            draft_translations = {
                segments[index].id: prepared[index][0] for index in proofread_indices
            }
            proofread_segments = [
                batch_segment for index, batch_segment in zip(indices, batch_segments, strict=True)
                if index in proofread_indices
            ]
            try:
                proofread = provider.translate_batch(
                    TranslationBatchInput(
                        sourceLanguage=source_language,
                        targetLanguage=target_language,
                        segments=proofread_segments,
                        previousContext=previous,
                        nextContext=next_segments,
                        glossary=glossary,
                        memoryNotes=memory.notes,
                        nameMemory=memory.names,
                        recentTranslations=memory.recent_translations,
                        topicSummary=memory.topic_summary,
                        draftTranslations=draft_translations,
                        styleGuide="proofread",
                    )
                )
            except Exception:
                proofread = None
            if proofread:
                for index, batch_segment in zip(proofread_indices, proofread_segments, strict=True):
                    translated, warnings, quality_warnings = prepared[index]
                    proofread_text = strip_source_echo(
                        proofread.translations[batch_segment.id],
                        batch_segment.text,
                    )
                    proofread_text = enforce_glossary(
                        proofread_text, glossary, target_language
                    )
                    proofread_text = polish_translation_text(
                        proofread_text, target_language
                    )
                    proofread_warnings = validate_translation_quality(
                        source_text=batch_segment.text,
                        translated_text=proofread_text,
                        target_language=target_language,
                    )
                    if translation_quality_score(
                        proofread_warnings, proofread_text
                    ) <= translation_quality_score(quality_warnings, translated):
                        warnings.extend(proofread.warnings)
                        warnings.append("translation proofread by model")
                        prepared[index] = (
                            proofread_text,
                            warnings,
                            proofread_warnings,
                        )
        translations = []
        for index in indices:
            translated, warnings, quality_warnings = prepared[index]
            warnings.extend(quality_warnings)
            translations.append(
                (
                    index,
                    TranslationSegment(
                        id=str(uuid4()),
                        jobId=job_id,
                        segmentId=segments[index].id,
                        targetLanguage=target_language,
                        translatedText=translated,
                        model=f"{provider.name}:{self.config.model}",
                        warnings=warnings,
                    ),
                )
            )
        return translations

    def _translate_single(
        self,
        *,
        provider: TranslationProvider,
        job_id: str,
        source_language: str,
        target_language: str,
        segments: list[TranscriptSegment],
        index: int,
        glossary: list[GlossaryEntry],
        memory: TranslationMemory,
    ) -> TranslationSegment:
        segment = segments[index]
        current_text = apply_source_corrections(
            segment.normalizedText or segment.sourceText, source_language
        )
        if self.config.semantic_context_enabled:
            previous = build_semantic_context(
                segments,
                max(0, index - self.config.context_previous_segments),
                index,
                self.config.semantic_context_max_chars,
            )
            next_segments = build_semantic_context(
                segments,
                index + 1,
                index + 1 + self.config.context_next_segments,
                self.config.semantic_context_max_chars,
            )
        else:
            previous = [
                apply_source_corrections(s.normalizedText or s.sourceText, source_language)
                for s in segments[max(0, index - self.config.context_previous_segments) : index]
            ]
            next_segments = [
                apply_source_corrections(s.normalizedText or s.sourceText, source_language)
                for s in segments[index + 1 : index + 1 + self.config.context_next_segments]
            ]
        output = provider.translate(
            TranslationInput(
                sourceLanguage=source_language,
                targetLanguage=target_language,
                currentText=current_text,
                previousContext=previous,
                nextContext=next_segments,
                glossary=glossary,
                memoryNotes=memory.notes,
                nameMemory=memory.names,
                recentTranslations=memory.recent_translations,
                topicSummary=memory.topic_summary,
                styleGuide=self.config.subtitle_style,
            )
        )
        translated = strip_source_echo(output.translatedText, current_text)
        translated = enforce_glossary(translated, glossary, target_language)
        translated = polish_translation_text(translated, target_language)
        warnings = list(output.warnings)
        quality_warnings = validate_translation_quality(
            source_text=current_text,
            translated_text=translated,
            target_language=target_language,
        )
        if quality_warnings and provider.name != "passthrough":
            retry_output = provider.translate(
                TranslationInput(
                    sourceLanguage=source_language,
                    targetLanguage=target_language,
                    currentText=current_text,
                    previousContext=[],
                    nextContext=[],
                    glossary=glossary,
                    memoryNotes=memory.notes,
                    nameMemory=memory.names,
                    recentTranslations=memory.recent_translations,
                    topicSummary=memory.topic_summary,
                    styleGuide="strict_retry",
                )
            )
            retry_text = strip_source_echo(retry_output.translatedText, current_text)
            retry_text = enforce_glossary(retry_text, glossary, target_language)
            retry_text = polish_translation_text(retry_text, target_language)
            retry_warnings = validate_translation_quality(
                source_text=current_text,
                translated_text=retry_text,
                target_language=target_language,
            )
            if translation_quality_score(retry_warnings, retry_text) <= translation_quality_score(
                quality_warnings, translated
            ):
                translated = retry_text
                warnings.extend(retry_output.warnings)
                warnings.append("translation retried without context after quality checks")
                quality_warnings = retry_warnings
        if enforce_glossary(strip_source_echo(output.translatedText, current_text), glossary, target_language) != strip_source_echo(output.translatedText, current_text):
            warnings.append("glossary terms were enforced after translation")
        if self.config.proofread_enabled and provider.name != "passthrough":
            proofread = provider.translate(
                TranslationInput(
                    sourceLanguage=source_language,
                    targetLanguage=target_language,
                    currentText=current_text,
                    previousContext=previous,
                    nextContext=next_segments,
                    glossary=glossary,
                    memoryNotes=memory.notes,
                    nameMemory=memory.names,
                    recentTranslations=memory.recent_translations,
                    topicSummary=memory.topic_summary,
                    draftTranslation=translated,
                    styleGuide="proofread",
                )
            )
            proofread_text = strip_source_echo(proofread.translatedText, current_text)
            proofread_text = enforce_glossary(proofread_text, glossary, target_language)
            proofread_text = polish_translation_text(proofread_text, target_language)
            proofread_warnings = validate_translation_quality(
                source_text=current_text,
                translated_text=proofread_text,
                target_language=target_language,
            )
            if translation_quality_score(proofread_warnings, proofread_text) <= translation_quality_score(
                quality_warnings, translated
            ):
                translated = proofread_text
                warnings.extend(proofread.warnings)
                warnings.append("translation proofread by model")
                quality_warnings = proofread_warnings
        warnings.extend(quality_warnings)
        return TranslationSegment(
            id=str(uuid4()),
            jobId=job_id,
            segmentId=segment.id,
            targetLanguage=target_language,
            translatedText=translated,
            model=f"{provider.name}:{self.config.model}",
            warnings=warnings,
        )

def seed_translation_memory(
    memory: TranslationMemory,
    *,
    segments: list[TranscriptSegment],
    glossary: list[GlossaryEntry],
    target_language: str,
    max_names: int,
) -> None:
    for entry in glossary:
        if target_language in entry.languages:
            add_unique_limited(memory.names, f"{entry.source} => {entry.target}", max_names)
    for segment in segments[:20]:
        for name in extract_memory_names(segment.normalizedText or segment.sourceText):
            add_unique_limited(memory.names, name, max_names)


def update_translation_memory(
    memory: TranslationMemory,
    *,
    source_text: str,
    translated_text: str,
    target_language: str,
    max_notes: int,
    max_names: int,
) -> None:
    for name in extract_memory_names(source_text):
        add_unique_limited(memory.names, name, max_names)
    for name in extract_memory_names(translated_text):
        add_unique_limited(memory.names, name, max_names)
    note = build_memory_note(source_text, translated_text)
    if note:
        add_unique_limited(memory.notes, note, max_notes)
    approved = translated_text.strip()
    if approved:
        if target_language == "zh-TW":
            approved = re.sub(r"\s+", "", approved)
        add_unique_limited(memory.recent_translations, approved[:80], max_notes)


def build_memory_note(source_text: str, translated_text: str) -> str:
    source = compact_text(source_text, 70)
    translated = compact_text(translated_text, 70)
    if not source or not translated:
        return ""
    return f"{source} => {translated}"


def compact_text(text: str, max_length: int) -> str:
    compacted = re.sub(r"\s+", " ", text).strip()
    if len(compacted) <= max_length:
        return compacted
    return compacted[: max_length - 1].rstrip() + "..."


def extract_memory_names(text: str) -> list[str]:
    candidates: list[str] = []
    patterns = [
        r"\b[A-Z][A-Za-z0-9]*(?:[ ._-][A-Z][A-Za-z0-9]*){0,4}\b",
        r"[\u30a0-\u30ffー]{3,}",
        r"《[^》]{1,30}》",
        r"「[^」]{1,30}」",
        r"『[^』]{1,30}』",
    ]
    for pattern in patterns:
        candidates.extend(re.findall(pattern, text))
    filtered = []
    stop_words = {
        "I",
        "The",
        "This",
        "That",
        "You",
        "We",
        "It",
        "A",
        "An",
        "And",
        "But",
        "So",
    }
    for candidate in candidates:
        cleaned = candidate.strip(" \t\r\n,.:;!?()[]{}")
        if not cleaned or cleaned in stop_words:
            continue
        if len(cleaned) < 2 or len(cleaned) > 60:
            continue
        filtered.append(cleaned)
    return dedupe_preserve_order(filtered)


def add_unique_limited(items: list[str], item: str, limit: int) -> None:
    normalized = item.strip()
    if not normalized:
        return
    items[:] = [existing for existing in items if existing.lower() != normalized.lower()]
    items.append(normalized)
    if len(items) > limit:
        del items[: len(items) - limit]


def dedupe_preserve_order(items: list[str]) -> list[str]:
    seen = set()
    deduped = []
    for item in items:
        key = item.lower()
        if key in seen:
            continue
        seen.add(key)
        deduped.append(item)
    return deduped


def enforce_glossary(text: str, glossary: list[GlossaryEntry], target_language: str) -> str:
    updated = text
    for entry in glossary:
        if target_language not in entry.languages:
            continue
        flags = 0 if entry.caseSensitive else re.IGNORECASE
        if re.search(re.escape(entry.source), updated, flags=flags):
            updated = re.sub(re.escape(entry.source), entry.target, updated, flags=flags)
    return updated


JA_ASR_CORRECTIONS = {
    "神ハンキ": "上半期",
    "神判基": "上半期",
    "椎茸占い": "しいたけ占い",
    "花フラシ": "花降らし",
    "カラクレピエロ": "からくりピエロ",
    "寝カフェ": "ネカフェ",
    "こんな1週間でしたか皆さん": "皆さん、どんな1週間でしたか",
}


def apply_source_corrections(text: str, source_language: str) -> str:
    if source_language != "ja":
        return text
    corrected = text
    for source, target in JA_ASR_CORRECTIONS.items():
        corrected = corrected.replace(source, target)
    return corrected


ZH_TW_POLISH_REPLACEMENTS = {
    "ㄧ": "一",
    "馬賽": "賽馬",
    "全覆蓋影片": "完整翻唱影片",
    "全覆蓋": "完整翻唱",
    "全 COVER": "完整翻唱",
    "卡拉克雷皮埃罗": "《機關小丑》",
    "從裡頭的馬戲團": "《機關小丑》",
    "沙羽里": "Sayuri",
    "紗ゆり": "Sayuri",
    "米澤斯": "Mrs. GREEN APPLE",
    "睡覺咖啡廳": "網咖",
    "那種咖啡廳": "網咖",
}


def polish_translation_text(text: str, target_language: str) -> str:
    if target_language != "zh-TW":
        return text
    polished = text
    for source, target in ZH_TW_POLISH_REPLACEMENTS.items():
        polished = polished.replace(source, target)
    return polished


def strip_source_echo(translated_text: str, source_text: str) -> str:
    text = translated_text.strip()
    source = source_text.strip()
    if source and text.startswith(source):
        text = text[len(source) :].strip(" \n\r\t:：-—")
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if len(lines) >= 2 and source:
        source_chars = set(source)
        kept = []
        for line in lines:
            overlap = sum(1 for char in line if char in source_chars)
            if overlap / max(1, len(line)) < 0.75:
                kept.append(line)
        if kept:
            text = "\n".join(kept)
    return clean_model_output(text)


def validate_translation_quality(
    *, source_text: str, translated_text: str, target_language: str
) -> list[str]:
    warnings: list[str] = []
    if not translated_text.strip():
        return ["empty translation"]
    if target_language == "zh-TW":
        kana_count = len(re.findall(r"[\u3040-\u30ff]", translated_text))
        cjk_count = len(re.findall(r"[\u4e00-\u9fff]", translated_text))
        latin_words = re.findall(r"[A-Za-z]+", translated_text)
        korean_count = len(re.findall(r"[\uac00-\ud7af]", translated_text))
        simplified_markers = set("这为汉语吗马门风发们对剧题犹够过应请点选个")
        if kana_count > max(2, cjk_count // 6):
            warnings.append("translation contains too much Japanese kana")
        noisy_latin = [
            word
            for word in latin_words
            if word.lower()
            not in {
                "vlog",
                "youtube",
                "openai",
                "pc",
                "live",
                "cover",
                "mrs",
                "green",
                "apple",
            }
        ]
        if noisy_latin:
            warnings.append("translation contains unexpected English words")
        if any(char in translated_text for char in simplified_markers):
            warnings.append("translation may contain Simplified Chinese")
        if korean_count:
            warnings.append("translation contains unexpected Korean text")
        if len(translated_text) > max(80, len(source_text) * 2.6):
            warnings.append("translation is unexpectedly long")
        noisy_fragments = ["{", "}", "<", ">", '\\"', ";)", ");", '!!"', "```"]
        if any(fragment in translated_text for fragment in noisy_fragments):
            warnings.append("translation contains formatting or code-like noise")
    return warnings


def translation_quality_score(warnings: list[str], text: str) -> int:
    score = 0
    weights = {
        "empty translation": 100,
        "translation contains formatting or code-like noise": 80,
        "translation contains unexpected English words": 50,
        "translation contains unexpected Korean text": 50,
        "translation contains too much Japanese kana": 40,
        "translation may contain Simplified Chinese": 25,
        "translation is unexpectedly long": 15,
    }
    for warning in warnings:
        score += weights.get(warning, 10)
    rare_latin = re.findall(r"\b[A-Za-z]{4,}\b", text)
    score += len(rare_latin) * 8
    return score
