from __future__ import annotations

import re
from abc import ABC, abstractmethod
from collections.abc import Callable
from uuid import uuid4

import httpx
from pydantic import BaseModel

from .config import TranslationConfig
from .schemas import GlossaryEntry, TranscriptSegment, TranslationSegment


class TranslationInput(BaseModel):
    sourceLanguage: str
    targetLanguage: str
    currentText: str
    previousContext: list[str]
    nextContext: list[str]
    glossary: list[GlossaryEntry] = []
    styleGuide: str | None = None


class TranslationOutput(BaseModel):
    translatedText: str
    warnings: list[str] = []


class TranslationProvider(ABC):
    name: str
    supportedLanguages: list[str]

    @abstractmethod
    def translate(self, input: TranslationInput) -> TranslationOutput:
        raise NotImplementedError


class PassthroughProvider(TranslationProvider):
    name = "passthrough"
    supportedLanguages = ["zh-TW", "en", "ja"]

    def translate(self, input: TranslationInput) -> TranslationOutput:
        return TranslationOutput(
            translatedText=input.currentText,
            warnings=["passthrough provider did not translate text"],
        )


class OllamaProvider(TranslationProvider):
    name = "ollama"
    supportedLanguages = ["zh-TW", "en", "ja"]

    def __init__(self, config: TranslationConfig):
        self.config = config

    def translate(self, input: TranslationInput) -> TranslationOutput:
        prompt = build_prompt(input)
        if self.config.model.startswith("qwen3"):
            prompt = f"/no_think\n{prompt}"
        with httpx.Client(timeout=self.config.request_timeout_seconds) as client:
            response = client.post(
                f"{self.config.ollama_base_url.rstrip('/')}/api/chat",
                json={
                    "model": self.config.model,
                    "messages": [
                        {
                            "role": "system",
                            "content": (
                                "You are a subtitle translation engine. "
                                "Return only the translation of CURRENT_SEGMENT. "
                                "Never translate context, never explain, never add markup."
                            ),
                        },
                        {"role": "user", "content": prompt},
                    ],
                    "stream": False,
                    "options": {
                        "temperature": 0.0,
                        "top_p": 0.8,
                        "repeat_penalty": 1.08,
                        "num_ctx": 8192,
                    },
                },
            )
            response.raise_for_status()
            text = response.json().get("message", {}).get("content", "").strip()
        return TranslationOutput(translatedText=clean_model_output(text))


class OpenAICompatibleProvider(TranslationProvider):
    name = "openai_compatible"
    supportedLanguages = ["zh-TW", "en", "ja"]

    def __init__(self, config: TranslationConfig):
        self.config = config

    def translate(self, input: TranslationInput) -> TranslationOutput:
        prompt = build_prompt(input)
        with httpx.Client(timeout=self.config.request_timeout_seconds) as client:
            response = client.post(
                f"{self.config.openai_compatible_base_url.rstrip('/')}/chat/completions",
                headers={"Authorization": f"Bearer {self.config.openai_compatible_api_key}"},
                json={
                    "model": self.config.model,
                    "messages": [
                        {"role": "system", "content": "You are a professional subtitle translator."},
                        {"role": "user", "content": prompt},
                    ],
                    "temperature": 0.1,
                },
            )
            response.raise_for_status()
            text = response.json()["choices"][0]["message"]["content"].strip()
        return TranslationOutput(translatedText=clean_model_output(text))


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
    glossary = "\n".join(
        f"- {entry.source} => {entry.target}"
        for entry in input.glossary
        if input.targetLanguage in entry.languages
    )
    previous = "\n".join(input.previousContext) or "(none)"
    next_context = "\n".join(input.nextContext) or "(none)"
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

<GLOSSARY>
{glossary or "(none)"}
</GLOSSARY>

Translation:"""


def clean_model_output(text: str) -> str:
    text = text.strip().strip('"').strip()
    text = re.sub(r"^(Translation|翻譯|訳)[:：]\s*", "", text, flags=re.IGNORECASE)
    return text.strip()


class TranslationOrchestrator:
    def __init__(self, config: TranslationConfig):
        self.config = config

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
        translations: list[TranslationSegment] = []
        completed = completed_segment_ids or set()
        completed_count = sum(1 for segment in segments if segment.id in completed)
        if progress_callback:
            progress_callback(completed_count, len(segments))
        for index, segment in enumerate(segments):
            if segment.id in completed:
                continue
            current_text = apply_source_corrections(
                segment.normalizedText or segment.sourceText, source_language
            )
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
            warnings.extend(quality_warnings)
            translation = TranslationSegment(
                id=str(uuid4()),
                jobId=job_id,
                segmentId=segment.id,
                targetLanguage=target_language,
                translatedText=translated,
                model=f"{provider.name}:{self.config.model}",
                warnings=warnings,
            )
            translations.append(translation)
            if segment_callback:
                segment_callback(translation)
            completed_count += 1
            if progress_callback:
                progress_callback(completed_count, len(segments))
        return translations


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
