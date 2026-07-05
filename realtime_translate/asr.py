from __future__ import annotations

from pathlib import Path
from uuid import uuid4

from .config import ASRConfig
from .schemas import TranscriptSegment


class ASRService:
    def __init__(self, config: ASRConfig):
        self.config = config
        self._model = None

    @property
    def model(self):
        if self._model is None:
            from faster_whisper import WhisperModel

            try:
                self._model = WhisperModel(
                    self.config.model,
                    device=self.config.device,
                    compute_type=self.config.compute_type,
                )
            except Exception:
                try:
                    self._model = WhisperModel(
                        self.config.fallback_model,
                        device=self.config.device,
                        compute_type=self.config.fallback_compute_type,
                    )
                except Exception:
                    self._model = WhisperModel(
                        self.config.fallback_model,
                        device="cpu",
                        compute_type="int8",
                    )
        return self._model

    def transcribe_file(self, job_id: str, audio_path: Path, source_language: str) -> tuple[str, list[TranscriptSegment]]:
        language = None if source_language == "auto" else source_language
        segments_iter, info = self.model.transcribe(
            str(audio_path),
            language=language,
            vad_filter=self.config.vad_filter,
            beam_size=self.config.beam_size,
            best_of=self.config.best_of,
            temperature=[0.0, 0.2, 0.4],
            word_timestamps=self.config.word_timestamps,
            condition_on_previous_text=self.config.condition_on_previous_text,
        )
        segments: list[TranscriptSegment] = []
        for index, segment in enumerate(segments_iter):
            text = " ".join(segment.text.strip().split())
            if not text:
                continue
            segments.append(
                TranscriptSegment(
                    id=str(uuid4()),
                    jobId=job_id,
                    index=index,
                    startMs=max(0, round(segment.start * 1000)),
                    endMs=max(0, round(segment.end * 1000)),
                    sourceText=text,
                    confidence=None,
                )
            )
        return info.language or source_language, segments

    def transcribe_chunk(self, job_id: str, chunk_path: Path, source_language: str) -> tuple[str, list[TranscriptSegment]]:
        return self.transcribe_file(job_id, chunk_path, source_language)
