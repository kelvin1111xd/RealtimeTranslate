from __future__ import annotations

import json
import traceback
from pathlib import Path
from uuid import uuid4

from .asr import ASRService
from .config import AppConfig
from .db import Database
from .normalization import normalize_segments
from .schemas import JobStatus, TranscriptSegment, TranslationSegment, VideoJob, utc_now_iso
from .subtitles import build_cues, write_subtitles
from .translation import TranslationOrchestrator
from .youtube import YouTubeIngestion, normalize_audio


class Pipeline:
    def __init__(self, config: AppConfig, db: Database):
        self.config = config
        self.db = db
        self.ingestion = YouTubeIngestion(config.storage.work_dir / "audio", config.youtube)
        self.asr = ASRService(config.asr)
        self.translator = TranslationOrchestrator(config.translation)

    def run_job(self, job_id: str, provider_override: str | None = None) -> None:
        job = self.db.get_job(job_id)
        if not job:
            return
        try:
            self._progress(job.id, JobStatus.downloading, 5, "downloading", "Fetching video metadata")
            metadata = self.ingestion.fetch_metadata(job.youtubeUrl)

            self._progress(job.id, JobStatus.downloading, 10, "downloading", "Downloading audio")
            audio = self.ingestion.download_audio(job.youtubeUrl, job.videoId)
            normalized_audio = audio.with_name(f"{job.videoId}.16k.wav")

            self._progress(job.id, JobStatus.downloading, 20, "audio", "Normalizing audio to 16 kHz mono")
            normalize_audio(audio, normalized_audio)
            self.db.update_job_status(
                job.id,
                JobStatus.transcribing,
                progress_percent=25,
                progress_stage="transcribing",
                progress_message="Starting speech recognition",
                progress_detail={"videoId": job.videoId},
                title=metadata.get("title"),
                duration=metadata.get("duration"),
                channel=metadata.get("channel"),
                thumbnail=metadata.get("thumbnail"),
                audio_path=str(normalized_audio),
            )

            self._progress(job.id, JobStatus.transcribing, 30, "transcribing", "Running ASR")
            detected_language, raw_segments = self.asr.transcribe_file(
                job.id, normalized_audio, job.sourceLanguage
            )
            self._progress(job.id, JobStatus.transcribing, 48, "normalizing", "Cleaning transcript")
            segments = normalize_segments(raw_segments)
            self.db.replace_transcript(job.id, segments)
            self._write_json(
                self.config.storage.work_dir / "transcripts" / f"{job.videoId}.source.json",
                [s.model_dump() for s in segments],
            )
            self.db.update_job_status(
                job.id,
                JobStatus.translating,
                detected_language=detected_language,
                progress_percent=52,
                progress_stage="translating_setup",
                progress_message="Preparing translation",
                progress_detail={
                    "phase": "preparing",
                    "segments": len(segments),
                    "languages": job.targetLanguages,
                },
            )

            latest_job = self.db.get_job(job.id) or job
            glossary = self.db.get_glossary(video_id=job.videoId, channel=latest_job.channel)
            self._translate_and_export(
                job=job,
                languages=job.targetLanguages,
                segments=segments,
                source_language=detected_language,
                glossary=glossary,
                provider_override=provider_override,
                base_percent=55,
                translate_span=30,
                export_base=88,
            )
            self._progress(job.id, JobStatus.done, 100, "done", "Subtitle generation complete")
        except Exception as exc:
            traceback.print_exc()
            self.db.update_job_status(
                job.id,
                JobStatus.failed,
                error=str(exc),
                progress_stage="failed",
                progress_message=str(exc),
            )

    def retranslate(self, job: VideoJob, languages: list[str], provider_override: str | None = None) -> None:
        try:
            self._progress(
                job.id,
                JobStatus.translating,
                52,
                "translating_setup",
                "Preparing translation",
                {"phase": "preparing", "languages": languages},
            )
            segments = self.db.list_transcript(job.id)
            if not segments:
                raise RuntimeError("Cannot retranslate before transcript exists.")
            glossary = self.db.get_glossary(video_id=job.videoId, channel=job.channel)
            source_language = job.detectedLanguage or job.sourceLanguage
            self._translate_and_export(
                job=job,
                languages=languages,
                segments=segments,
                source_language=source_language,
                glossary=glossary,
                provider_override=provider_override,
                base_percent=55,
                translate_span=35,
                export_base=90,
            )
            self._progress(job.id, JobStatus.done, 100, "done", "Subtitle generation complete")
        except Exception as exc:
            traceback.print_exc()
            self.db.update_job_status(
                job.id,
                JobStatus.failed,
                error=str(exc),
                progress_stage="failed",
                progress_message=str(exc),
            )

    def resume_or_recover(self, job: VideoJob, provider_override: str | None = None) -> None:
        segments = self.db.list_transcript(job.id)
        if not segments:
            self.run_job(job.id, provider_override=provider_override)
            return

        segment_ids = {segment.id for segment in segments}
        missing_languages = [
            language
            for language in job.targetLanguages
            if not segment_ids.issubset(self.db.translated_segment_ids(job.id, language))
        ]
        if missing_languages:
            self.retranslate(job, missing_languages, provider_override=provider_override)
            return

        self.rebuild_translation_files(job)
        repaired = self.repair_subtitle_files(job)
        if repaired:
            self._progress(
                job.id,
                JobStatus.done,
                100,
                "done",
                "Recovered missing subtitle files",
                {"repaired": repaired},
            )
            return

        self._progress(job.id, JobStatus.done, 100, "done", "Recovered completed job")

    def rebuild_translation_files(self, job: VideoJob) -> list[str]:
        rebuilt: list[str] = []
        for language in job.targetLanguages:
            translations = self.db.list_translations(job.id, language)
            if not translations:
                continue
            self._write_json(
                self.config.storage.work_dir
                / "translations"
                / f"{job.videoId}.{language}.json",
                [translation.model_dump() for translation in translations],
            )
            rebuilt.append(language)
        return rebuilt

    def repair_subtitle_files(self, job: VideoJob) -> list[str]:
        repaired: list[str] = []
        segments = self.db.list_transcript(job.id)
        if not segments:
            return repaired
        for language in job.targetLanguages:
            translations = self.db.list_translations(job.id, language)
            if not translations:
                continue
            output_dir = self.config.storage.work_dir / "subtitles"
            expected = [
                output_dir / f"{job.videoId}.{language}.{fmt}"
                for fmt in self.config.subtitle.formats
            ]
            existing_cues = self.db.list_cues(job.videoId, language)
            missing_file = any(not path.exists() for path in expected)
            if not existing_cues or missing_file:
                cues = build_cues(segments, translations, language, self.config.subtitle)
                self.db.replace_cues(job.id, job.videoId, language, cues)
                write_subtitles(
                    output_dir,
                    job.videoId,
                    language,
                    cues,
                    self.config.subtitle.formats,
                )
                repaired.append(language)
        return repaired

    def languages_missing_translation_files(self, job: VideoJob, languages: list[str]) -> list[str]:
        missing: list[str] = []
        for language in languages:
            path = self.config.storage.work_dir / "translations" / f"{job.videoId}.{language}.json"
            if not path.exists():
                missing.append(language)
        return missing

    def reset_language_translation(self, job: VideoJob, language: str) -> None:
        self.db.delete_translations(job.id, language)
        self.db.delete_cues(job.id, language)
        translation_path = (
            self.config.storage.work_dir / "translations" / f"{job.videoId}.{language}.json"
        )
        translation_path.unlink(missing_ok=True)
        for fmt in self.config.subtitle.formats:
            (
                self.config.storage.work_dir
                / "subtitles"
                / f"{job.videoId}.{language}.{fmt}"
            ).unlink(missing_ok=True)

    def restore_from_file_cache(
        self,
        *,
        video_id: str,
        youtube_url: str,
        source_language: str,
        target_languages: list[str],
    ) -> VideoJob | None:
        transcript_path = self.config.storage.work_dir / "transcripts" / f"{video_id}.source.json"
        if not transcript_path.exists():
            return None

        translation_paths = {
            language: self.config.storage.work_dir / "translations" / f"{video_id}.{language}.json"
            for language in target_languages
        }
        if any(not path.exists() for path in translation_paths.values()):
            return None

        job_id = str(uuid4())
        now = utc_now_iso()
        job = VideoJob(
            id=job_id,
            youtubeUrl=youtube_url,
            videoId=video_id,
            sourceLanguage=source_language,
            targetLanguages=target_languages,
            status=JobStatus.done,
            progressPercent=100,
            progressStage="done",
            progressMessage="Restored subtitles from file cache",
            progressDetail={"cache": "files", "languages": target_languages},
            createdAt=now,
            updatedAt=now,
        )
        if (self.config.storage.work_dir / "audio" / f"{video_id}.16k.wav").exists():
            job.audioPath = str(self.config.storage.work_dir / "audio" / f"{video_id}.16k.wav")

        segments = [
            TranscriptSegment(**{**item, "jobId": job_id})
            for item in json.loads(transcript_path.read_text(encoding="utf-8"))
        ]
        if not segments:
            return None

        self.db.save_job(job)
        self.db.replace_transcript(job_id, segments)

        for language, path in translation_paths.items():
            translations = [
                TranslationSegment(**{**item, "jobId": job_id, "targetLanguage": language})
                for item in json.loads(path.read_text(encoding="utf-8"))
            ]
            self.db.replace_translations(job_id, language, translations)
            cues = build_cues(segments, translations, language, self.config.subtitle)
            self.db.replace_cues(job_id, video_id, language, cues)
            write_subtitles(
                self.config.storage.work_dir / "subtitles",
                video_id,
                language,
                cues,
                self.config.subtitle.formats,
            )

        return self.db.get_job(job_id) or job

    def _translate_and_export(
        self,
        *,
        job: VideoJob,
        languages: list[str],
        segments,
        source_language: str,
        glossary,
        provider_override: str | None,
        base_percent: int,
        translate_span: int,
        export_base: int,
    ) -> None:
        total_languages = max(1, len(languages))
        for language_index, language in enumerate(languages):
            completed_ids = self.db.translated_segment_ids(job.id, language)
            completed_count = sum(1 for segment in segments if segment.id in completed_ids)
            self._progress(
                job.id,
                JobStatus.translating,
                base_percent,
                "translating_setup",
                f"Preparing {language} translation",
                {
                    "phase": "preparing",
                    "language": language,
                    "languageIndex": language_index + 1,
                    "languageTotal": total_languages,
                    "segments": len(segments),
                    "segmentCompleted": completed_count,
                    "segmentTotal": len(segments),
                    "segmentPercent": round(100 * completed_count / max(1, len(segments))),
                },
            )

            def persist_segment(
                translation: TranslationSegment,
                *,
                lang: str = language,
            ) -> None:
                self.db.upsert_translation(translation)
                persisted = self.db.list_translations(job.id, lang)
                self._write_json(
                    self.config.storage.work_dir
                    / "translations"
                    / f"{job.videoId}.{lang}.json",
                    [item.model_dump() for item in persisted],
                )
                cues = build_cues(segments, persisted, lang, self.config.subtitle)
                self.db.replace_cues(job.id, job.videoId, lang, cues)
                write_subtitles(
                    self.config.storage.work_dir / "subtitles",
                    job.videoId,
                    lang,
                    cues,
                    self.config.subtitle.formats,
                )

            self.translator.translate_segments(
                job_id=job.id,
                source_language=source_language,
                target_language=language,
                segments=segments,
                glossary=glossary,
                provider_override=provider_override,
                progress_callback=lambda done, total, lang=language, lang_index=language_index: self._progress(
                    job.id,
                    JobStatus.translating,
                    base_percent,
                    "translating_segments",
                    f"Translating {lang}: {done}/{total} segments",
                    {
                        "phase": "segments",
                        "language": lang,
                        "languageIndex": lang_index + 1,
                        "languageTotal": total_languages,
                        "segmentCompleted": done,
                        "segmentTotal": total,
                        "segmentPercent": round(100 * done / max(1, total)),
                    },
                ),
                segment_callback=persist_segment,
                completed_segment_ids=completed_ids,
            )
            translations = self.db.list_translations(job.id, language)

            self._progress(
                job.id,
                JobStatus.exporting,
                export_base + round(8 * (language_index + 1) / total_languages),
                "exporting",
                f"Exporting {language} subtitles",
                {"language": language, "formats": self.config.subtitle.formats},
            )
            cues = build_cues(segments, translations, language, self.config.subtitle)
            self.db.replace_cues(job.id, job.videoId, language, cues)
            write_subtitles(
                self.config.storage.work_dir / "subtitles",
                job.videoId,
                language,
                cues,
                self.config.subtitle.formats,
            )

    def _progress(
        self,
        job_id: str,
        status: JobStatus,
        percent: int,
        stage: str,
        message: str,
        detail: dict | None = None,
    ) -> None:
        self.db.update_job_status(
            job_id,
            status,
            progress_percent=max(0, min(100, percent)),
            progress_stage=stage,
            progress_message=message,
            progress_detail=detail or {},
        )

    @staticmethod
    def _write_json(path: Path, payload: object) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(f"{path.suffix}.tmp")
        temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        temporary.replace(path)
