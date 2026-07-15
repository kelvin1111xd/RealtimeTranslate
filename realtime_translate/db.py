from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Iterable

from .schemas import (
    GlossaryEntry,
    JobStatus,
    SubtitleCue,
    TranscriptSegment,
    TranslationSegment,
    VideoJob,
    utc_now_iso,
)


class Database:
    def __init__(self, path: Path):
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.init()

    def connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.path)
        conn.row_factory = sqlite3.Row
        return conn

    def init(self) -> None:
        with self.connect() as conn:
            conn.executescript(
                """
                PRAGMA journal_mode = WAL;
                CREATE TABLE IF NOT EXISTS jobs (
                  id TEXT PRIMARY KEY,
                  youtube_url TEXT NOT NULL,
                  video_id TEXT NOT NULL,
                  title TEXT,
                  duration REAL,
                  channel TEXT,
                  thumbnail TEXT,
                  audio_path TEXT,
                  status TEXT NOT NULL,
                  progress_percent INTEGER NOT NULL DEFAULT 0,
                  progress_stage TEXT NOT NULL DEFAULT 'queued',
                  progress_message TEXT NOT NULL DEFAULT 'Queued',
                  progress_detail TEXT NOT NULL DEFAULT '{}',
                  source_language TEXT NOT NULL,
                  detected_language TEXT,
                  target_languages TEXT NOT NULL,
                  pipeline_fingerprint TEXT,
                  error TEXT,
                  created_at TEXT NOT NULL,
                  updated_at TEXT NOT NULL
                );
                CREATE INDEX IF NOT EXISTS idx_jobs_video_id ON jobs(video_id);

                CREATE TABLE IF NOT EXISTS transcript_segments (
                  id TEXT PRIMARY KEY,
                  job_id TEXT NOT NULL,
                  segment_index INTEGER NOT NULL,
                  start_ms INTEGER NOT NULL,
                  end_ms INTEGER NOT NULL,
                  source_text TEXT NOT NULL,
                  normalized_text TEXT,
                  speaker TEXT,
                  confidence REAL,
                  FOREIGN KEY(job_id) REFERENCES jobs(id)
                );
                CREATE INDEX IF NOT EXISTS idx_transcript_job ON transcript_segments(job_id);

                CREATE TABLE IF NOT EXISTS translation_segments (
                  id TEXT PRIMARY KEY,
                  job_id TEXT NOT NULL,
                  segment_id TEXT NOT NULL,
                  target_language TEXT NOT NULL,
                  translated_text TEXT NOT NULL,
                  polished_text TEXT,
                  model TEXT NOT NULL,
                  warnings TEXT NOT NULL DEFAULT '[]',
                  FOREIGN KEY(job_id) REFERENCES jobs(id),
                  FOREIGN KEY(segment_id) REFERENCES transcript_segments(id)
                );
                CREATE INDEX IF NOT EXISTS idx_translation_job_lang
                  ON translation_segments(job_id, target_language);

                CREATE TABLE IF NOT EXISTS subtitle_cues (
                  id INTEGER PRIMARY KEY AUTOINCREMENT,
                  job_id TEXT NOT NULL,
                  video_id TEXT NOT NULL,
                  language TEXT NOT NULL,
                  start_ms INTEGER NOT NULL,
                  end_ms INTEGER NOT NULL,
                  text TEXT NOT NULL
                );
                CREATE INDEX IF NOT EXISTS idx_cues_video_lang
                  ON subtitle_cues(video_id, language);

                CREATE TABLE IF NOT EXISTS video_glossary (
                  video_id TEXT NOT NULL,
                  source TEXT NOT NULL,
                  target TEXT NOT NULL,
                  languages TEXT NOT NULL,
                  case_sensitive INTEGER NOT NULL DEFAULT 0
                );

                CREATE TABLE IF NOT EXISTS channel_glossary (
                  channel TEXT NOT NULL,
                  source TEXT NOT NULL,
                  target TEXT NOT NULL,
                  languages TEXT NOT NULL,
                  case_sensitive INTEGER NOT NULL DEFAULT 0
                );
                """
            )
            self._migrate_jobs(conn)
            self._migrate_translations(conn)

    @staticmethod
    def _migrate_jobs(conn: sqlite3.Connection) -> None:
        columns = {
            row["name"]
            for row in conn.execute("PRAGMA table_info(jobs)").fetchall()
        }
        migrations = {
            "progress_percent": "ALTER TABLE jobs ADD COLUMN progress_percent INTEGER NOT NULL DEFAULT 0",
            "progress_stage": "ALTER TABLE jobs ADD COLUMN progress_stage TEXT NOT NULL DEFAULT 'queued'",
            "progress_message": "ALTER TABLE jobs ADD COLUMN progress_message TEXT NOT NULL DEFAULT 'Queued'",
            "progress_detail": "ALTER TABLE jobs ADD COLUMN progress_detail TEXT NOT NULL DEFAULT '{}'",
            "pipeline_fingerprint": "ALTER TABLE jobs ADD COLUMN pipeline_fingerprint TEXT",
        }
        for column, statement in migrations.items():
            if column not in columns:
                conn.execute(statement)
        conn.execute(
            """
            UPDATE jobs
            SET progress_percent = CASE
                WHEN status = 'done' THEN 100
                WHEN status = 'failed' THEN 100
                WHEN status = 'exporting' THEN 92
                WHEN status = 'translating' THEN 65
                WHEN status = 'transcribing' THEN 35
                WHEN status = 'downloading' THEN 15
                ELSE progress_percent
              END,
              progress_stage = CASE
                WHEN progress_stage = 'queued' THEN status
                ELSE progress_stage
              END,
              progress_message = CASE
                WHEN progress_message = 'Queued' AND status = 'done' THEN 'Subtitle generation complete'
                WHEN progress_message = 'Queued' AND status = 'failed' THEN 'Job failed'
                WHEN progress_message = 'Queued' THEN status
                ELSE progress_message
              END
            WHERE progress_percent = 0 OR progress_message = 'Queued'
            """
        )

    @staticmethod
    def _migrate_translations(conn: sqlite3.Connection) -> None:
        conn.execute(
            """
            DELETE FROM translation_segments
            WHERE rowid NOT IN (
              SELECT MAX(rowid)
              FROM translation_segments
              GROUP BY job_id, target_language, segment_id
            )
            """
        )
        conn.execute(
            """
            CREATE UNIQUE INDEX IF NOT EXISTS idx_translation_job_lang_segment
            ON translation_segments(job_id, target_language, segment_id)
            """
        )

    def save_job(self, job: VideoJob) -> None:
        with self.connect() as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO jobs (
                  id, youtube_url, video_id, title, duration, channel, thumbnail, audio_path,
                  status, progress_percent, progress_stage, progress_message, progress_detail,
                  source_language, detected_language, target_languages, pipeline_fingerprint, error,
                  created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    job.id,
                    job.youtubeUrl,
                    job.videoId,
                    job.title,
                    job.duration,
                    job.channel,
                    job.thumbnail,
                    job.audioPath,
                    job.status.value,
                    job.progressPercent,
                    job.progressStage,
                    job.progressMessage,
                    json.dumps(job.progressDetail, ensure_ascii=False),
                    job.sourceLanguage,
                    job.detectedLanguage,
                    json.dumps(job.targetLanguages),
                    job.pipelineFingerprint,
                    job.error,
                    job.createdAt,
                    utc_now_iso(),
                ),
            )

    def get_job(self, job_id: str) -> VideoJob | None:
        with self.connect() as conn:
            row = conn.execute("SELECT * FROM jobs WHERE id = ?", (job_id,)).fetchone()
        return self._row_to_job(row) if row else None

    def delete_job(self, job_id: str) -> bool:
        with self.connect() as conn:
            exists = conn.execute("SELECT 1 FROM jobs WHERE id = ?", (job_id,)).fetchone()
            if not exists:
                return False
            for table in ("subtitle_cues", "translation_segments", "transcript_segments"):
                conn.execute(f"DELETE FROM {table} WHERE job_id = ?", (job_id,))
            conn.execute("DELETE FROM jobs WHERE id = ?", (job_id,))
        return True

    def get_latest_job_by_video(self, video_id: str) -> VideoJob | None:
        with self.connect() as conn:
            row = conn.execute(
                "SELECT * FROM jobs WHERE video_id = ? ORDER BY created_at DESC LIMIT 1", (video_id,)
            ).fetchone()
        return self._row_to_job(row) if row else None

    def get_active_job_by_video(self, video_id: str) -> VideoJob | None:
        with self.connect() as conn:
            row = conn.execute(
                """SELECT * FROM jobs
                   WHERE video_id = ? AND status IN (?, ?, ?, ?, ?)
                   ORDER BY created_at DESC LIMIT 1""",
                (
                    video_id,
                    JobStatus.queued.value,
                    JobStatus.downloading.value,
                    JobStatus.transcribing.value,
                    JobStatus.translating.value,
                    JobStatus.exporting.value,
                ),
            ).fetchone()
        return self._row_to_job(row) if row else None

    def list_jobs_by_statuses(self, statuses: list[JobStatus]) -> list[VideoJob]:
        if not statuses:
            return []
        placeholders = ",".join("?" for _ in statuses)
        with self.connect() as conn:
            rows = conn.execute(
                f"""
                SELECT * FROM jobs
                WHERE status IN ({placeholders})
                ORDER BY created_at
                """,
                tuple(status.value for status in statuses),
            ).fetchall()
        return [self._row_to_job(row) for row in rows]

    def get_latest_job_with_transcript(self, video_id: str) -> VideoJob | None:
        with self.connect() as conn:
            row = conn.execute(
                """
                SELECT j.* FROM jobs j
                WHERE j.video_id = ?
                  AND EXISTS (
                    SELECT 1 FROM transcript_segments ts WHERE ts.job_id = j.id LIMIT 1
                  )
                ORDER BY j.created_at DESC
                LIMIT 1
                """,
                (video_id,),
            ).fetchone()
        return self._row_to_job(row) if row else None

    def get_latest_completed_job_for_languages(
        self, video_id: str, languages: list[str], fingerprint: str | None = None
    ) -> VideoJob | None:
        if not languages:
            return None
        placeholders = ",".join("?" for _ in languages)
        with self.connect() as conn:
            row = conn.execute(
                f"""
                SELECT j.* FROM jobs j
                WHERE j.video_id = ?
                  AND j.status = ?
                  AND (? IS NULL OR j.pipeline_fingerprint = ?)
                  AND (
                    SELECT COUNT(DISTINCT sc.language)
                    FROM subtitle_cues sc
                    WHERE sc.video_id = j.video_id
                      AND sc.job_id = j.id
                      AND sc.language IN ({placeholders})
                  ) = ?
                ORDER BY j.created_at DESC
                LIMIT 1
                """,
                (video_id, JobStatus.done.value, fingerprint, fingerprint, *languages, len(set(languages))),
            ).fetchone()
        return self._row_to_job(row) if row else None

    def languages_with_cues(self, video_id: str, languages: list[str]) -> set[str]:
        if not languages:
            return set()
        placeholders = ",".join("?" for _ in languages)
        with self.connect() as conn:
            rows = conn.execute(
                f"""
                SELECT DISTINCT language FROM subtitle_cues
                WHERE video_id = ? AND language IN ({placeholders})
                """,
                (video_id, *languages),
            ).fetchall()
        return {r["language"] for r in rows}

    def update_job_target_languages(self, job_id: str, target_languages: list[str]) -> None:
        with self.connect() as conn:
            conn.execute(
                """
                UPDATE jobs
                SET target_languages = ?, updated_at = ?, error = NULL
                WHERE id = ?
                """,
                (json.dumps(target_languages), utc_now_iso(), job_id),
            )

    def update_job_status(
        self,
        job_id: str,
        status: JobStatus,
        *,
        error: str | None = None,
        detected_language: str | None = None,
        audio_path: str | None = None,
        title: str | None = None,
        duration: float | None = None,
        channel: str | None = None,
        thumbnail: str | None = None,
        progress_percent: int | None = None,
        progress_stage: str | None = None,
        progress_message: str | None = None,
        progress_detail: dict | None = None,
    ) -> None:
        fields = ["status = ?", "updated_at = ?"]
        values: list[object] = [status.value, utc_now_iso()]
        if status != JobStatus.failed and error is None:
            fields.append("error = NULL")
        optional = {
            "error": error,
            "detected_language": detected_language,
            "audio_path": audio_path,
            "title": title,
            "duration": duration,
            "channel": channel,
            "thumbnail": thumbnail,
            "progress_percent": progress_percent,
            "progress_stage": progress_stage,
            "progress_message": progress_message,
            "progress_detail": json.dumps(progress_detail, ensure_ascii=False)
            if progress_detail is not None
            else None,
        }
        for key, value in optional.items():
            if value is not None:
                fields.append(f"{key} = ?")
                values.append(value)
        values.append(job_id)
        with self.connect() as conn:
            conn.execute(f"UPDATE jobs SET {', '.join(fields)} WHERE id = ?", values)

    def replace_transcript(self, job_id: str, segments: Iterable[TranscriptSegment]) -> None:
        with self.connect() as conn:
            conn.execute("DELETE FROM transcript_segments WHERE job_id = ?", (job_id,))
            conn.executemany(
                """
                INSERT INTO transcript_segments (
                  id, job_id, segment_index, start_ms, end_ms, source_text, normalized_text,
                  speaker, confidence
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                [
                    (
                        s.id,
                        s.jobId,
                        s.index,
                        s.startMs,
                        s.endMs,
                        s.sourceText,
                        s.normalizedText,
                        s.speaker,
                        s.confidence,
                    )
                    for s in segments
                ],
            )

    def list_transcript(self, job_id: str) -> list[TranscriptSegment]:
        with self.connect() as conn:
            rows = conn.execute(
                "SELECT * FROM transcript_segments WHERE job_id = ? ORDER BY segment_index",
                (job_id,),
            ).fetchall()
        return [
            TranscriptSegment(
                id=r["id"],
                jobId=r["job_id"],
                index=r["segment_index"],
                startMs=r["start_ms"],
                endMs=r["end_ms"],
                sourceText=r["source_text"],
                normalizedText=r["normalized_text"],
                speaker=r["speaker"],
                confidence=r["confidence"],
            )
            for r in rows
        ]

    def replace_translations(
        self, job_id: str, language: str, translations: Iterable[TranslationSegment]
    ) -> None:
        with self.connect() as conn:
            conn.execute(
                "DELETE FROM translation_segments WHERE job_id = ? AND target_language = ?",
                (job_id, language),
            )
            conn.executemany(
                """
                INSERT INTO translation_segments (
                  id, job_id, segment_id, target_language, translated_text, polished_text,
                  model, warnings
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                [
                    (
                        t.id,
                        t.jobId,
                        t.segmentId,
                        t.targetLanguage,
                        t.translatedText,
                        t.polishedText,
                        t.model,
                        json.dumps(t.warnings, ensure_ascii=False),
                    )
                    for t in translations
                ],
            )

    def upsert_translation(self, translation: TranslationSegment) -> None:
        with self.connect() as conn:
            conn.execute(
                """
                INSERT INTO translation_segments (
                  id, job_id, segment_id, target_language, translated_text, polished_text,
                  model, warnings
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(job_id, target_language, segment_id) DO UPDATE SET
                  id = excluded.id,
                  translated_text = excluded.translated_text,
                  polished_text = excluded.polished_text,
                  model = excluded.model,
                  warnings = excluded.warnings
                """,
                (
                    translation.id,
                    translation.jobId,
                    translation.segmentId,
                    translation.targetLanguage,
                    translation.translatedText,
                    translation.polishedText,
                    translation.model,
                    json.dumps(translation.warnings, ensure_ascii=False),
                ),
            )

    def delete_translations(self, job_id: str, language: str) -> None:
        with self.connect() as conn:
            conn.execute(
                "DELETE FROM translation_segments WHERE job_id = ? AND target_language = ?",
                (job_id, language),
            )

    def reset_job_outputs(self, job_id: str) -> None:
        with self.connect() as conn:
            conn.execute("DELETE FROM subtitle_cues WHERE job_id = ?", (job_id,))
            conn.execute("DELETE FROM translation_segments WHERE job_id = ?", (job_id,))

    def translated_segment_ids(self, job_id: str, language: str) -> set[str]:
        with self.connect() as conn:
            rows = conn.execute(
                """
                SELECT segment_id FROM translation_segments
                WHERE job_id = ? AND target_language = ?
                """,
                (job_id, language),
            ).fetchall()
        return {row["segment_id"] for row in rows}

    def list_translations(self, job_id: str, language: str) -> list[TranslationSegment]:
        with self.connect() as conn:
            rows = conn.execute(
                """
                SELECT tr.* FROM translation_segments tr
                JOIN transcript_segments ts
                  ON ts.id = tr.segment_id AND ts.job_id = tr.job_id
                WHERE tr.job_id = ? AND tr.target_language = ?
                ORDER BY ts.segment_index
                """,
                (job_id, language),
            ).fetchall()
        return [
            TranslationSegment(
                id=r["id"],
                jobId=r["job_id"],
                segmentId=r["segment_id"],
                targetLanguage=r["target_language"],
                translatedText=r["translated_text"],
                polishedText=r["polished_text"],
                model=r["model"],
                warnings=json.loads(r["warnings"]),
            )
            for r in rows
        ]

    def replace_cues(self, job_id: str, video_id: str, language: str, cues: Iterable[SubtitleCue]) -> None:
        with self.connect() as conn:
            conn.execute(
                "DELETE FROM subtitle_cues WHERE job_id = ? AND language = ?", (job_id, language)
            )
            conn.executemany(
                """
                INSERT INTO subtitle_cues (job_id, video_id, language, start_ms, end_ms, text)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                [(job_id, video_id, c.language, c.startMs, c.endMs, c.text) for c in cues],
            )

    def delete_cues(self, job_id: str, language: str) -> None:
        with self.connect() as conn:
            conn.execute(
                "DELETE FROM subtitle_cues WHERE job_id = ? AND language = ?",
                (job_id, language),
            )

    def list_cues(self, video_id: str, language: str) -> list[SubtitleCue]:
        with self.connect() as conn:
            rows = conn.execute(
                """
                SELECT * FROM subtitle_cues
                WHERE video_id = ? AND language = ?
                ORDER BY start_ms, end_ms
                """,
                (video_id, language),
            ).fetchall()
        return [
            SubtitleCue(startMs=r["start_ms"], endMs=r["end_ms"], text=r["text"], language=r["language"])
            for r in rows
        ]

    def set_video_glossary(self, video_id: str, entries: list[GlossaryEntry]) -> None:
        with self.connect() as conn:
            conn.execute("DELETE FROM video_glossary WHERE video_id = ?", (video_id,))
            conn.executemany(
                """
                INSERT INTO video_glossary (video_id, source, target, languages, case_sensitive)
                VALUES (?, ?, ?, ?, ?)
                """,
                [
                    (
                        video_id,
                        e.source,
                        e.target,
                        json.dumps(e.languages),
                        1 if e.caseSensitive else 0,
                    )
                    for e in entries
                ],
            )

    def get_glossary(self, *, video_id: str, channel: str | None = None) -> list[GlossaryEntry]:
        entries: list[GlossaryEntry] = []
        with self.connect() as conn:
            rows = conn.execute("SELECT * FROM video_glossary WHERE video_id = ?", (video_id,)).fetchall()
            entries.extend(self._row_to_glossary(r) for r in rows)
            if channel:
                rows = conn.execute(
                    "SELECT * FROM channel_glossary WHERE channel = ?", (channel,)
                ).fetchall()
                entries.extend(self._row_to_glossary(r) for r in rows)
        return entries

    @staticmethod
    def _row_to_glossary(row: sqlite3.Row) -> GlossaryEntry:
        return GlossaryEntry(
            source=row["source"],
            target=row["target"],
            languages=json.loads(row["languages"]),
            caseSensitive=bool(row["case_sensitive"]),
        )

    @staticmethod
    def _row_to_job(row: sqlite3.Row) -> VideoJob:
        return VideoJob(
            id=row["id"],
            youtubeUrl=row["youtube_url"],
            videoId=row["video_id"],
            title=row["title"],
            duration=row["duration"],
            channel=row["channel"],
            thumbnail=row["thumbnail"],
            audioPath=row["audio_path"],
            status=JobStatus(row["status"]),
            progressPercent=row["progress_percent"],
            progressStage=row["progress_stage"],
            progressMessage=row["progress_message"],
            progressDetail=json.loads(row["progress_detail"]),
            sourceLanguage=row["source_language"],
            detectedLanguage=row["detected_language"],
            targetLanguages=json.loads(row["target_languages"]),
            pipelineFingerprint=row["pipeline_fingerprint"] if "pipeline_fingerprint" in row.keys() else None,
            error=row["error"],
            createdAt=row["created_at"],
            updatedAt=row["updated_at"],
        )
