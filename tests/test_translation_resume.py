from realtime_translate.config import TranslationConfig
from realtime_translate.db import Database
from realtime_translate.schemas import (
    JobStatus,
    TranscriptSegment,
    TranslationSegment,
    VideoJob,
)
from realtime_translate.translation import TranslationOrchestrator


def make_segment(segment_id: str, index: int) -> TranscriptSegment:
    return TranscriptSegment(
        id=segment_id,
        jobId="job-1",
        index=index,
        startMs=index * 1000,
        endMs=(index + 1) * 1000,
        sourceText=f"segment {index}",
    )


def test_translation_resumes_after_completed_segments():
    orchestrator = TranslationOrchestrator(
        TranslationConfig(primary_provider="passthrough", model="test")
    )
    segments = [make_segment("s1", 0), make_segment("s2", 1), make_segment("s3", 2)]
    persisted = []
    progress = []

    translations = orchestrator.translate_segments(
        job_id="job-1",
        source_language="en",
        target_language="zh-TW",
        segments=segments,
        glossary=[],
        completed_segment_ids={"s1", "s2"},
        segment_callback=persisted.append,
        progress_callback=lambda done, total: progress.append((done, total)),
    )

    assert [translation.segmentId for translation in translations] == ["s3"]
    assert [translation.segmentId for translation in persisted] == ["s3"]
    assert progress == [(2, 3), (3, 3)]


def test_translation_segment_is_upserted(tmp_path):
    database = Database(tmp_path / "db.sqlite")
    segment = make_segment("s1", 0)

    database.save_job(
        VideoJob(
            id="job-1",
            youtubeUrl="https://www.youtube.com/watch?v=test",
            videoId="test",
            status=JobStatus.translating,
            targetLanguages=["zh-TW"],
        )
    )
    database.replace_transcript("job-1", [segment])

    first = TranslationSegment(
        id="t1",
        jobId="job-1",
        segmentId="s1",
        targetLanguage="zh-TW",
        translatedText="first",
        model="test",
    )
    second = first.model_copy(update={"id": "t2", "translatedText": "second"})
    database.upsert_translation(first)
    database.upsert_translation(second)

    translations = database.list_translations("job-1", "zh-TW")
    assert len(translations) == 1
    assert translations[0].id == "t2"
    assert translations[0].translatedText == "second"
