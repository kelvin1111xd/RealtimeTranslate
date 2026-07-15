from realtime_translate.config import TranslationConfig
from realtime_translate.db import Database
from realtime_translate.schemas import (
    GlossaryEntry,
    JobStatus,
    TranscriptSegment,
    TranslationSegment,
    VideoJob,
)
from realtime_translate.translation import (
    BatchTranslationSegment,
    TranslationBatchInput,
    TranslationInput,
    TranslationMemory,
    TranslationOrchestrator,
    build_batch_prompt,
    build_translation_batches,
    build_prompt,
    parse_batch_translation_output,
    update_translation_memory,
)


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


def test_translation_batches_segments_by_default():
    orchestrator = TranslationOrchestrator(
        TranslationConfig(primary_provider="passthrough", model="test", proofread_enabled=False)
    )
    segments = [make_segment(f"s{index}", index) for index in range(13)]
    persisted = []

    translations = orchestrator.translate_segments(
        job_id="job-1",
        source_language="en",
        target_language="zh-TW",
        segments=segments,
        glossary=[],
        segment_callback=persisted.append,
    )

    assert len(translations) == 13
    assert [translation.segmentId for translation in translations] == [
        segment.id for segment in segments
    ]
    assert [translation.segmentId for translation in persisted] == [
        segment.id for segment in segments
    ]


def test_translation_batches_respect_token_limit():
    config = TranslationConfig(batch_size=12, batch_token_limit=1000)
    segments = [
        make_segment(f"s{index}", index).model_copy(update={"sourceText": "long " * 300})
        for index in range(3)
    ]

    batches = build_translation_batches([0, 1, 2], segments, config)

    assert len(batches) == 3


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


def test_translation_prompt_includes_memory_blocks():
    prompt = build_prompt(
        TranslationInput(
            sourceLanguage="en",
            targetLanguage="zh-TW",
            currentText="Alice mentioned OpenAI again.",
            previousContext=["Alice started the demo."],
            nextContext=["Bob replies later."],
            glossary=[
                GlossaryEntry(source="OpenAI", target="OpenAI", languages=["zh-TW"])
            ],
            memoryNotes=["Alice started the demo. => Alice 開始示範。"],
            nameMemory=["Alice", "Bob"],
            recentTranslations=["Alice 開始示範。"],
        )
    )

    assert "<MEMORY>" in prompt
    assert "Alice" in prompt
    assert "Recent topic notes" in prompt
    assert "Alice 開始示範。" in prompt


def test_batch_translation_prompt_requires_json_with_ids():
    prompt = build_batch_prompt(
        TranslationBatchInput(
            sourceLanguage="en",
            targetLanguage="zh-TW",
            segments=[
                BatchTranslationSegment(id="s1", text="Alice starts."),
                BatchTranslationSegment(id="s2", text="Bob replies."),
            ],
            previousContext=[],
            nextContext=[],
            nameMemory=["Alice", "Bob"],
        )
    )

    assert "Return ONLY valid JSON" in prompt
    assert '"id": "s1"' in prompt
    assert '"translation": "..."' in prompt


def test_parse_batch_translation_output_requires_all_ids():
    output = parse_batch_translation_output(
        """
        ```json
        [
          {"id": "s1", "translation": "第一段"},
          {"id": "s2", "translation": "第二段"}
        ]
        ```
        """,
        ["s1", "s2"],
    )

    assert output.translations == {"s1": "第一段", "s2": "第二段"}


def test_translation_memory_keeps_recent_limited_context():
    memory = TranslationMemory()

    for index in range(3):
        update_translation_memory(
            memory,
            source_text=f"Alice topic {index}",
            translated_text=f"Alice 話題 {index}",
            target_language="zh-TW",
            max_notes=2,
            max_names=2,
        )

    assert len(memory.notes) == 2
    assert memory.notes[-1] == "Alice topic 2 => Alice 話題 2"
    assert memory.names == ["Alice"]
