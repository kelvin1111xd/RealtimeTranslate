from realtime_translate.config import SubtitleConfig
from realtime_translate.schemas import TranscriptSegment, TranslationSegment
from realtime_translate.subtitles import build_cues, export_srt, export_vtt


def test_build_cues_wraps_traditional_chinese():
    segment = TranscriptSegment(
        id="s1",
        jobId="j1",
        index=0,
        startMs=1000,
        endMs=4200,
        sourceText="hello",
        normalizedText="hello",
    )
    translation = TranslationSegment(
        id="t1",
        jobId="j1",
        segmentId="s1",
        targetLanguage="zh-TW",
        translatedText="這是一段用來測試字幕換行品質的繁體中文字幕",
        model="test",
    )
    cues = build_cues([segment], [translation], "zh-TW", SubtitleConfig())
    assert len(cues) == 1
    assert "\n" in cues[0].text
    assert cues[0].startMs == 1000


def test_export_formats():
    segment = TranscriptSegment(id="s1", jobId="j1", index=0, startMs=0, endMs=1500, sourceText="x")
    translation = TranslationSegment(
        id="t1",
        jobId="j1",
        segmentId="s1",
        targetLanguage="en",
        translatedText="Hello world.",
        model="test",
    )
    cues = build_cues([segment], [translation], "en", SubtitleConfig())
    assert "00:00:00,000 --> 00:00:01,500" in export_srt(cues)
    assert export_vtt(cues).startswith("WEBVTT")

