# YouTube AI Subtitle Translation System Technical Spec

## 1. System Goal

Build a local-first subtitle translation system that can:

- Accept a YouTube URL.
- Extract the video's audio.
- Convert speech to text.
- Support Chinese, English, and Japanese speech recognition.
- Translate the transcript into one or more target languages.
- Export subtitles as `SRT`, `VTT`, and optionally `ASS`.
- Inject generated subtitles into the YouTube player through a Chrome extension or HTML overlay.
- Keep the architecture extensible for future realtime translation.

The initial priority is quality, not realtime latency. The system should run on a consumer GPU such as an RTX 3060 Ti 8GB.

## 2. High-Level Architecture

```text
YouTube URL
  -> Video Ingestion Service
  -> Audio Extraction Service
  -> ASR Service
  -> Transcript Normalization Service
  -> Translation Orchestrator
  -> Subtitle Segmentation Service
  -> Subtitle Export Service
  -> Chrome Extension / HTML Overlay
```

Future realtime path:

```text
Live Audio Capture
  -> Streaming ASR
  -> Streaming Translation
  -> Realtime Subtitle Overlay
```

## 3. Recommended Tech Stack

| Module | Recommended Technology |
|---|---|
| YouTube audio download | `yt-dlp` |
| Audio processing | `ffmpeg` |
| Speech recognition | `faster-whisper` |
| ASR model | `large-v3` for quality, `medium` as fallback |
| Translation models | TowerInstruct 7B Q4/Q5, Qwen 7B/8B Instruct, MADLAD-400-3B-MT |
| Local LLM runtime | `llama.cpp`, `Ollama`, or compatible local inference server |
| Backend API | Python `FastAPI` |
| Job queue | `Redis + RQ` or `Celery` |
| Database | SQLite for local-first operation, PostgreSQL optional later |
| Subtitle formats | `SRT`, `VTT`, `ASS` |
| Browser integration | Chrome Extension Manifest V3 |
| Frontend | React, Vue, or Svelte |
| Local API host | `127.0.0.1:8765` |

## 4. Core Data Models

```ts
type VideoJob = {
  id: string;
  youtubeUrl: string;
  videoId: string;
  title?: string;
  duration?: number;
  channel?: string;
  thumbnail?: string;
  audioPath?: string;
  status:
    | "queued"
    | "downloading"
    | "transcribing"
    | "translating"
    | "exporting"
    | "done"
    | "failed";
  progressPercent: number;
  progressStage: string;
  progressMessage: string;
  progressDetail?: Record<string, unknown>;
  sourceLanguage?: "auto" | "zh" | "en" | "ja";
  detectedLanguage?: string;
  targetLanguages: string[];
  error?: string;
  createdAt: string;
  updatedAt: string;
};

type TranscriptSegment = {
  id: string;
  jobId: string;
  index: number;
  startMs: number;
  endMs: number;
  sourceText: string;
  normalizedText?: string;
  speaker?: string;
  confidence?: number;
};

type TranslationSegment = {
  id: string;
  jobId: string;
  segmentId: string;
  targetLanguage: string;
  translatedText: string;
  polishedText?: string;
  model: string;
  warnings?: string[];
};

type SubtitleCue = {
  startMs: number;
  endMs: number;
  text: string;
  language: string;
};
```

## 5. Pipeline Details

### 5.1 YouTube Ingestion

Input example:

```json
{
  "youtubeUrl": "https://www.youtube.com/watch?v=xxxx",
  "sourceLanguage": "auto",
  "targetLanguages": ["zh-TW", "en", "ja"],
  "qualityMode": "quality"
}
```

Responsibilities:

- Use `yt-dlp` to fetch video metadata.
- Download audio only unless the user later requests burned-in subtitles.
- Store metadata:
  - `video_id`
  - `title`
  - `duration`
  - `channel`
  - `thumbnail`
  - `audio_path`

Recommended command:

```bash
yt-dlp -f ba -x --audio-format wav --audio-quality 0 -o "work/audio/%(id)s.%(ext)s" "<youtube_url>"
```

Notes:

- The user is responsible for ensuring they have the right to process the video.
- If the video has official subtitles, the system may optionally import them as reference text, but quality mode should still support ASR from audio.
- Members-only or age/region-gated videos may require user-provided browser cookies. The backend must support a Netscape-format cookies file and pass it to `yt-dlp`.
- If YouTube requires JavaScript challenge solving, the backend must support configurable JavaScript runtimes for `yt-dlp`.
- The preferred audio format selector is resilient fallback such as `bestaudio/best`, not a single fixed format id.
- The API should expose a diagnostics endpoint that lists available YouTube formats using the same config as job creation, so cookie/runtime issues can be verified without starting a full job.

### 5.2 Audio Extraction

Normalize audio with `ffmpeg`:

```bash
ffmpeg -i input.wav -ac 1 -ar 16000 -vn output_16k_mono.wav
```

Output format:

```text
16kHz mono WAV
```

Reason:

- Whisper-based ASR works reliably with 16kHz mono audio.
- The same format can be reused by future realtime ASR chunks.

### 5.3 ASR Service

Recommended model:

```text
faster-whisper large-v3
device: cuda
compute_type: float16 or int8_float16
```

Quality-first settings:

```python
model.transcribe(
  audio_path,
  language=None,
  vad_filter=True,
  beam_size=5,
  best_of=5,
  temperature=[0.0, 0.2, 0.4],
  word_timestamps=True,
  condition_on_previous_text=True
)
```

Language strategy:

- `sourceLanguage = auto`: let Whisper detect the language.
- Supported initial languages: Chinese, English, Japanese.
- If detection is wrong, allow users to manually specify language and rerun ASR.

Output example:

```json
[
  {
    "index": 0,
    "startMs": 1230,
    "endMs": 4560,
    "sourceText": "..."
  }
]
```

### 5.4 Transcript Normalization

Purpose: improve the ASR transcript before translation.

Tasks:

- Fix punctuation.
- Merge overly short segments.
- Split overly long segments.
- Normalize names and technical terms.
- Remove obvious repeated filler words.
- Preserve timing as much as possible.

Recommended rules:

- Translation units should usually be 5-20 seconds.
- Translation should use context windows, not isolated subtitle lines.
- Display subtitle cues should be generated separately from ASR segments.

### 5.5 Translation Orchestrator

Quality-first flow:

```text
Transcript segments
  -> Build context window
  -> Translate current segment
  -> Validate glossary terms
  -> Optimize subtitle length
  -> Save target-language segments
```

Context strategy for segment `N`:

- Previous 3-5 normalized source segments.
- Current source segment.
- Next 1-3 normalized source segments.
- The model must output only the translation for the current segment.

This improves:

- Pronoun translation.
- Name consistency.
- Technical term consistency.
- Japanese subject omission handling.
- Long English sentence alignment.

Recommended model choices:

| Scenario | Recommended Model |
|---|---|
| English to Traditional Chinese | TowerInstruct 7B Q4/Q5 or Qwen 7B/8B |
| Japanese to Traditional Chinese | Qwen 7B/8B with glossary |
| Chinese to English | TowerInstruct 7B or Qwen |
| Commercial-friendly multilingual route | MADLAD-400-3B-MT |
| Fast fallback | NLLB distilled 600M |

Licensing note:

- TowerInstruct and NLLB have non-commercial restrictions.
- For commercial use, prefer permissive models such as Apache-licensed options where available.
- The implementation must use a provider abstraction so models can be swapped.

Provider interface:

```ts
interface TranslationProvider {
  name: string;
  supportedLanguages: string[];
  translate(input: TranslationInput): Promise<TranslationOutput>;
}

type TranslationInput = {
  sourceLanguage: string;
  targetLanguage: string;
  currentText: string;
  previousContext: string[];
  nextContext: string[];
  glossary?: GlossaryEntry[];
  styleGuide?: string;
};

type TranslationOutput = {
  translatedText: string;
  warnings?: string[];
};
```

### 5.6 Prompt Templates

Traditional Chinese subtitle prompt:

```text
You are a professional subtitle translator.

Task:
Translate the CURRENT segment into Traditional Chinese for video subtitles.

Rules:
- Use natural Traditional Chinese used in Taiwan.
- Preserve the meaning accurately.
- Do not add explanations.
- Do not translate names inconsistently.
- Keep the translation concise enough for subtitles.
- Use the glossary when provided.
- Output only the translated subtitle text.

Previous context:
{{previous_context}}

Current segment:
{{current_text}}

Next context:
{{next_context}}

Glossary:
{{glossary}}

Translation:
```

Japanese subtitle prompt:

```text
Translate the CURRENT segment into natural Japanese subtitles.
Keep it concise, accurate, and suitable for video subtitles.
Output only Japanese text.
```

English subtitle prompt:

```text
Translate the CURRENT segment into natural English subtitles.
Preserve tone and meaning.
Output only the translated subtitle text.
```

### 5.7 Glossary

Glossary entry format:

```json
[
  {
    "source": "OpenAI",
    "target": "OpenAI",
    "languages": ["zh-TW", "ja", "en"],
    "caseSensitive": false
  },
  {
    "source": "machine learning",
    "target": "機器學習",
    "languages": ["zh-TW"]
  }
]
```

Requirements:

- Each video can have its own glossary.
- Each channel can have a default glossary.
- Users can edit glossary terms in the UI.
- Translation output should be validated against glossary rules.

### 5.8 Subtitle Segmentation

Do not directly use long translated text as subtitles. Regenerate readable subtitle text and line breaks, but preserve the source transcript timing exactly.

Timing rule:

- `SubtitleCue.startMs` must equal the matched `TranscriptSegment.startMs`.
- `SubtitleCue.endMs` must equal the matched `TranscriptSegment.endMs`.
- Translation must not shorten, cap, or split cue duration independently from the source timeline.
- Readability improvements should be applied through line wrapping and concise translation, not by changing the timeline.

Traditional Chinese:

- 12-20 Chinese characters per line.
- Maximum 2 lines.
- Avoid punctuation-only lines.

English:

- 32-42 characters per line.
- Maximum 2 lines.

Japanese:

- Around 13-18 characters per line.
- Maximum 2 lines.

Output formats:

- `SRT`: general compatibility.
- `VTT`: best for browser playback and Chrome extension overlay.
- `ASS`: best for styling and burned-in subtitle workflows.

### 5.9 Cache, Recovery, and Repair

The system must avoid rerunning expensive stages when durable artifacts already exist.

Durable state:

- SQLite is the primary job index and stores jobs, transcript segments, translation segments, subtitle cues, progress, and errors.
- `work/` stores recoverable artifacts:
  - `work/audio/{videoId}.wav`
  - `work/audio/{videoId}.16k.wav`
  - `work/transcripts/{videoId}.source.json`
  - `work/translations/{videoId}.{lang}.json`
  - `work/subtitles/{videoId}.{lang}.srt|vtt|ass`

Create-job cache policy:

- If DB has a completed job with transcript, translations, cues, and the requested translation JSON files, return the existing job with `cache = "hit"`. Do not download, ASR, or translate.
- If DB has transcript and cues/translations, but subtitle files are missing, rebuild subtitle files from DB and return `cache = "hit"` with repair detail. Do not retranslate.
- If DB has transcript but the requested translation JSON file is missing, treat the translation artifact as stale and queue retranslation for that language. Do not download audio or rerun ASR.
- If DB has transcript but no translations for a requested language, return `cache = "partial"` and translate only the missing language.
- If DB has no usable transcript but `work/transcripts/{videoId}.source.json` and all requested `work/translations/{videoId}.{lang}.json` files exist, restore them into DB, rebuild cues/files, and return `cache = "files"`.
- If no usable DB or file cache exists, return `cache = "miss"` and run the full pipeline from YouTube ingestion.

Interrupted job recovery on API startup:

- Active jobs in `queued`, `downloading`, `transcribing`, `translating`, or `exporting` must be inspected.
- If no transcript exists, rerun the full job.
- If transcript exists but translations are missing or incomplete, resume translation only.
- Translation recovery must compare transcript segment IDs with persisted translation segment IDs and skip every segment that is already complete.
- Recovery must continue from the first untranslated segment instead of restarting the target language.
- If translations exist but cues or subtitle files are missing, rebuild subtitle cues/files.
- Completed jobs must be repairable when `GET /api/subtitles/{videoId}` is called and subtitle files are missing.

Incremental translation durability:

- Each translated segment must be upserted into SQLite immediately after model inference completes.
- After each segment, atomically rewrite `work/translations/{videoId}.{lang}.json` from persisted DB rows using a temporary file and rename.
- After each segment, rebuild the currently available subtitle cues and write partial `SRT`, `VTT`, and `ASS` files.
- A process interruption may lose only the segment currently being inferred. Previously completed segments must remain usable.
- Translation rows must be unique by `(job_id, target_language, segment_id)`.
- The frontend must be able to read source and translated text before the complete language export finishes.

## 6. Chrome Extension Spec

Purpose: display locally generated subtitles over the YouTube player.

Structure:

```text
Chrome Extension
  -> content script
  -> background service worker
  -> options page
  -> local API client
```

Workflow:

1. User opens a YouTube video.
2. Content script detects the `videoId`.
3. Content script queries the local API:

```http
GET http://localhost:8765/api/subtitles/:videoId?lang=zh-TW
```

4. API returns VTT or JSON cues.
5. Extension inserts a subtitle overlay.
6. Extension watches the `<video>` element's `currentTime`.
7. Overlay displays the matching subtitle cue.

Overlay HTML:

```html
<div id="local-ai-subtitle-overlay">
  <div class="subtitle-line">...</div>
</div>
```

Required features:

- Language switching.
- Font size control.
- Position control.
- Fixed white subtitle text on black background for both normal and stacked display modes.
- Sync with YouTube play, pause, and seek.
- Avoid modifying the original YouTube player internals beyond adding an overlay.
- Load source transcript through the local API when source subtitle or bilingual mode is enabled.
- Auto-load generated subtitles when enabled.
- Auto-create translation jobs when enabled and subtitles are missing.

Popup UI:

- Opened by clicking the browser toolbar icon.
- Must contain common playback-time controls:
  - Enable/disable subtitles.
  - Target language.
  - Show/hide source subtitles.
  - Mono/bilingual display.
  - Normal/stacked display mode.
  - Subtitle position.
  - Font size.
  - Reload subtitles.
  - Create translation job.

Options page:

- Must contain complete and advanced settings:
  - Default target language.
  - Source/translation vertical order.
  - Subtitle style settings.
  - Background opacity.
  - Local API base URL.
  - Shortcut settings.
  - Auto-load subtitles.
  - Auto-create translation jobs.
  - Overlay toolbar visibility.
  - Stacked subtitle mode settings.

YouTube overlay toolbar:

- A compact toolbar should appear near the YouTube player or subtitle overlay.
- Must support immediate playback-time adjustments:
  - Enable/disable subtitles.
  - Toggle mono/bilingual mode.
  - Toggle normal/stacked mode.
  - Switch target language.
  - Move subtitles up/down.
  - Increase/decrease font size.
  - Reload subtitles.

Stacked subtitle mode:

- Current subtitle is displayed at the configured active position.
- When the next subtitle appears, the previous subtitle is pushed in the configured direction and becomes smaller and more transparent.
- Older subtitles continue stacking until the configured maximum visible count is reached.
- Configurable settings:
  - Font size.
  - Horizontal alignment: left, center, or right.
  - Display height / active baseline position.
  - Maximum visible subtitle count.
  - Push direction: up or down.
  - Past subtitle opacity.
  - Past subtitle scale.
- Stacked mode must also use white text on black background.

Manifest V3 example:

```json
{
  "manifest_version": 3,
  "permissions": ["storage"],
  "host_permissions": [
    "https://www.youtube.com/*",
    "http://localhost:8765/*"
  ],
  "content_scripts": [
    {
      "matches": ["https://www.youtube.com/watch*"],
      "js": ["content.js"],
      "css": ["subtitle.css"]
    }
  ]
}
```

## 7. Backend API Spec

Create job:

```http
POST /api/jobs
```

```json
{
  "youtubeUrl": "https://www.youtube.com/watch?v=xxxx",
  "sourceLanguage": "auto",
  "targetLanguages": ["zh-TW", "ja"],
  "qualityMode": "quality"
}
```

Get job status:

```http
GET /api/jobs/:jobId
```

Job status response must include:

- `progressPercent`
- `progressStage`
- `progressMessage`
- `progressDetail`
- `links.cache` when returned from create-job cache handling

Translation progress has two distinct phases:

- `translating_setup`: prompt/context/provider preparation. This uses the main pipeline progress display.
- `translating_segments`: actual per-segment model inference. The frontend must show a separate segment progress bar using:
  - `segmentCompleted`
  - `segmentTotal`
  - `segmentPercent`

The main pipeline progress bar must not be reused as the per-segment progress bar.

Create-job cache return values:

| `links.cache` | Meaning |
|---|---|
| `hit` | Existing completed subtitles are usable; missing subtitle files may have been repaired. |
| `partial` | Transcript exists; only missing target languages are queued for translation. |
| `files` | DB was missing usable records, but transcript/translation files were restored from `work/`. |
| `stale` | DB has prior translation data, but translation JSON artifacts are missing; retranslation is queued without download/ASR. |
| `miss` | No usable cache exists; full pipeline starts from YouTube ingestion. |

Get subtitles:

```http
GET /api/subtitles/:videoId?lang=zh-TW&format=json
GET /api/subtitles/:videoId?lang=zh-TW&format=vtt
GET /api/subtitles/:videoId?lang=zh-TW&format=srt
```

Retranslate:

```http
POST /api/jobs/:jobId/retranslate
```

Update glossary:

```http
PUT /api/videos/:videoId/glossary
```

Diagnostics:

```http
GET /api/diagnostics/youtube-formats?url=<youtubeUrl>
```

This endpoint must use the same YouTube ingestion config as create job and return whether audio formats are visible, whether cookies are configured/found, configured JavaScript runtimes, and the detected audio formats.

Incremental translation stream:

```http
GET /api/jobs/:jobId/translation-stream/:lang
WS /ws/jobs/:jobId/translations/:lang
```

The stream payload must include job status, segment completion counts, percentage, and ordered segments containing:

- `startMs`
- `endMs`
- `sourceText`
- `translatedText` when available

The frontend must update the source/translation list as each segment is persisted.

## 8. Local File Structure

```text
data/
  db.sqlite
work/
  audio/
    {videoId}.wav
    {videoId}.16k.wav
  transcripts/
    {videoId}.source.json
  translations/
    {videoId}.{lang}.json
  subtitles/
    {videoId}.{lang}.srt
    {videoId}.{lang}.vtt
    {videoId}.{lang}.ass
models/
  whisper/
  translation/
config/
  app.yaml
  glossary/
  secrets/
    youtube-cookies.txt
```

Storage rules:

- `data/db.sqlite` is the primary operational database.
- Files in `work/` are durable artifacts and may be used to restore DB records when possible.
- Deleting only subtitle files should trigger subtitle repair.
- Deleting translation JSON files should trigger retranslation from the existing transcript.
- Deleting transcript data and transcript files should force the full pipeline.

## 9. Realtime Translation Extensibility

The batch pipeline must not be designed as "process full video only". All services should support chunk-based processing:

```text
AudioChunk -> TranscriptSegment -> TranslationSegment -> SubtitleCue
```

Batch mode:

```text
Full audio file -> many AudioChunks -> pipeline
```

Realtime mode:

```text
Browser/tab/system audio stream -> rolling AudioChunks -> pipeline
```

Future realtime additions:

- Audio capture:
  - Chrome tab capture.
  - System audio loopback.
  - Microphone.
- Streaming ASR:
  - Whisper chunked inference.
  - Or a true streaming ASR engine.
- Translation cache:
  - Avoid retranslating repeated or finalized sentences.
- Partial subtitles:
  - Show provisional text.
  - Replace with final text after segment confirmation.
- Latency budget:
  - ASR: 1-3 seconds.
  - Translation: 0.5-2 seconds.
  - Overlay delay buffer: configurable 1-3 seconds.

Design requirements for realtime compatibility:

- ASR Service must support both `audio_path` and `audio_chunk`.
- Translation Service must support segment windows instead of complete transcripts only.
- Subtitle Renderer must support incremental cue updates.
- Chrome Extension overlay must support both complete VTT files and WebSocket subtitle pushes.

## 10. Development Phases

### Phase 1: Complete Local Batch Pipeline

- Accept YouTube URL.
- Download audio.
- Transcribe with `faster-whisper`.
- Translate into one target language.
- Export `SRT` and `VTT`.
- Provide a local preview page.

### Phase 2: Chrome Extension

- Detect YouTube `videoId`.
- Load subtitles from `localhost`.
- Display subtitle overlay.
- Support language switching.
- Support subtitle style settings.

### Phase 3: Quality Improvements

- Context-aware translation.
- Glossary support.
- ASR transcript cleanup.
- Translation polishing.
- Multi-language batch translation.
- Manual correction UI.

### Phase 4: Realtime Translation

- Browser tab audio capture.
- Chunked ASR.
- Streaming translation.
- WebSocket subtitle push.
- Latency control.

## 11. Quality-First Default Config

```yaml
asr:
  engine: faster-whisper
  model: large-v3
  device: cuda
  compute_type: float16
  vad_filter: true
  beam_size: 5
  word_timestamps: true

translation:
  primary_provider: local_llm
  model: towerinstruct-7b-q4
  context_previous_segments: 5
  context_next_segments: 2
  glossary_enabled: true
  subtitle_style: concise

subtitle:
  formats: ["srt", "vtt", "ass"]
  max_lines: 2
  zh_tw_chars_per_line: 18
  en_chars_per_line: 42
  ja_chars_per_line: 18
  preserve_source_timeline: true

server:
  host: 127.0.0.1
  port: 8765

youtube:
  cookies_file: config/secrets/youtube-cookies.txt
  js_runtimes: []
```

## 12. Core Implementation Principle

Do not implement this as:

```text
Download video -> translate whole transcript once -> split translated text into subtitles
```

That approach usually produces poor subtitle timing and weak context handling.

The correct approach is:

```text
ASR segments
  -> clean source transcript
  -> translate each segment with context
  -> preserve original segment start/end times
  -> regenerate subtitle line breaks
  -> export synchronized subtitle cues
```

This keeps the system accurate, readable, multi-language capable, and ready for realtime translation later.
