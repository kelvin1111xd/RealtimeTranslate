# Realtime Translate

Local-first prototype for generating translated subtitles from YouTube videos. The project accepts a YouTube URL, extracts audio, transcribes speech, translates transcript segments with local or OpenAI-compatible LLM providers, exports subtitle files, and provides a local preview page plus a Chrome extension overlay.

This was built as a personal AI-assisted prototype to practice API design, media-processing pipelines, job-state handling, and browser integration.

## Features

- Create background translation jobs from YouTube URLs.
- Download audio with `yt-dlp` and normalize it to 16 kHz mono WAV with `ffmpeg`.
- Transcribe Chinese, English, and Japanese audio with `faster-whisper`.
- Normalize transcript segments before translation.
- Translate each segment with previous/next context and optional glossary entries.
- Support Ollama, OpenAI-compatible local servers, and a passthrough provider for tests.
- Export subtitles as `SRT`, `VTT`, and `ASS`.
- Serve jobs, transcripts, translations, and subtitles through a FastAPI API.
- Persist job state, transcript segments, translations, glossary entries, and subtitle cues in SQLite.
- Recover interrupted jobs and repair missing subtitle files from persisted records.
- Stream translation progress through HTTP polling and WebSocket endpoints.
- Provide a local preview UI and a Chrome Manifest V3 YouTube subtitle overlay.

## Tech Stack

- Python 3.10+
- FastAPI, Uvicorn, Pydantic
- SQLite
- faster-whisper
- yt-dlp, FFmpeg
- httpx, websockets
- Chrome Extension Manifest V3
- pytest, Ruff

## Project Structure

```text
realtime_translate/
  api.py              # FastAPI routes and WebSocket endpoints
  pipeline.py         # Download -> ASR -> translation -> subtitle export flow
  db.py               # SQLite schema, migration, persistence helpers
  translation.py      # Translation providers, prompts, glossary handling
  asr.py              # faster-whisper wrapper
  youtube.py          # yt-dlp ingestion and FFmpeg audio normalization
  subtitles.py        # SRT/VTT/ASS cue generation
  normalization.py    # Transcript cleanup and segmentation helpers
extension/            # Chrome extension overlay
web/                  # Local preview UI
tests/                # Unit tests for subtitle export and translation resume
config/
  app.example.yaml    # Public example config
```

Runtime folders such as `data/`, `work/`, `models/`, and `config/secrets/` are intentionally ignored by Git.

## Setup

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -e .[dev]
```

Install external tools:

```powershell
ffmpeg -version
yt-dlp --version
```

Create a local config:

```powershell
Copy-Item config\app.example.yaml config\app.yaml
```

Edit `config/app.yaml` for your local ASR device, translation provider, and model. If you use YouTube cookies for videos you are allowed to access, keep the cookie file under `config/secrets/`; never commit cookies.

## Run

Start the local API:

```powershell
python -m realtime_translate.api
```

Open the preview UI:

```text
http://127.0.0.1:8765
```

Create a job:

```powershell
Invoke-RestMethod -Method Post http://127.0.0.1:8765/api/jobs `
  -ContentType 'application/json' `
  -Body '{"youtubeUrl":"https://www.youtube.com/watch?v=VIDEO_ID","sourceLanguage":"auto","targetLanguages":["zh-TW"],"qualityMode":"quality"}'
```

Fetch subtitles:

```text
http://127.0.0.1:8765/api/subtitles/VIDEO_ID?lang=zh-TW&format=vtt
```

## CLI Examples

Run a job synchronously:

```powershell
python -m realtime_translate.cli run "https://www.youtube.com/watch?v=VIDEO_ID" --source auto --targets zh-TW --provider ollama
```

Use passthrough mode to verify ingestion, segmentation, and export without calling a translation model:

```powershell
python -m realtime_translate.cli run "https://www.youtube.com/watch?v=VIDEO_ID" --targets zh-TW --provider passthrough
```

Check available audio formats with the same YouTube config used by jobs:

```powershell
python -m realtime_translate.cli formats "https://www.youtube.com/watch?v=VIDEO_ID"
```

## Chrome Extension

1. Start the local API at `http://127.0.0.1:8765`.
2. Open Chrome `chrome://extensions`.
3. Enable Developer mode.
4. Load unpacked extension from `extension/`.
5. Open a YouTube video that already has generated local subtitles.

The extension adds a local subtitle overlay and reads the YouTube `<video>` element time. It supports subtitle toggling, language selection, bilingual mode, stacked subtitle display, font size, position, and reload controls.

## Tests

```powershell
pytest
```

Current tests cover:

- subtitle cue wrapping and SRT/VTT export
- translation resume behavior
- SQLite upsert behavior for repeated segment translation

## Notes

- This is a local prototype, not a production subtitle service.
- Audio files, transcripts, subtitle outputs, databases, model files, and cookie files are ignored by Git.
- The user is responsible for ensuring they have the right to process a video.
