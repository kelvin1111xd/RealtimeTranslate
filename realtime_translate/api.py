from __future__ import annotations

import asyncio
import secrets
import shutil
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from uuid import uuid4

from fastapi import (
    BackgroundTasks,
    FastAPI,
    HTTPException,
    Query,
    Request,
    Response,
    WebSocket,
    WebSocketDisconnect,
)
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, PlainTextResponse
from fastapi.staticfiles import StaticFiles
import httpx

from .config import load_config, pipeline_fingerprint
from .db import Database
from .pipeline import Pipeline
from .schemas import (
    CreateJobRequest,
    GlossaryEntry,
    JobResponse,
    JobStatus,
    RetranslateRequest,
    VideoJob,
    utc_now_iso,
)
from .subtitles import export_ass, export_srt, export_vtt
from .youtube import YouTubeIngestion
from .youtube import extract_video_id


config = load_config()
CURRENT_FINGERPRINT = pipeline_fingerprint(config)
db = Database(config.storage.data_dir / "db.sqlite")
pipeline = Pipeline(config, db)
executor = ThreadPoolExecutor(max_workers=1)
in_flight_jobs: set[str] = set()
in_flight_lock = threading.Lock()

app = FastAPI(title="Realtime Translate", version="0.1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost", "http://127.0.0.1", "https://www.youtube.com"],
    allow_origin_regex=r"https://.*\.youtube\.com",
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.middleware("http")
async def protect_local_api(request: Request, call_next):
    token = config.server.api_token
    if token and request.url.path.startswith("/api/"):
        supplied = request.headers.get("authorization", "")
        if not secrets.compare_digest(supplied, f"Bearer {token}"):
            return JSONResponse({"detail": "Unauthorized"}, status_code=401)
    return await call_next(request)

static_dir = Path(__file__).resolve().parent.parent / "web"
if static_dir.exists():
    app.mount("/static", StaticFiles(directory=static_dir), name="static")


def schedule_job(job_id: str, provider_override: str | None = None) -> None:
    submit_tracked(job_id, pipeline.run_job, job_id, provider_override)


def schedule_recovery(job: VideoJob) -> None:
    submit_tracked(job.id, pipeline.resume_or_recover, job, None)


def schedule_retranslate(
    job: VideoJob, languages: list[str], provider_override: str | None = None
) -> None:
    submit_tracked(job.id, pipeline.retranslate, job, languages, provider_override)


def submit_tracked(job_id: str, function, *args) -> bool:
    with in_flight_lock:
        if job_id in in_flight_jobs:
            return False
        in_flight_jobs.add(job_id)

    def run() -> None:
        try:
            function(*args)
        finally:
            with in_flight_lock:
                in_flight_jobs.discard(job_id)

    try:
        executor.submit(run)
    except Exception:
        with in_flight_lock:
            in_flight_jobs.discard(job_id)
        raise
    return True


@app.on_event("startup")
def recover_interrupted_jobs() -> None:
    active_statuses = [
        JobStatus.queued,
        JobStatus.downloading,
        JobStatus.transcribing,
        JobStatus.translating,
        JobStatus.exporting,
    ]
    for job in db.list_jobs_by_statuses(active_statuses):
        db.update_job_status(
            job.id,
            JobStatus.queued,
            progress_percent=max(1, job.progressPercent),
            progress_stage="queued",
            progress_message="Recovering interrupted job",
            progress_detail={"previousStatus": job.status.value},
        )
        recovered_job = db.get_job(job.id) or job
        schedule_recovery(recovered_job)


@app.get("/", response_class=HTMLResponse)
def index() -> FileResponse:
    return FileResponse(static_dir / "index.html")


@app.get("/api/health")
def health() -> dict:
    database_ok = True
    try:
        with db.connect() as conn:
            conn.execute("SELECT 1").fetchone()
    except Exception:
        database_ok = False
    ollama_ok = None
    if config.translation.primary_provider == "ollama":
        try:
            response = httpx.get(
                f"{config.translation.ollama_base_url.rstrip('/')}/api/tags", timeout=1.5
            )
            ollama_ok = response.is_success
        except Exception:
            ollama_ok = False
    return {
        "ok": database_ok and (ollama_ok is not False),
        "version": "0.1.0",
        "database": database_ok,
        "ollama": ollama_ok,
        "ffmpeg": shutil.which("ffmpeg") is not None,
        "ytDlp": shutil.which("yt-dlp") is not None,
        "diskFreeBytes": shutil.disk_usage(config.storage.work_dir).free,
    }


@app.get("/api/diagnostics/youtube-formats")
def diagnose_youtube_formats(url: str) -> dict:
    ingestion = YouTubeIngestion(config.storage.work_dir / "audio", config.youtube)
    formats = ingestion.list_formats(url)
    audio_formats = [item for item in formats if item.get("acodec") != "none"]
    return {
        "ok": bool(audio_formats),
        "config": {
            "cookiesFileConfigured": bool(config.youtube.cookies_file),
            "cookiesFileExists": bool(config.youtube.cookies_file and config.youtube.cookies_file.exists()),
            "jsRuntimes": config.youtube.js_runtimes,
            "workDir": str(config.storage.work_dir),
        },
        "totalFormats": len(formats),
        "audioFormats": audio_formats,
    }


@app.post("/api/jobs", response_model=JobResponse)
def create_job(request: CreateJobRequest, background: BackgroundTasks) -> JobResponse:
    video_id = extract_video_id(str(request.youtubeUrl))
    target_languages = dedupe_languages(request.targetLanguages)

    active_job = db.get_active_job_by_video(video_id)
    if active_job:
        with in_flight_lock:
            running = active_job.id in in_flight_jobs
        if not running:
            db.update_job_status(
                active_job.id,
                JobStatus.queued,
                progress_stage="queued",
                progress_message="Recovering job after interrupted worker or missing files",
                progress_detail={"cache": "recovery", "previousStatus": active_job.status.value},
            )
            active_job = db.get_job(active_job.id) or active_job
            background.add_task(schedule_recovery, active_job)
            return JobResponse(
                job=active_job,
                links={
                    **job_links(active_job),
                    "cache": "recovery",
                    "message": "Previous job state found; recovery was queued.",
                },
            )
        return JobResponse(
            job=active_job,
            links={
                **job_links(active_job),
                "cache": "active",
                "message": "An active job already exists for this video.",
            },
        )

    cached_job = db.get_latest_completed_job_for_languages(
        video_id, target_languages, CURRENT_FINGERPRINT
    )
    if cached_job:
        missing_translation_files = pipeline.languages_missing_translation_files(cached_job, target_languages)
        if missing_translation_files:
            for language in missing_translation_files:
                pipeline.reset_language_translation(cached_job, language)
            db.update_job_status(
                cached_job.id,
                JobStatus.queued,
                progress_percent=45,
                progress_stage="queued",
                progress_message="Translation cache file missing; queued retranslation",
                progress_detail={"cache": "stale", "missingTranslationFiles": missing_translation_files},
            )
            cached_job = db.get_job(cached_job.id) or cached_job
            background.add_task(schedule_retranslate, cached_job, missing_translation_files, None)
            return JobResponse(
                job=cached_job,
                links={
                    **job_links(cached_job),
                    "cache": "stale",
                    "message": "Transcript exists, but translation files are missing; retranslation was queued.",
                    "missingLanguages": ",".join(missing_translation_files),
                },
            )

        repaired = pipeline.repair_subtitle_files(cached_job)
        db.update_job_status(
            cached_job.id,
            JobStatus.done,
            progress_percent=100,
            progress_stage="done",
            progress_message="Using cached subtitles"
            if not repaired
            else "Using cached subtitles; repaired missing files",
            progress_detail={"cache": "hit", "languages": target_languages, "repaired": repaired},
        )
        cached_job = db.get_job(cached_job.id) or cached_job
        return JobResponse(
            job=cached_job,
            links={
                **job_links(cached_job),
                "cache": "hit",
                "message": "Existing subtitles found; no new job was created.",
            },
        )

    transcript_job = db.get_latest_job_with_transcript(video_id)
    if transcript_job:
        transcript_segment_ids = {
            segment.id for segment in db.list_transcript(transcript_job.id)
        }
        missing_languages = [
            language
            for language in target_languages
            if not transcript_segment_ids.issubset(
                db.translated_segment_ids(transcript_job.id, language)
            )
        ]
    else:
        missing_languages = target_languages

    if transcript_job and missing_languages:
        merged_languages = dedupe_languages([*transcript_job.targetLanguages, *missing_languages])
        db.update_job_target_languages(transcript_job.id, merged_languages)
        db.update_job_status(
            transcript_job.id,
            JobStatus.queued,
            progress_percent=45,
            progress_stage="queued",
            progress_message="Transcript cache found; queued missing languages",
            progress_detail={"cache": "partial", "missingLanguages": missing_languages},
        )
        transcript_job = db.get_job(transcript_job.id) or transcript_job
        transcript_job.targetLanguages = merged_languages
        background.add_task(schedule_retranslate, transcript_job, missing_languages, None)
        return JobResponse(
            job=transcript_job,
            links={
                **job_links(transcript_job),
                "cache": "partial",
                "message": "Transcript cache found; only missing target languages will be translated.",
                "missingLanguages": ",".join(missing_languages),
            },
        )

    if transcript_job and not missing_languages:
        repaired = pipeline.repair_subtitle_files(transcript_job)
        db.update_job_status(
            transcript_job.id,
            JobStatus.done,
            progress_percent=100,
            progress_stage="done",
            progress_message="Recovered completed translation",
            progress_detail={"cache": "recovered", "repaired": repaired},
        )
        transcript_job = db.get_job(transcript_job.id) or transcript_job
        return JobResponse(
            job=transcript_job,
            links={
                **job_links(transcript_job),
                "cache": "hit",
                "message": "Completed segment translations were recovered without download or ASR.",
            },
        )

    restored_job = pipeline.restore_from_file_cache(
        video_id=video_id,
        youtube_url=str(request.youtubeUrl),
        source_language=request.sourceLanguage,
        target_languages=target_languages,
    )
    if restored_job:
        return JobResponse(
            job=restored_job,
            links={
                **job_links(restored_job),
                "cache": "files",
                "message": "Existing transcript and translation files were restored; no download was needed.",
            },
        )

    now = utc_now_iso()
    job = VideoJob(
        id=str(uuid4()),
        youtubeUrl=str(request.youtubeUrl),
        videoId=video_id,
        sourceLanguage=request.sourceLanguage,
        targetLanguages=target_languages,
        pipelineFingerprint=CURRENT_FINGERPRINT,
        status=JobStatus.queued,
        createdAt=now,
        updatedAt=now,
    )
    db.save_job(job)
    background.add_task(schedule_job, job.id)
    return JobResponse(
        job=job,
        links={**job_links(job), "cache": "miss", "message": "No usable cache found; full job queued."},
    )


@app.get("/api/jobs/{job_id}", response_model=JobResponse)
def get_job(job_id: str) -> JobResponse:
    job = db.get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    return JobResponse(job=job, links=job_links(job))


@app.post("/api/jobs/{job_id}/recover", response_model=JobResponse)
def recover_job(job_id: str, background: BackgroundTasks) -> JobResponse:
    job = db.get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    with in_flight_lock:
        running = job.id in in_flight_jobs
    if running:
        raise HTTPException(status_code=409, detail="A worker is already running for this job")
    db.update_job_status(
        job.id,
        JobStatus.queued,
        progress_stage="queued",
        progress_message="Recovery queued",
        progress_detail={"cache": "recovery", "previousStatus": job.status.value},
    )
    recovered = db.get_job(job.id) or job
    background.add_task(schedule_recovery, recovered)
    return JobResponse(
        job=recovered,
        links={
            **job_links(recovered),
            "cache": "recovery",
            "message": "Job recovery was queued.",
        },
    )


@app.delete("/api/jobs/{job_id}")
def delete_job(job_id: str, purge_files: bool = Query(False)) -> dict:
    job = db.get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    if job.status in {
        JobStatus.queued,
        JobStatus.downloading,
        JobStatus.transcribing,
        JobStatus.translating,
        JobStatus.exporting,
    }:
        raise HTTPException(status_code=409, detail="Cannot delete an active job")
    if purge_files:
        safe_delete_job_files(job)
    db.delete_job(job_id)
    return {"deleted": True, "jobId": job_id, "purgedFiles": purge_files}


@app.post("/api/jobs/{job_id}/retranslate", response_model=JobResponse)
def retranslate(job_id: str, request: RetranslateRequest, background: BackgroundTasks) -> JobResponse:
    job = db.get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    languages = request.targetLanguages or job.targetLanguages
    if not submit_tracked(job.id, pipeline.retranslate, job, languages, request.provider):
        raise HTTPException(status_code=409, detail="A translation worker is already running for this job")
    return JobResponse(job=job, links=job_links(job))


@app.get("/api/jobs/{job_id}/transcript")
def get_transcript(job_id: str) -> dict:
    job = db.get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    return {"segments": [segment.model_dump() for segment in db.list_transcript(job_id)]}


@app.get("/api/transcripts/{video_id}")
def get_transcript_by_video(video_id: str) -> dict:
    job = db.get_latest_job_by_video(video_id)
    if not job:
        raise HTTPException(status_code=404, detail="Video job not found")
    segments = db.list_transcript(job.id)
    if not segments:
        raise HTTPException(status_code=404, detail="No transcript found for this video")
    cues = [
        {
            "startMs": segment.startMs,
            "endMs": segment.endMs,
            "text": segment.normalizedText or segment.sourceText,
            "language": job.detectedLanguage or job.sourceLanguage,
        }
        for segment in segments
    ]
    return {"videoId": video_id, "language": job.detectedLanguage or job.sourceLanguage, "cues": cues}


@app.get("/api/jobs/{job_id}/translations/{lang}")
def get_translations(job_id: str, lang: str) -> dict:
    job = db.get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    return {"segments": [segment.model_dump() for segment in db.list_translations(job_id, lang)]}


@app.get("/api/jobs/{job_id}/translation-stream/{lang}")
def get_translation_stream(job_id: str, lang: str) -> dict:
    job = db.get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    return translation_stream_payload(job, lang)


@app.get("/api/subtitles/{video_id}")
def get_subtitles(
    video_id: str,
    lang: str = Query("zh-TW"),
    format: str = Query("json", pattern="^(json|vtt|srt|ass)$"),
) -> Response:
    job = db.get_latest_completed_job_for_languages(video_id, [lang])
    if job:
        pipeline.repair_subtitle_files(job)
    cues = db.list_cues(video_id, lang)
    if not cues:
        raise HTTPException(status_code=404, detail="No subtitles found for this video/language")
    if format == "json":
        return JSONResponse({"videoId": video_id, "language": lang, "cues": [c.model_dump() for c in cues]})
    if format == "vtt":
        return PlainTextResponse(export_vtt(cues), media_type="text/vtt; charset=utf-8")
    if format == "srt":
        return PlainTextResponse(export_srt(cues), media_type="application/x-subrip; charset=utf-8")
    return PlainTextResponse(export_ass(cues), media_type="text/plain; charset=utf-8")


@app.put("/api/videos/{video_id}/glossary")
def put_glossary(video_id: str, entries: list[GlossaryEntry]) -> dict:
    db.set_video_glossary(video_id, entries)
    return {"videoId": video_id, "entries": len(entries)}


@app.get("/api/videos/{video_id}/glossary")
def get_glossary(video_id: str) -> dict:
    job = db.get_latest_job_by_video(video_id)
    channel = job.channel if job else None
    return {"videoId": video_id, "entries": [e.model_dump() for e in db.get_glossary(video_id=video_id, channel=channel)]}


@app.websocket("/ws/subtitles/{video_id}")
async def websocket_subtitles(websocket: WebSocket, video_id: str, lang: str = "zh-TW") -> None:
    if not websocket_authorized(websocket):
        await websocket.close(code=1008)
        return
    await websocket.accept()
    last_payload = None
    try:
        while True:
            cues = db.list_cues(video_id, lang)
            payload = {"videoId": video_id, "language": lang, "cues": [c.model_dump() for c in cues]}
            if payload != last_payload:
                await websocket.send_json(payload)
                last_payload = payload
            await asyncio.sleep(1)
    except Exception:
        await websocket.close()


@app.websocket("/ws/jobs/{job_id}/translations/{lang}")
async def websocket_job_translations(websocket: WebSocket, job_id: str, lang: str) -> None:
    if not websocket_authorized(websocket):
        await websocket.close(code=1008)
        return
    await websocket.accept()
    last_payload = None
    try:
        while True:
            job = db.get_job(job_id)
            if not job:
                await websocket.send_json({"error": "Job not found"})
                return
            payload = translation_stream_payload(job, lang)
            if payload != last_payload:
                await websocket.send_json(payload)
                last_payload = payload
            if job.status in {JobStatus.done, JobStatus.failed}:
                return
            await asyncio.sleep(0.5)
    except WebSocketDisconnect:
        return


def translation_stream_payload(job: VideoJob, language: str) -> dict:
    transcript = db.list_transcript(job.id)
    translations = {
        item.segmentId: item for item in db.list_translations(job.id, language)
    }
    segments = [
        {
            "id": segment.id,
            "index": segment.index,
            "startMs": segment.startMs,
            "endMs": segment.endMs,
            "sourceText": segment.normalizedText or segment.sourceText,
            "translatedText": (
                translations[segment.id].polishedText
                or translations[segment.id].translatedText
            )
            if segment.id in translations
            else None,
        }
        for segment in transcript
    ]
    completed = sum(1 for segment in segments if segment["translatedText"] is not None)
    total = len(segments)
    return {
        "jobId": job.id,
        "videoId": job.videoId,
        "language": language,
        "status": job.status.value,
        "progressStage": job.progressStage,
        "completed": completed,
        "total": total,
        "percent": round(100 * completed / max(1, total)),
        "segments": segments,
    }


def websocket_authorized(websocket: WebSocket) -> bool:
    token = config.server.api_token
    if not token:
        return True
    supplied = websocket.headers.get("authorization", "")
    query_token = websocket.query_params.get("token", "")
    return secrets.compare_digest(supplied, f"Bearer {token}") or secrets.compare_digest(
        query_token, token
    )


def safe_delete_job_files(job: VideoJob) -> None:
    root = config.storage.work_dir.resolve()
    candidates = [
        Path(job.audioPath) if job.audioPath else None,
        root / "transcripts" / f"{job.videoId}.source.json",
    ]
    for language in job.targetLanguages:
        candidates.extend(
            [
                root / "translations" / f"{job.videoId}.{language}.json",
                *[root / "subtitles" / f"{job.videoId}.{language}.{fmt}" for fmt in config.subtitle.formats],
            ]
        )
    for candidate in candidates:
        if not candidate:
            continue
        try:
            resolved = candidate.resolve()
            if root == resolved or root in resolved.parents:
                resolved.unlink(missing_ok=True)
        except OSError:
            continue


def job_links(job: VideoJob) -> dict[str, str]:
    links: dict[str, str] = {"self": f"/api/jobs/{job.id}", "preview": f"/?videoId={job.videoId}"}
    for language in job.targetLanguages:
        links[f"subtitles_{language}_json"] = f"/api/subtitles/{job.videoId}?lang={language}&format=json"
        links[f"subtitles_{language}_vtt"] = f"/api/subtitles/{job.videoId}?lang={language}&format=vtt"
        links[f"subtitles_{language}_srt"] = f"/api/subtitles/{job.videoId}?lang={language}&format=srt"
    return links


def dedupe_languages(languages: list[str]) -> list[str]:
    result: list[str] = []
    for language in languages:
        normalized = language.strip()
        if normalized and normalized not in result:
            result.append(normalized)
    return result or ["zh-TW"]


def main() -> None:
    import uvicorn

    uvicorn.run("realtime_translate.api:app", host=config.server.host, port=config.server.port, reload=False)


if __name__ == "__main__":
    main()
