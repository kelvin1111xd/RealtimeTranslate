from __future__ import annotations

import argparse
import time
from uuid import uuid4

from .config import load_config
from .db import Database
from .pipeline import Pipeline
from .schemas import JobStatus, VideoJob, utc_now_iso
from .youtube import YouTubeIngestion, extract_video_id


def main() -> None:
    parser = argparse.ArgumentParser(description="Realtime Translate CLI")
    subparsers = parser.add_subparsers(dest="command", required=True)

    run_parser = subparsers.add_parser("run", help="Run a YouTube subtitle job synchronously")
    run_parser.add_argument("youtube_url")
    run_parser.add_argument("--source", default="auto")
    run_parser.add_argument("--targets", default="zh-TW")
    run_parser.add_argument("--provider", default=None, choices=["ollama", "openai_compatible", "passthrough"])

    formats_parser = subparsers.add_parser("formats", help="List available yt-dlp formats for a URL")
    formats_parser.add_argument("youtube_url")

    args = parser.parse_args()
    config = load_config()
    if args.command == "formats":
        ingestion = YouTubeIngestion(config.storage.work_dir / "audio", config.youtube)
        formats = ingestion.list_formats(args.youtube_url)
        audio_formats = [item for item in formats if item.get("acodec") != "none"]
        for item in audio_formats or formats:
            print(
                f"{item.get('format_id')}\t{item.get('ext')}\t"
                f"a:{item.get('acodec')}\tv:{item.get('vcodec')}\t"
                f"abr:{item.get('abr')}\t{item.get('format_note')}"
            )
        print(f"total_formats={len(formats)} audio_or_muxed={len(audio_formats)}")
        return

    if args.command == "run":
        db = Database(config.storage.data_dir / "db.sqlite")
        video_id = extract_video_id(args.youtube_url)
        now = utc_now_iso()
        job = VideoJob(
            id=str(uuid4()),
            youtubeUrl=args.youtube_url,
            videoId=video_id,
            sourceLanguage=args.source,
            targetLanguages=[t.strip() for t in args.targets.split(",") if t.strip()],
            status=JobStatus.queued,
            createdAt=now,
            updatedAt=now,
        )
        db.save_job(job)
        Pipeline(config, db).run_job(job.id, provider_override=args.provider)
        while True:
            current = db.get_job(job.id)
            print(current.model_dump_json(indent=2) if current else "missing job")
            if current and current.status in {JobStatus.done, JobStatus.failed}:
                break
            time.sleep(1)


if __name__ == "__main__":
    main()
