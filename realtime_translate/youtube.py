from __future__ import annotations

import re
import subprocess
from pathlib import Path
from urllib.parse import parse_qs, urlparse

from yt_dlp import YoutubeDL

from .config import YouTubeConfig


def extract_video_id(url: str) -> str:
    parsed = urlparse(url)
    if parsed.hostname in {"youtu.be", "www.youtu.be"}:
        return parsed.path.strip("/")
    if parsed.hostname and "youtube.com" in parsed.hostname:
        query_id = parse_qs(parsed.query).get("v")
        if query_id:
            return query_id[0]
        match = re.search(r"/(?:shorts|live)/([^/?#]+)", parsed.path)
        if match:
            return match.group(1)
    raise ValueError("Could not extract a YouTube video id from the URL.")


class YouTubeIngestion:
    def __init__(self, audio_dir: Path, config: YouTubeConfig | None = None):
        self.audio_dir = audio_dir
        self.config = config or YouTubeConfig()
        self.audio_dir.mkdir(parents=True, exist_ok=True)

    def fetch_metadata(self, url: str) -> dict:
        options = self._base_options()
        options.update({"quiet": True, "skip_download": True, "noplaylist": True})
        with YoutubeDL(options) as ydl:
            return ydl.extract_info(url, download=False)

    def list_formats(self, url: str) -> list[dict]:
        info = self.fetch_metadata(url)
        formats = info.get("formats") or []
        return [
            {
                "format_id": item.get("format_id"),
                "ext": item.get("ext"),
                "acodec": item.get("acodec"),
                "vcodec": item.get("vcodec"),
                "abr": item.get("abr"),
                "filesize": item.get("filesize") or item.get("filesize_approx"),
                "format_note": item.get("format_note"),
            }
            for item in formats
        ]

    def download_audio(self, url: str, video_id: str) -> Path:
        output_template = str(self.audio_dir / f"{video_id}.%(ext)s")
        options = self._base_options()
        options.update({
            "format": "bestaudio/best",
            "outtmpl": output_template,
            "postprocessors": [
                {
                    "key": "FFmpegExtractAudio",
                    "preferredcodec": "wav",
                    "preferredquality": "0",
                }
            ],
            "noplaylist": True,
            "quiet": False,
        })
        with YoutubeDL(options) as ydl:
            ydl.download([url])
        wav_path = self.audio_dir / f"{video_id}.wav"
        if not wav_path.exists():
            raise FileNotFoundError(f"yt-dlp did not produce expected audio file: {wav_path}")
        return wav_path

    def _base_options(self) -> dict:
        options: dict = {}
        if self.config.cookies_file:
            if not self.config.cookies_file.exists():
                raise FileNotFoundError(
                    f"YouTube cookies file is configured but does not exist: {self.config.cookies_file}"
                )
            options["cookiefile"] = str(self.config.cookies_file)
        elif self.config.cookies_from_browser:
            profile = self.config.browser_profile
            options["cookiesfrombrowser"] = (
                self.config.cookies_from_browser,
                profile,
                None,
                None,
            )
        if self.config.js_runtimes:
            runtimes = {}
            for runtime in self.config.js_runtimes:
                name, _, path = runtime.partition(":")
                runtimes[name] = {"path": path} if path else {}
            options["js_runtimes"] = runtimes
        return options


def normalize_audio(input_path: Path, output_path: Path) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "ffmpeg",
        "-y",
        "-i",
        str(input_path),
        "-ac",
        "1",
        "-ar",
        "16000",
        "-vn",
        str(output_path),
    ]
    subprocess.run(cmd, check=True)
    return output_path
