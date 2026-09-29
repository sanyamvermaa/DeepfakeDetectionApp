"""
Download utilities for analyzing videos from a YouTube/Instagram URL.
Isolates yt-dlp usage from server.py, mirroring how audio_processor.py
and video_processor.py isolate ffmpeg/cv2 concerns.
"""
import os
import glob
import shutil
import tempfile
import logging
from urllib.parse import urlparse

import yt_dlp
import imageio_ffmpeg

from config import Config

logger = logging.getLogger(__name__)


class UrlDownloadError(Exception):
    """Base class for all URL-download errors. Carries an HTTP-ish status code."""
    status_code = 400


class UnsupportedHostError(UrlDownloadError):
    status_code = 400


class VideoUnavailableError(UrlDownloadError):
    status_code = 422


class VideoNotFoundError(UrlDownloadError):
    status_code = 404


class VideoTooLongError(UrlDownloadError):
    status_code = 413


class DownloadTimeoutError(UrlDownloadError):
    status_code = 504


def validate_host(url: str) -> None:
    """Raise UnsupportedHostError unless the URL's host is on the allow-list."""
    try:
        host = (urlparse(url).hostname or "").lower()
    except Exception:
        host = ""

    if not host or not any(host == h or host.endswith("." + h) for h in Config.ALLOWED_URL_HOSTS):
        raise UnsupportedHostError("Only YouTube and Instagram links are supported")


# Internal alias kept for readability at call sites within this module.
_check_allowed_host = validate_host


def _is_instagram(url: str) -> bool:
    host = (urlparse(url).hostname or "").lower()
    return "instagram.com" in host


def _base_ydl_opts() -> dict:
    return {
        "quiet": True,
        "no_warnings": True,
        "noplaylist": True,
        "socket_timeout": Config.URL_DOWNLOAD_TIMEOUT_SEC,
        # Modern YouTube mostly serves separate video/audio DASH streams —
        # merging them needs ffmpeg. No system ffmpeg is guaranteed to be on
        # PATH, so point yt-dlp at the one bundled with imageio-ffmpeg
        # (already a project dependency, used by AudioProcessor too).
        "ffmpeg_location": imageio_ffmpeg.get_ffmpeg_exe(),
        # Only allow yt-dlp's dedicated extractors — never fall back to the
        # generic extractor, which would turn this into an arbitrary-URL fetcher.
        "allowed_extractors": ["youtube", "youtube:*", "instagram", "instagram:*"],
    }


def probe_video(url: str) -> dict:
    """
    Fetch metadata only (no download). Returns {"duration": seconds, "title": str}.
    Raises UnsupportedHostError / VideoUnavailableError / VideoNotFoundError / VideoTooLongError.
    """
    _check_allowed_host(url)

    opts = _base_ydl_opts()
    try:
        with yt_dlp.YoutubeDL(opts) as ydl:
            info = ydl.extract_info(url, download=False)
    except yt_dlp.utils.DownloadError as e:
        raise _map_download_error(e, url) from e

    duration = info.get("duration") or 0
    if duration and duration > Config.MAX_URL_VIDEO_DURATION_SEC:
        raise VideoTooLongError(
            f"Video is {int(duration)}s long, max allowed is {Config.MAX_URL_VIDEO_DURATION_SEC}s"
        )

    return {"duration": duration, "title": info.get("title") or ""}


def download_video(url: str) -> str:
    """
    Download the video to a temp file and return its local path.
    Caller is responsible for deleting the file when done.
    """
    _check_allowed_host(url)

    tmp_dir = tempfile.mkdtemp(prefix="url_dl_")
    out_template = os.path.join(tmp_dir, "video.%(ext)s")

    opts = _base_ydl_opts()
    opts.update({
        "format": Config.URL_DOWNLOAD_FORMAT,
        "outtmpl": out_template,
        "merge_output_format": "mp4",
        "max_filesize": Config.MAX_URL_VIDEO_FILESIZE_MB * 1024 * 1024,
    })

    try:
        try:
            with yt_dlp.YoutubeDL(opts) as ydl:
                ydl.extract_info(url, download=True)
        except yt_dlp.utils.DownloadError as e:
            raise _map_download_error(e, url) from e

        # Merging bestvideo+bestaudio can change the final extension (e.g. to
        # .mkv if mp4 muxing isn't possible), so find the actual output rather
        # than trust a pre-merge filename guess.
        candidates = glob.glob(os.path.join(tmp_dir, "video.*"))
        if not candidates:
            raise VideoUnavailableError("Download completed but no file was produced")

        return candidates[0]
    except Exception:
        # Clean up on any failure — a partial/failed download can leave
        # fragment files behind, and an empty dir otherwise leaks forever.
        shutil.rmtree(tmp_dir, ignore_errors=True)
        raise


def _map_download_error(e: "yt_dlp.utils.DownloadError", url: str) -> UrlDownloadError:
    msg = str(e).lower()

    if "timed out" in msg or "timeout" in msg:
        return DownloadTimeoutError("Download timed out")

    if any(s in msg for s in ("private", "login", "rate-limit", "restricted", "sign in", "not available in your country")):
        if _is_instagram(url):
            return VideoUnavailableError(
                "This Instagram post may be private or require login — not all Instagram content can be fetched"
            )
        return VideoUnavailableError("This video isn't publicly accessible")

    if any(s in msg for s in ("unable to download webpage", "404", "unavailable", "does not exist", "has been removed")):
        return VideoNotFoundError("Video not found or has been removed")

    # Generic Instagram failures are common enough (fragile extractor) to call out specifically.
    if _is_instagram(url):
        return VideoUnavailableError(
            "Couldn't fetch this Instagram content — Instagram support is best-effort and frequently breaks"
        )

    return VideoUnavailableError(f"Could not download video: {e}")
