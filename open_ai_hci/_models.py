"""Locate / download the MediaPipe ``.task`` model bundles used by the Tasks backend.

The legacy ``mp.solutions`` backend ships its models inside the wheel and never
touches this module. The Tasks backend needs files that are downloaded once and
cached (default ``~/.cache/open_ai_hci/models``; override with the
``OPEN_AI_HCI_HOME`` environment variable or ``MediaPipeConfig.model_dir``).

Only the Python standard library is used for downloading.
"""
from __future__ import annotations

import logging
import os
import tempfile
import urllib.error
import urllib.request
from pathlib import Path

__all__ = ["MODEL_URLS", "ModelUnavailableError", "cache_dir", "ensure_model"]

logger = logging.getLogger("open_ai_hci")

_BASE = "https://storage.googleapis.com/mediapipe-models"

#: Official MediaPipe model bundles (float16).
MODEL_URLS: dict[str, str] = {
    "hand": f"{_BASE}/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task",
    "face": f"{_BASE}/face_landmarker/face_landmarker/float16/1/face_landmarker.task",
    "pose_lite": f"{_BASE}/pose_landmarker/pose_landmarker_lite/float16/1/pose_landmarker_lite.task",
    "pose_full": f"{_BASE}/pose_landmarker/pose_landmarker_full/float16/1/pose_landmarker_full.task",
    "pose_heavy": f"{_BASE}/pose_landmarker/pose_landmarker_heavy/float16/1/pose_landmarker_heavy.task",
}

_MIN_BYTES = 100_000  # anything smaller is certainly an error page, not a model


class ModelUnavailableError(RuntimeError):
    """The model file is missing and could not be downloaded."""


def cache_dir(override: str | None = None) -> Path:
    """Directory where downloaded models live."""
    if override:
        return Path(override).expanduser()
    home = os.environ.get("OPEN_AI_HCI_HOME")
    base = Path(home).expanduser() if home else Path.home() / ".cache" / "open_ai_hci"
    return base / "models"


def ensure_model(
    kind: str,
    explicit_path: str | None = None,
    model_dir: str | None = None,
    auto_download: bool = True,
) -> Path:
    """Return a local path to the model for ``kind`` (a key of :data:`MODEL_URLS`).

    Resolution order: ``explicit_path`` -> cached file -> download.

    Raises:
        ModelUnavailableError: with manual-download instructions when the file is
            missing and cannot (or must not) be downloaded.
    """
    if kind not in MODEL_URLS:
        raise ValueError(f"Unknown model kind {kind!r}; expected one of {sorted(MODEL_URLS)}")

    if explicit_path:
        path = Path(explicit_path).expanduser()
        if not path.is_file():
            raise ModelUnavailableError(f"Configured {kind} model not found: {path}")
        return path

    url = MODEL_URLS[kind]
    target_dir = cache_dir(model_dir)
    path = target_dir / Path(url).name
    if path.is_file() and path.stat().st_size >= _MIN_BYTES:
        return path

    option = f"MediaPipeConfig.{kind.split('_')[0]}_model_path"  # pose_lite/full/heavy -> pose_model_path
    manual = f"Download {url}\nand either place it at {path}\nor point {option} at the downloaded file."
    if not auto_download:
        raise ModelUnavailableError(f"MediaPipe {kind} model is not cached and auto_download is off.\n{manual}")

    logger.info("Downloading MediaPipe %s model (one-time) from %s", kind, url)
    target_dir.mkdir(parents=True, exist_ok=True)
    tmp_name: str | None = None
    try:
        with urllib.request.urlopen(url, timeout=60) as resp, tempfile.NamedTemporaryFile(
            dir=target_dir, suffix=".part", delete=False
        ) as tmp:
            tmp_name = tmp.name
            while chunk := resp.read(1 << 16):
                tmp.write(chunk)
        if os.path.getsize(tmp_name) < _MIN_BYTES:
            raise ModelUnavailableError("downloaded file is implausibly small")
        os.replace(tmp_name, path)
        tmp_name = None
    except (urllib.error.URLError, OSError, ModelUnavailableError) as exc:
        raise ModelUnavailableError(f"Could not download the MediaPipe {kind} model ({exc}).\n{manual}") from exc
    finally:
        if tmp_name and os.path.exists(tmp_name):
            os.unlink(tmp_name)
    return path
