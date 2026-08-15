# ### Imports ###
from __future__ import annotations

import json
import math
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np


# ### Constants ###
VIDEO_FILE_FILTER = (
    "Video Files (*.mp4 *.mov *.avi *.mkv *.webm *.m4v);;All Files (*)"
)
DEFAULT_VIDEO_FPS = 30.0
DEFAULT_FRAME_CACHE_SIZE = 12


# ### Video models ###
@dataclass(frozen=True)
class VideoMetadata:
    """Metadata needed to address source-video frames deterministically."""

    path: str
    frame_count: int
    fps: float
    width: int
    height: int

    def __post_init__(self) -> None:
        if int(self.frame_count) < 0:
            raise ValueError("Video frame count cannot be negative.")
        if not math.isfinite(float(self.fps)) or float(self.fps) <= 0.0:
            raise ValueError("Video FPS must be positive and finite.")
        if int(self.width) <= 0 or int(self.height) <= 0:
            raise ValueError("Video dimensions must be positive.")

    @property
    def source_path(self) -> str:
        return self.path

    @property
    def width_pixels(self) -> int:
        return self.width

    @property
    def height_pixels(self) -> int:
        return self.height

    @property
    def duration_seconds(self) -> float:
        return float(self.frame_count) / float(self.fps)

    def timestamp_for_frame(self, frame_index: int) -> float:
        if frame_index < 0 or frame_index >= self.frame_count:
            raise IndexError("Video frame index is outside the video.")
        return float(frame_index) / float(self.fps)

    def to_dict(self) -> dict[str, str | int | float]:
        return {
            "path": self.path,
            "frame_count": int(self.frame_count),
            "fps": float(self.fps),
            "width": int(self.width),
            "height": int(self.height),
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "VideoMetadata":
        return cls(
            path=str(payload.get("path", payload.get("source_path", ""))),
            frame_count=int(payload["frame_count"]),
            fps=float(payload["fps"]),
            width=int(payload.get("width", payload.get("width_pixels"))),
            height=int(payload.get("height", payload.get("height_pixels"))),
        )

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), sort_keys=True)

    @classmethod
    def from_json(cls, payload: str) -> "VideoMetadata":
        decoded = json.loads(payload)
        if not isinstance(decoded, dict):
            raise ValueError("Video metadata JSON must contain an object.")
        return cls.from_dict(decoded)


# ### Video sources ###
class VideoFrameSource:
    """Lazy, seekable OpenCV video reader with a small frame cache."""

    def __init__(
        self,
        video_path: str | Path,
        cache_size: int = DEFAULT_FRAME_CACHE_SIZE,
    ) -> None:
        self.video_path = str(Path(video_path).resolve())
        self.cache_size = max(1, int(cache_size))
        self._capture = cv2.VideoCapture(self.video_path)
        if not self._capture.isOpened():
            self._capture.release()
            raise ValueError(f"Unable to open video file: {self.video_path}")

        self.metadata = _read_video_metadata(self._capture, self.video_path)
        self._frame_cache: OrderedDict[int, np.ndarray] = OrderedDict()
        self._next_frame_index = 0

    def get_frame(self, frame_index: int) -> np.ndarray:
        normalized_index = self._validate_frame_index(frame_index)
        cached_frame = self._frame_cache.get(normalized_index)
        if cached_frame is not None:
            self._frame_cache.move_to_end(normalized_index)
            return cached_frame.copy()

        if normalized_index != self._next_frame_index:
            self._capture.set(cv2.CAP_PROP_POS_FRAMES, normalized_index)

        did_read, frame = self._capture.read()
        if not did_read or frame is None:
            raise ValueError(
                f"Unable to read frame {normalized_index} from: {self.video_path}"
            )

        normalized_frame = normalize_video_frame(frame)
        self._next_frame_index = normalized_index + 1
        self._cache_frame(normalized_index, normalized_frame)
        return normalized_frame.copy()

    def timestamp_seconds(self, frame_index: int) -> float:
        normalized_index = self._validate_frame_index(frame_index)
        return normalized_index / max(self.metadata.fps, 1e-6)

    def close(self) -> None:
        if self._capture.isOpened():
            self._capture.release()
        self._frame_cache.clear()

    def __enter__(self) -> "VideoFrameSource":
        return self

    def __exit__(self, *_args: object) -> None:
        self.close()

    def _validate_frame_index(self, frame_index: int) -> int:
        normalized_index = int(frame_index)
        if normalized_index < 0 or normalized_index >= self.metadata.frame_count:
            raise IndexError(
                f"Frame index {normalized_index} is outside the video range."
            )
        return normalized_index

    def _cache_frame(self, frame_index: int, frame: np.ndarray) -> None:
        self._frame_cache[frame_index] = frame.copy()
        self._frame_cache.move_to_end(frame_index)
        while len(self._frame_cache) > self.cache_size:
            self._frame_cache.popitem(last=False)


# ### Public helpers ###
def probe_video(video_path: str | Path) -> VideoMetadata:
    normalized_path = str(Path(video_path).resolve())
    capture = cv2.VideoCapture(normalized_path)
    if not capture.isOpened():
        capture.release()
        raise ValueError(f"Unable to open video file: {normalized_path}")

    try:
        return _read_video_metadata(capture, normalized_path)
    finally:
        capture.release()


def normalize_video_frame(frame: np.ndarray) -> np.ndarray:
    if frame.ndim == 2:
        return cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
    if frame.ndim != 3:
        raise ValueError("Video frames must be two- or three-dimensional arrays.")
    if frame.shape[2] == 4:
        return cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
    if frame.shape[2] != 3:
        raise ValueError("Video frames must contain one, three, or four channels.")

    return np.ascontiguousarray(frame)


# ### Metadata helpers ###
def _read_video_metadata(
    capture: cv2.VideoCapture,
    video_path: str,
) -> VideoMetadata:
    frame_count = max(0, int(round(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0.0)))
    fps = float(capture.get(cv2.CAP_PROP_FPS) or 0.0)
    width = max(0, int(round(capture.get(cv2.CAP_PROP_FRAME_WIDTH) or 0.0)))
    height = max(0, int(round(capture.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0.0)))

    if frame_count <= 0:
        raise ValueError(f"Video does not contain readable frames: {video_path}")
    if width <= 0 or height <= 0:
        raise ValueError(f"Video has invalid dimensions: {video_path}")
    if not np.isfinite(fps) or fps <= 0.0:
        fps = DEFAULT_VIDEO_FPS

    return VideoMetadata(
        path=video_path,
        frame_count=frame_count,
        fps=fps,
        width=width,
        height=height,
    )
