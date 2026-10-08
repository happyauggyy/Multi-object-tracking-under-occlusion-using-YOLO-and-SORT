"""Video and image sequence processing utilities."""

import glob
import os
from typing import Generator, List, Optional, Tuple
import cv2
import numpy as np


class VideoReader:
    """
    Unified reader for either video files (.mp4, .avi, etc.) or image sequence directories
    (e.g., MOT17 img1/ containing sequential JPEG/PNG files).
    """

    def __init__(self, input_path: str):
        """
        Initialize video or sequence reader.

        Args:
            input_path: Path to video file or folder containing image sequence.
        """
        self.input_path = input_path
        self.is_image_folder = os.path.isdir(input_path)
        self.image_files: List[str] = []
        self.cap: Optional[cv2.VideoCapture] = None

        if self.is_image_folder:
            # Check for img1 subfolder (MOT benchmark layout) or direct images
            img_dir = os.path.join(input_path, "img1") if os.path.isdir(os.path.join(input_path, "img1")) else input_path
            patterns = ["*.jpg", "*.jpeg", "*.png", "*.bmp"]
            for pat in patterns:
                self.image_files.extend(glob.glob(os.path.join(img_dir, pat)))
            self.image_files.sort()

            if not self.image_files:
                raise ValueError(f"No image files found in directory: {input_path}")

            sample = cv2.imread(self.image_files[0])
            if sample is None:
                raise ValueError(f"Failed to load sample image: {self.image_files[0]}")
            self.height, self.width = sample.shape[:2]
            self.fps = 30.0  # Default assumed FPS for sequence if unspecified
            self.total_frames = len(self.image_files)
        else:
            if not os.path.exists(input_path):
                raise FileNotFoundError(f"Input video file not found: {input_path}")
            self.cap = cv2.VideoCapture(input_path)
            if not self.cap.isOpened():
                raise ValueError(f"Failed to open video file: {input_path}")
            self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps_val = self.cap.get(cv2.CAP_PROP_FPS)
            self.fps = fps_val if fps_val > 0 else 30.0
            self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))

    def __iter__(self) -> Generator[Tuple[int, np.ndarray], None, None]:
        """
        Yields (frame_idx_1_indexed, frame_bgr).
        """
        if self.is_image_folder:
            for idx, img_path in enumerate(self.image_files, start=1):
                frame = cv2.imread(img_path)
                if frame is not None:
                    yield idx, frame
        else:
            frame_idx = 1
            while True:
                ret, frame = self.cap.read()
                if not ret or frame is None:
                    break
                yield frame_idx, frame
                frame_idx += 1

    def close(self) -> None:
        """Release underlying resources."""
        if self.cap is not None:
            self.cap.release()
            self.cap = None


class VideoWriter:
    """Helper to write output frames into an encoded video file."""

    def __init__(self, output_path: str, fps: float, width: int, height: int):
        """
        Initialize VideoWriter.

        Args:
            output_path: Destination path for .mp4 or .avi.
            fps: Frame rate.
            width: Frame width in pixels.
            height: Frame height in pixels.
        """
        self.output_path = output_path
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        self.writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    def write(self, frame: np.ndarray) -> None:
        """Write a BGR frame to video."""
        self.writer.write(frame)

    def release(self) -> None:
        """Finalize video encoding."""
        self.writer.release()
