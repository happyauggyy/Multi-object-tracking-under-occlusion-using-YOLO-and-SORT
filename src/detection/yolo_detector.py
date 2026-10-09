"""YOLO Person Detector abstraction for Multi-Object Tracking."""

from typing import List, Optional, Union
import numpy as np


class YOLODetector:
    """Wrapper around Ultralytics YOLO to detect people (COCO class 0)."""

    def __init__(
        self,
        model_name: str = "yolo11n.pt",
        conf_thresh: float = 0.40,
        person_class_id: int = 0,
        device: str = "",
        imgsz: Optional[int] = None,
    ):
        """
        Initialize the YOLO detector.

        Args:
            model_name: Path or name of the pretrained YOLO model (e.g., 'yolo11n.pt', 'yolov8n.pt').
            conf_thresh: Confidence threshold for person detections (default: 0.40).
            person_class_id: Class ID for person in COCO dataset (default: 0).
            device: Computing device ('cpu', 'cuda:0', or empty string for auto-detection).
            imgsz: Optional inference image size (e.g. 640, 960, 1280). Default: model default.
        """
        self.model_name = model_name
        self.conf_thresh = float(conf_thresh)
        self.person_class_id = int(person_class_id)
        self.device = device if device else None
        self.imgsz = int(imgsz) if (imgsz is not None and int(imgsz) > 0) else None
        self._model = None

    def _load_model(self):
        """Lazy load the YOLO model."""
        if self._model is None:
            try:
                from ultralytics import YOLO
                self._model = YOLO(self.model_name)
            except Exception as e:
                # If specified model fails (e.g., specific version not found), attempt yolov8n.pt
                if self.model_name != "yolov8n.pt":
                    try:
                        from ultralytics import YOLO
                        print(f"[YOLODetector] Failed to load {self.model_name} ({e}). Falling back to yolov8n.pt...")
                        self.model_name = "yolov8n.pt"
                        self._model = YOLO("yolov8n.pt")
                    except Exception as fallback_err:
                        raise RuntimeError(
                            f"Failed to load YOLO model '{self.model_name}' and fallback 'yolov8n.pt': {fallback_err}"
                        ) from e
                else:
                    raise RuntimeError(f"Failed to load YOLO model '{self.model_name}': {e}") from e

    def detect(self, frame: np.ndarray) -> np.ndarray:
        """
        Run person detection on a single frame.

        Args:
            frame: Input image/frame in BGR format (numpy array).

        Returns:
            np.ndarray of shape (N, 5): [[x1, y1, x2, y2, score], ...]
            If no persons are detected, returns an empty array of shape (0, 5).
        """
        self._load_model()
        if frame is None or frame.size == 0:
            return np.empty((0, 5), dtype=np.float32)

        kwargs = {
            "conf": self.conf_thresh,
            "classes": [self.person_class_id],
            "verbose": False,
        }
        if self.device is not None:
            kwargs["device"] = self.device
        if self.imgsz is not None:
            kwargs["imgsz"] = self.imgsz

        results = self._model(frame, **kwargs)
        if not results:
            return np.empty((0, 5), dtype=np.float32)

        result = results[0]
        boxes = result.boxes
        if boxes is None or len(boxes) == 0:
            return np.empty((0, 5), dtype=np.float32)

        xyxy = boxes.xyxy.cpu().numpy()
        conf = boxes.conf.cpu().numpy().reshape(-1, 1)

        # Concatenate bounding box coordinates and confidence score
        detections = np.hstack([xyxy, conf]).astype(np.float32)
        return detections
