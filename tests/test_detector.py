"""Unit tests for YOLODetector and inference configuration options."""

import numpy as np
import pytest

from src.detection.yolo_detector import YOLODetector


def test_yolo_detector_initialization():
    """Verify default initialization parameters."""
    det = YOLODetector(model_name="yolo11n.pt", conf_thresh=0.35)
    assert det.model_name == "yolo11n.pt"
    assert det.conf_thresh == 0.35
    assert det.person_class_id == 0
    assert det.imgsz is None


def test_yolo_detector_custom_imgsz():
    """Verify custom image size setting."""
    det = YOLODetector(model_name="yolo11n.pt", conf_thresh=0.35, imgsz=960)
    assert det.imgsz == 960

    det_disabled = YOLODetector(model_name="yolo11n.pt", imgsz=-1)
    assert det_disabled.imgsz is None


def test_yolo_detector_empty_frame():
    """Verify empty or None input handling."""
    det = YOLODetector(model_name="yolo11n.pt", conf_thresh=0.35)
    out_none = det.detect(None)
    assert out_none.shape == (0, 5)

    out_empty = det.detect(np.empty((0, 0, 3), dtype=np.uint8))
    assert out_empty.shape == (0, 5)


def test_yolo_detector_synthetic_inference():
    """Verify detection call runs cleanly with both default and custom imgsz."""
    det_default = YOLODetector(model_name="yolo11n.pt", conf_thresh=0.35)
    dummy_frame = np.zeros((320, 320, 3), dtype=np.uint8)
    res_def = det_default.detect(dummy_frame)
    assert isinstance(res_def, np.ndarray)
    assert res_def.shape[1] == 5 or res_def.shape == (0, 5)

    det_960 = YOLODetector(model_name="yolo11n.pt", conf_thresh=0.35, imgsz=960)
    res_960 = det_960.detect(dummy_frame)
    assert isinstance(res_960, np.ndarray)
    assert res_960.shape[1] == 5 or res_960.shape == (0, 5)
