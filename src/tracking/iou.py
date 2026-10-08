"""Intersection over Union (IoU) computation for 2D bounding boxes."""

from typing import Union
import numpy as np


def compute_iou(
    bb_test: Union[np.ndarray, list],
    bb_gt: Union[np.ndarray, list],
) -> np.ndarray:
    """
    Computes Intersection over Union (IoU) between two sets of bounding boxes.
    Boxes are in format [x1, y1, x2, y2].

    Args:
        bb_test: Array or list of shape (N, 4) or (4,). Test bounding boxes.
        bb_gt: Array or list of shape (M, 4) or (4,). Ground truth or candidate bounding boxes.

    Returns:
        np.ndarray of shape (N, M) containing pairwise IoU values in range [0.0, 1.0].
        If inputs are empty, returns an empty array of shape (N, M).
    """
    boxes1 = np.asarray(bb_test, dtype=np.float64)
    boxes2 = np.asarray(bb_gt, dtype=np.float64)

    # Ensure 2D arrays (N, 4) and (M, 4)
    if boxes1.ndim == 1:
        if boxes1.size == 0:
            boxes1 = boxes1.reshape(0, 4)
        else:
            boxes1 = boxes1.reshape(1, -1)
    if boxes2.ndim == 1:
        if boxes2.size == 0:
            boxes2 = boxes2.reshape(0, 4)
        else:
            boxes2 = boxes2.reshape(1, -1)

    n = boxes1.shape[0]
    m = boxes2.shape[0]

    if n == 0 or m == 0:
        return np.zeros((n, m), dtype=np.float64)

    # Validate coordinate dimensions
    if boxes1.shape[1] < 4 or boxes2.shape[1] < 4:
        raise ValueError(f"Bounding boxes must have at least 4 coordinates [x1, y1, x2, y2], got shapes {boxes1.shape} and {boxes2.shape}")

    # Extract coordinates (only first 4 cols if extra cols like confidence exist)
    b1_x1 = boxes1[:, 0:1]  # (N, 1)
    b1_y1 = boxes1[:, 1:2]
    b1_x2 = boxes1[:, 2:3]
    b1_y2 = boxes1[:, 3:4]

    b2_x1 = boxes2[:, 0:1].T  # (1, M)
    b2_y1 = boxes2[:, 1:2].T
    b2_x2 = boxes2[:, 2:3].T
    b2_y2 = boxes2[:, 3:4].T

    # Intersection rectangle coordinates
    inter_x1 = np.maximum(b1_x1, b2_x1)  # (N, M)
    inter_y1 = np.maximum(b1_y1, b2_y1)
    inter_x2 = np.minimum(b1_x2, b2_x2)
    inter_y2 = np.minimum(b1_y2, b2_y2)

    inter_w = np.maximum(0.0, inter_x2 - inter_x1)
    inter_h = np.maximum(0.0, inter_y2 - inter_y1)
    inter_area = inter_w * inter_h

    # Areas of each bounding box
    b1_w = np.maximum(0.0, b1_x2 - b1_x1)
    b1_h = np.maximum(0.0, b1_y2 - b1_y1)
    area1 = b1_w * b1_h  # (N, 1)

    b2_w = np.maximum(0.0, b2_x2 - b2_x1)
    b2_h = np.maximum(0.0, b2_y2 - b2_y1)
    area2 = b2_w * b2_h  # (1, M)

    union_area = area1 + area2 - inter_area

    # Avoid division by zero
    iou = np.zeros((n, m), dtype=np.float64)
    valid_mask = union_area > 0.0
    iou[valid_mask] = inter_area[valid_mask] / union_area[valid_mask]

    return np.clip(iou, 0.0, 1.0)


def single_iou(box1: np.ndarray, box2: np.ndarray) -> float:
    """
    Compute IoU between two single bounding boxes [x1, y1, x2, y2].

    Args:
        box1: [x1, y1, x2, y2]
        box2: [x1, y1, x2, y2]

    Returns:
        Scalar IoU value between 0.0 and 1.0.
    """
    matrix = compute_iou(box1, box2)
    if matrix.size == 0:
        return 0.0
    return float(matrix[0, 0])
