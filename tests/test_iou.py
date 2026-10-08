"""Unit tests for Intersection over Union (IoU) calculation."""

import numpy as np
import pytest

from src.tracking.iou import compute_iou, single_iou


def test_identical_boxes():
    box1 = [10.0, 20.0, 50.0, 80.0]
    box2 = [10.0, 20.0, 50.0, 80.0]
    iou = single_iou(box1, box2)
    assert pytest.approx(iou, rel=1e-5) == 1.0


def test_no_overlap():
    box1 = [0.0, 0.0, 10.0, 10.0]
    box2 = [20.0, 20.0, 30.0, 30.0]
    iou = single_iou(box1, box2)
    assert iou == 0.0


def test_partial_overlap():
    # Box 1: [0, 0, 10, 10], area = 100
    # Box 2: [5, 0, 15, 10], area = 100
    # Intersection: [5, 0, 10, 10], area = 50
    # Union: 100 + 100 - 50 = 150
    # IoU: 50 / 150 = 1/3
    box1 = [0.0, 0.0, 10.0, 10.0]
    box2 = [5.0, 0.0, 15.0, 10.0]
    iou = single_iou(box1, box2)
    assert pytest.approx(iou, rel=1e-4) == 1.0 / 3.0


def test_edge_touching_boxes():
    # Boxes touching on vertical boundary (x=10)
    box1 = [0.0, 0.0, 10.0, 10.0]
    box2 = [10.0, 0.0, 20.0, 10.0]
    iou = single_iou(box1, box2)
    assert iou == 0.0

    # Boxes touching on horizontal boundary (y=10)
    box3 = [0.0, 0.0, 10.0, 10.0]
    box4 = [0.0, 10.0, 10.0, 20.0]
    iou2 = single_iou(box3, box4)
    assert iou2 == 0.0


def test_empty_or_inverted_boxes():
    # Box with x2 <= x1
    invalid_box = [10.0, 10.0, 5.0, 20.0]
    normal_box = [0.0, 0.0, 20.0, 20.0]
    iou = single_iou(invalid_box, normal_box)
    assert iou == 0.0


def test_matrix_iou_shapes():
    boxes_a = np.array([
        [0.0, 0.0, 10.0, 10.0],
        [10.0, 10.0, 20.0, 20.0],
        [30.0, 30.0, 40.0, 40.0],
    ])
    boxes_b = np.array([
        [0.0, 0.0, 10.0, 10.0],
        [50.0, 50.0, 60.0, 60.0],
    ])

    mat = compute_iou(boxes_a, boxes_b)
    assert mat.shape == (3, 2)
    assert pytest.approx(mat[0, 0], rel=1e-5) == 1.0
    assert mat[0, 1] == 0.0
    assert mat[1, 0] == 0.0
    assert mat[2, 0] == 0.0


def test_empty_arrays():
    empty = np.empty((0, 4))
    boxes = np.array([[0.0, 0.0, 10.0, 10.0]])
    assert compute_iou(empty, boxes).shape == (0, 1)
    assert compute_iou(boxes, empty).shape == (1, 0)
    assert compute_iou(empty, empty).shape == (0, 0)
