"""Utilities for MOT (Multiple Object Tracking) format input/output handling."""

import os
from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd


def xyxy_to_xywh(bbox: Union[np.ndarray, list]) -> np.ndarray:
    """
    Convert bounding box from [x1, y1, x2, y2] to [x, y, w, h] (top-left x, top-left y, width, height).

    Args:
        bbox: Array of shape (..., 4) or 1D list/array.

    Returns:
        np.ndarray with same prefix shape and [x, y, w, h].
    """
    bbox = np.asarray(bbox, dtype=np.float64)
    out = bbox.copy()
    out[..., 2] = bbox[..., 2] - bbox[..., 0]  # w = x2 - x1
    out[..., 3] = bbox[..., 3] - bbox[..., 1]  # h = y2 - y1
    return out


def xywh_to_xyxy(bbox: Union[np.ndarray, list]) -> np.ndarray:
    """
    Convert bounding box from [x, y, w, h] to [x1, y1, x2, y2].

    Args:
        bbox: Array of shape (..., 4) or 1D list/array.

    Returns:
        np.ndarray with same prefix shape and [x1, y1, x2, y2].
    """
    bbox = np.asarray(bbox, dtype=np.float64)
    out = bbox.copy()
    out[..., 2] = bbox[..., 0] + bbox[..., 2]  # x2 = x + w
    out[..., 3] = bbox[..., 1] + bbox[..., 3]  # y2 = y + h
    return out


def save_mot_results(
    filepath: str,
    results: List[Tuple[int, int, float, float, float, float, float]],
) -> None:
    """
    Write tracking results to a MOT-format text/csv file.
    Format: frame,id,x,y,w,h,confidence,-1,-1,-1

    Args:
        filepath: Destination file path.
        results: List of tuples (frame, id, x, y, w, h, confidence).
    """
    os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
    with open(filepath, "w", encoding="utf-8") as f:
        for item in results:
            frame, track_id, x, y, w, h, conf = item
            f.write(f"{int(frame)},{int(track_id)},{x:.2f},{y:.2f},{w:.2f},{h:.2f},{conf:.4f},-1,-1,-1\n")


def load_mot_results(filepath: str) -> pd.DataFrame:
    """
    Load tracking results or MOT ground truth file into a pandas DataFrame.

    Expected standard columns:
    frame, id, x, y, w, h, conf, (optional 3 extra cols or class/vis cols).

    Args:
        filepath: Path to MOT file (.txt).

    Returns:
        DataFrame with standardized columns:
        ['frame', 'id', 'x', 'y', 'w', 'h', 'conf']
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"MOT file not found: {filepath}")

    # Inspect first non-empty line to check delimiter (comma or whitespace)
    with open(filepath, "r", encoding="utf-8") as f:
        first_line = ""
        for line in f:
            if line.strip():
                first_line = line.strip()
                break

    sep = "," if "," in first_line else r"\s+"

    df = pd.read_csv(filepath, sep=sep, header=None, engine="python")
    col_names = ["frame", "id", "x", "y", "w", "h", "conf", "x3", "y3", "z3"]
    df.columns = col_names[: len(df.columns)]

    # Cast integer columns
    df["frame"] = df["frame"].astype(int)
    df["id"] = df["id"].astype(int)
    df["x"] = df["x"].astype(float)
    df["y"] = df["y"].astype(float)
    df["w"] = df["w"].astype(float)
    df["h"] = df["h"].astype(float)

    return df
