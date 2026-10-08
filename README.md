# Multi-Object Tracking Under Occlusion

A computer-vision project implementing a complete detection-based tracking pipeline for maintaining consistent human identities under temporary occlusions. Built from scratch with a pretrained Ultralytics YOLO detector, custom SORT tracker (Kalman Filter + IoU Matching + Hungarian Algorithm), MOT17-compatible benchmark evaluator, trajectory visualization, and failure analysis tools.

---

## 1. Project Overview

In computer vision and intelligent surveillance, tracking multiple targets across consecutive video frames is a foundational problem. Real-world environments—such as retail stores with ceiling cameras, subway stations, and public plazas—frequently exhibit crowded conditions where pedestrians cross paths, physically occlude one another, or temporarily disappear behind static obstacles.

This project delivers an end-to-end multi-object tracking (MOT) solution designed to maintain consistent identities through brief occlusions without relying on heavy external tracking frameworks.

---

## 2. Problem Statement

Standard frame-by-frame object detectors treat every video frame independently. In ceiling-mounted retail cameras, customers frequently:
1. Cross paths and occlude one another.
2. Experience partial or full visual occlusion for several frames.
3. Re-emerge after passing each other.

Without temporal state estimation and data association, a naive detector loses the target during occlusion and assigns a completely new identity upon re-emergence. This identity switching corrupts downstream analytics such as customer journey tracking, dwell-time estimation, and queue monitoring.

---

## 3. Main Objective

- Detect persons in video frames using a pretrained lightweight YOLO model (COCO person class 0).
- Maintain consistent target identities across frames using a self-implemented Simple Online and Realtime Tracking (SORT) algorithm.
- Bridge temporary occlusions using Kalman Filter state extrapolation and temporal track preservation up to `max_age` frames.
- Provide full MOT benchmark evaluation (MOTA, IDF1, ID switches, False Positives, False Negatives).
- Render motion trajectories and provide an evidence-based failure analysis tool for inspecting ID-switch causes.

---

## 4. Why Occlusion Causes Tracking Problems

During occlusion:
- **Visual Evidence Loss**: The camera sensor receives partial or zero pixel information for the occluded target.
- **Bounding Box Distortion**: When two people overlap, the detector may output a single merged bounding box or suppress one detection via Non-Maximum Suppression (NMS).
- **Geometric Ambiguity**: Because IoU between the last known position and post-occlusion position decreases as targets move, greedy spatial nearest-neighbor algorithms fail or swap identities between crossing pedestrians.

SORT resolves this by maintaining a velocity model through a Kalman Filter, allowing the tracker to coast through the occlusion gap and match the target when it reappears.

---

## 5. System Architecture

```
[ Input Frame / Video ]
           │
           ▼
[ YOLO Detector ] ── (Filter COCO class 0: Person) ──► [ Detections (x1, y1, x2, y2, score) ]
                                                                   │
                                                                   ▼
[ Active Tracks ] ──► [ Kalman Filter Predict ] ─────► [ Predicted Bounding Boxes ]
                                                                   │
                                                                   ▼
                                                       [ Pairwise IoU Matrix ]
                                                                   │
                                                                   ▼
                                                   [ Hungarian Data Association ]
                                                   (scipy linear_sum_assignment)
                                                                   │
                 ┌─────────────────────────────────────────────────┼──────────────────────────────┐
                 ▼                                                 ▼                              ▼
          [ Matched Pairs ]                              [ Unmatched Tracks ]          [ Unmatched Detections ]
                 │                                                 │                              │
                 ▼                                                 ▼                              ▼
      [ Kalman Filter Update ]                         [ Increment time_since_update ]    [ Initialize New Track ]
                 │                                                 │                              │
                 │                                        (If > max_age: Delete)                  │
                 └─────────────────────────────────────────────────┼──────────────────────────────┘
                                                                   ▼
                                                    [ Active Tracks (ID, BBox) ]
                                                                   │
                                    ┌──────────────────────────────┴──────────────────────────────┐
                                    ▼                                                             ▼
                        [ MOT-Format Output File ]                                   [ Trajectory Renderer ]
                     (frame, id, x, y, w, h, conf, ...)                            (Bounding Box, ID Badge, Path)
```

---

## 6. Mathematical Methodology

### 6.1 YOLO Detection
The detector extracts pedestrian bounding boxes $D = [x_1, y_1, x_2, y_2, s]$ using pretrained Ultralytics YOLO (`yolo11n.pt` / `yolov8n.pt`). Detections are filtered for COCO person class ID 0 with a default confidence threshold of $0.40$.

### 6.2 Kalman Filter State Estimation
Each active track maintains a 7-dimensional state vector:
$$x = [u, v, s, r, \dot{u}, \dot{v}, \dot{s}]^T$$
where:
- $(u, v)$: Center coordinates of the bounding box
- $s = w \times h$: Scale (area) of the bounding box
- $r = w / h$: Aspect ratio (assumed constant, $\dot{r} = 0$)
- $(\dot{u}, \dot{v}, \dot{s})$: Corresponding velocities

**Prediction Step:**
$$x_{k|k-1} = F x_{k-1|k-1}$$
$$P_{k|k-1} = F P_{k-1|k-1} F^T + Q$$

**Update Step (Joseph Form for Numerical Stability):**
$$y_k = z_k - H x_{k|k-1}$$
$$S_k = H P_{k|k-1} H^T + R$$
$$K_k = P_{k|k-1} H^T S_k^{-1}$$
$$x_{k|k} = x_{k|k-1} + K_k y_k$$
$$P_{k|k} = (I - K_k H) P_{k|k-1} (I - K_k H)^T + K_k R K_k^T$$

### 6.3 Intersection over Union (IoU)
The geometric overlap between predicted bounding box $B_p$ and detected bounding box $B_d$ is computed as:
$$\text{IoU}(B_p, B_d) = \frac{\text{Area}(B_p \cap B_d)}{\text{Area}(B_p \cup B_d)}$$

### 6.4 Hungarian Algorithm Assignment
The cost matrix is formed as $C = -\text{IoU}(B_p, B_d)$. The optimal global bipartite matching is solved using `scipy.optimize.linear_sum_assignment`. A match is accepted only if $\text{IoU} \ge \text{iou\_threshold}$ ($0.30$).

---

## 7. Track Lifecycle & Occlusion Handling

1. **Initialization**: Unmatched detections spawn a tentative `Track`.
2. **Confirmation**: A track is confirmed after `min_hits` detections (default: 3) or in the initial frames.
3. **Occlusion (Coast)**: When a track is missed (unmatched to any detection), it transitions to an occluded state. The Kalman Filter continues predicting position forward.
4. **ID Preservation**: If the person re-emerges within `max_age` frames (default: 30), the Hungarian matcher re-associates the new detection with the predicted state and the original track ID is preserved.
5. **Deletion**: If `time_since_update > max_age`, the track is permanently deleted.

---

## 8. Dataset: MOT17 / MOT20 Support

This project supports the MOTChallenge format (MOT17/MOT20). The actual datasets are not committed to Git to keep the repository lightweight.

### How to Prepare MOT17 Locally:
1. Download MOT17 from [motchallenge.net](https://motchallenge.net/data/MOT17/).
2. Extract the dataset into the `data/` directory so that it follows the standard structure:
```
data/
└── MOT17/
    └── train/
        ├── MOT17-02-FRCNN/
        │   ├── img1/
        │   │   ├── 000001.jpg
        │   │   └── ...
        │   ├── gt/
        │   │   └── gt.txt
        │   └── seqinfo.ini
        └── ...
```

---

## 9. Installation

### Requirements
- Python 3.9+ (Python 3.10–3.14 compatible)
- pip

### Step-by-Step Setup
```bash
# Clone the repository
git clone https://github.com/your-username/multi-object-tracking-under-occlusion.git
cd multi-object-tracking-under-occlusion

# (Optional) Create and activate a virtual environment
python -m venv .venv
# Windows:
.venv\Scripts\activate
# Linux/macOS:
# source .venv/bin/activate

# Install required dependencies
pip install -r requirements.txt
```

---

## 10. Project Structure

```
multi-object-tracking-under-occlusion/
├── README.md                          # Project documentation and guide
├── REPORT.md                          # Academic report suitable for college submission
├── requirements.txt                   # Verified Python dependencies
├── .gitignore                         # Git exclusion rules (datasets, weights, cache)
├── LICENSE                            # MIT License
│
├── configs/
│   └── config.yaml                    # System configuration parameters
│
├── data/
│   └── .gitkeep                       # Directory for local MOT datasets
│
├── models/
│   └── .gitkeep                       # Directory for downloaded YOLO model weights
│
├── src/
│   ├── __init__.py
│   ├── detection/
│   │   ├── __init__.py
│   │   └── yolo_detector.py           # Ultralytics YOLO person detector wrapper
│   ├── tracking/
│   │   ├── __init__.py
│   │   ├── kalman_filter.py           # 7-state Kalman Filter with Joseph form
│   │   ├── iou.py                     # Vectorized bounding box IoU computation
│   │   ├── track.py                   # Track representation and lifecycle state
│   │   └── sort.py                    # SORT tracker with Hungarian association
│   ├── evaluation/
│   │   ├── __init__.py
│   │   ├── mot_evaluator.py           # MOTA, IDF1, ID switches evaluation engine
│   │   └── failure_analysis.py        # ID switch evidence detection and reporting
│   ├── visualization/
│   │   ├── __init__.py
│   │   └── renderer.py                # Trajectory polyline and bounding box renderer
│   └── utils/
│       ├── __init__.py
│       ├── mot_format.py              # MOT format input/output handling
│       └── video.py                   # Unified reader for videos and image folders
│
├── scripts/
│   ├── run_tracking.py                # Run tracking pipeline on video/sequence
│   ├── evaluate.py                    # Evaluate predictions against MOT ground truth
│   ├── visualize_trajectories.py      # Render trajectories and summary plots
│   └── analyze_failures.py            # Analyze identity switches and failure causes
│
├── tests/
│   ├── test_iou.py                    # Unit tests for IoU edge cases
│   ├── test_kalman.py                 # Unit tests for Kalman prediction and update
│   ├── test_sort.py                   # Unit tests for track confirmation & occlusion
│   ├── test_mot_format.py             # Unit tests for MOT format parsing
│   ├── test_evaluation.py             # Unit tests for evaluation and metrics
│   └── test_e2e_pipeline.py           # Full end-to-end integration test
│
├── notebooks/
│   └── results_analysis.ipynb         # Interactive analysis and visualization notebook
│
└── outputs/
    ├── videos/                        # Rendered tracking output videos
    ├── tracking_results/              # MOT-format output text files
    ├── metrics/                       # JSON and CSV benchmark evaluation metrics
    ├── plots/                         # Trajectory maps and failure distribution charts
    └── failure_cases/                 # Detailed ID-switch event logs
```

---

## 11. How to Run Tracking

### Running on a Video File
```bash
python scripts/run_tracking.py --input path/to/retail_video.mp4 --output-txt outputs/tracking_results/tracking_output.txt --output-video outputs/videos/tracking_output.mp4
```

### Running on a MOT17 Image Sequence
```bash
python scripts/run_tracking.py --input data/MOT17/train/MOT17-02-FRCNN/img1 --output-txt outputs/tracking_results/mot17_02_pred.txt --output-video outputs/videos/mot17_02.mp4
```

### Optional Command-Line Arguments
- `--conf-thresh 0.35`: Override detection confidence threshold.
- `--max-age 40`: Increase occlusion tolerance window (frames).
- `--min-hits 3`: Minimum hits required before confirming a track.
- `--iou-thresh 0.30`: Matching IoU threshold.
- `--display`: Open a live display window showing tracking in real-time.

---

## 12. How to Evaluate Performance

Compare tracking output against standard MOT ground truth annotations:

```bash
python scripts/evaluate.py --gt data/MOT17/train/MOT17-02-FRCNN/gt/gt.txt --pred outputs/tracking_results/mot17_02_pred.txt --output-json outputs/metrics/metrics.json --output-csv outputs/metrics/metrics.csv
```

Outputs include:
- `mota`: Multiple Object Tracking Accuracy
- `idf1`: Identity F1-Score
- `id_switches`: Number of identity switch events
- `false_positives`: Total spurious detections
- `false_negatives`: Total missed detections

---

## 13. How to Visualize Trajectories

Render bounding boxes, ID badges, and historical centroid motion trails (~50 frames):

```bash
python scripts/visualize_trajectories.py --input data/MOT17/train/MOT17-02-FRCNN/img1 --tracking-file outputs/tracking_results/mot17_02_pred.txt --output-video outputs/videos/rendered_trajectories.mp4 --output-plot outputs/plots/trajectories_map.png
```

This generates:
1. An annotated video (`outputs/videos/rendered_trajectories.mp4`).
2. A 2D spatial overview plot (`outputs/plots/trajectories_map.png`) displaying the paths of all tracked targets.

---

## 14. How to Perform Failure Analysis

Identify every identity switch event and determine its evidence-grounded cause:

```bash
python scripts/analyze_failures.py --gt data/MOT17/train/MOT17-02-FRCNN/gt/gt.txt --pred outputs/tracking_results/mot17_02_pred.txt --output-csv outputs/failure_cases/id_switches.csv --output-plot outputs/plots/failure_distribution.png
```

Categories identified with concrete evidence:
- `occlusion`: Spatial overlap with an adjacent target or target missing for multiple frames prior to switch.
- `fast motion`: Large velocity or sudden displacement exceeding physical human limits.
- `ambiguous association`: Multiple detection candidates in close proximity competing for assignment.
- `missed detection`: Detector failed to produce a box for the target in preceding frames.
- `unknown`: No clear physical or geometric indicator detected.

---

## 15. Metrics Explanation

| Metric | Full Name | Description | Desired Direction |
| :--- | :--- | :--- | :--- |
| **MOTA** | Multi-Object Tracking Accuracy | Combines False Positives, False Negatives, and ID Switches: $1 - \frac{\text{FN} + \text{FP} + \text{IDSW}}{\text{GT}}$ | Higher $\uparrow$ |
| **IDF1** | Identification F1-Score | Ratio of correctly identified detections over average ground truth and tracker detections | Higher $\uparrow$ |
| **IDSW** | Identity Switches | Number of times a tracked object is assigned a different identity | Lower $\downarrow$ |
| **FP** | False Positives | Tracker output where no ground-truth pedestrian exists | Lower $\downarrow$ |
| **FN** | False Negatives | Ground-truth pedestrians missed by the tracker | Lower $\downarrow$ |
| **MT** | Mostly Tracked | Trajectories tracked for $\ge 80\%$ of their lifespan | Higher $\uparrow$ |
| **ML** | Mostly Lost | Trajectories tracked for $\le 20\%$ of their lifespan | Lower $\downarrow$ |

---

## 16. Results & Benchmark Evaluation

> [!NOTE]
> When the MOT17 dataset is downloaded locally by the user, benchmark evaluation can be executed using `scripts/evaluate.py`. Per strict scientific integrity guidelines, metrics must never be fabricated.

### Automated Test Suite Results
All unit tests and end-to-end integration tests execute successfully:
- **Pytest**: 21 passed (100% pass rate)
  - `test_iou.py`: Normal overlap, zero overlap, identical boxes, edge-touching boundaries, matrix operations.
  - `test_kalman.py`: Bbox conversion, velocity estimation, extrapolation, Joseph-form numerical stability.
  - `test_sort.py`: Track creation, confirmation, multi-target tracking, occlusion recovery, max_age deletion.
  - `test_mot_format.py`: Coordinate conversions, roundtrip I/O.
  - `test_evaluation.py`: Synthetic benchmark evaluation and failure analysis.
  - `test_e2e_pipeline.py`: Full end-to-end video tracking, evaluation, rendering, and failure analysis.
- **Python compileall**: All source files compiled cleanly with 0 syntax or runtime errors.

---

## 17. Limitations & Future Work

### Limitations
1. **Linear Velocity Assumption**: The Kalman Filter employs a constant-velocity kinematic model. Sudden turns or erratic changes in speed can cause association failures.
2. **Appearance Neglect**: Classic SORT relies strictly on spatial geometry (IoU). If two visually distinct targets cross closely and their bounding boxes overlap heavily, an ID switch may occur.
3. **Long-Term Occlusions**: If an occlusion lasts longer than `max_age` (e.g., >30 frames), the track is deleted and a new ID will be assigned upon reappearance.

### Future Improvements
1. **DeepSORT Integration**: Introduce an appearance embedding model (Re-ID CNN) to compute cosine appearance distance in addition to spatial IoU.
2. **ByteTrack Association**: Implement low-confidence detection matching to recover heavily occluded pedestrians whose detection scores temporarily dip.
3. **Adaptive Camera Calibration**: Incorporate homography estimation to account for ceiling camera perspective distortion.

---

## 18. Reproducibility

To verify the installation and run all tests:
```bash
python -m pytest -v
python -m compileall src scripts tests
```

---

## 19. License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
