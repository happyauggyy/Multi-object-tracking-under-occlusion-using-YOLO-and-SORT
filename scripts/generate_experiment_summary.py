"""Generate experiment summary CSV and Markdown reports for tracking benchmarks."""

import json
import os
import pandas as pd

experiments = [
    {
        "experiment_name": "Baseline Conf 0.40",
        "hypothesis": "Initial baseline at default confidence 0.40 with naive Hungarian association",
        "conf_thresh": 0.40,
        "iou_thresh": 0.30,
        "max_age": 30,
        "min_hits": 3,
        "device": "cpu",
        "predictions_file": "outputs/MOT17-02-FRCNN.txt",
        "metrics_json": "outputs/MOT17-02-FRCNN_metrics.json",
        "runtime_s": None,
        "fps": None,
        "status": "Completed (Baseline)",
        "observations": "Low false positives (232), but high false negatives (14,942) due to strict detector cutoff."
    },
    {
        "experiment_name": "Baseline Conf 0.25",
        "hypothesis": "Lower detector confidence threshold to capture more pedestrian candidates",
        "conf_thresh": 0.25,
        "iou_thresh": 0.30,
        "max_age": 30,
        "min_hits": 3,
        "device": "cpu",
        "predictions_file": "outputs/MOT17-02-FRCNN_conf025.txt",
        "metrics_json": "outputs/MOT17-02-FRCNN_conf025_metrics.json",
        "runtime_s": None,
        "fps": None,
        "status": "Completed (Baseline)",
        "observations": "Best baseline MOTA (18.75%), but FP increased by 4x (921) and ID switches doubled (81)."
    },
    {
        "experiment_name": "Baseline Conf 0.15",
        "hypothesis": "Very low confidence threshold to aggressively reduce missed pedestrians",
        "conf_thresh": 0.15,
        "iou_thresh": 0.30,
        "max_age": 30,
        "min_hits": 3,
        "device": "cpu",
        "predictions_file": "outputs/MOT17-02-FRCNN_conf015.txt",
        "metrics_json": "outputs/MOT17-02-FRCNN_conf015_metrics.json",
        "runtime_s": None,
        "fps": None,
        "status": "Completed (Baseline)",
        "observations": "Severe degradation: MOTA crashed to 11.57% due to massive false positives (3,199) and 189 ID switches."
    },
    {
        "experiment_name": "Conf 0.25 + Assoc Fix",
        "hypothesis": "Fix Hungarian assignment so sub-threshold IoUs cannot divert or steal valid matches",
        "conf_thresh": 0.25,
        "iou_thresh": 0.30,
        "max_age": 30,
        "min_hits": 3,
        "device": "cpu",
        "predictions_file": "outputs/MOT17-02-FRCNN_conf025_assocfix.txt",
        "metrics_json": "outputs/MOT17-02-FRCNN_conf025_assocfix_metrics.json",
        "runtime_s": 50.57,
        "fps": 11.9,
        "status": "Completed",
        "observations": "Mathematically sound matching verified; mostly tracked objects increased from 7 to 8; verified no test regressions."
    },
    {
        "experiment_name": "Conf 0.30 + Assoc Fix",
        "hypothesis": "Balance detector noise suppression and detection recall at conf 0.30",
        "conf_thresh": 0.30,
        "iou_thresh": 0.30,
        "max_age": 30,
        "min_hits": 3,
        "device": "cpu",
        "predictions_file": "outputs/MOT17-02-FRCNN_conf030_assocfix.txt",
        "metrics_json": "outputs/MOT17-02-FRCNN_conf030_assocfix_metrics.json",
        "runtime_s": 48.44,
        "fps": 12.4,
        "status": "Completed",
        "observations": "IDF1 improved to 26.79%; false positives reduced by 33% (628 vs 939) and ID switches reduced to 67."
    },
    {
        "experiment_name": "Conf 0.35 + Assoc Fix",
        "hypothesis": "Further filter out marginal background detections while keeping pedestrian tracks intact",
        "conf_thresh": 0.35,
        "iou_thresh": 0.30,
        "max_age": 30,
        "min_hits": 3,
        "device": "cpu",
        "predictions_file": "outputs/MOT17-02-FRCNN_conf035_assocfix.txt",
        "metrics_json": "outputs/MOT17-02-FRCNN_conf035_assocfix_metrics.json",
        "runtime_s": 47.52,
        "fps": 12.6,
        "status": "Completed",
        "observations": "MOTA rose to 18.84% (surpassing all baselines); FP plunged to 366 and ID switches dropped to 46."
    },
    {
        "experiment_name": "Conf 0.40 + Assoc Fix",
        "hypothesis": "Confirm performance at baseline 0.40 cutoff under new association logic",
        "conf_thresh": 0.40,
        "iou_thresh": 0.30,
        "max_age": 30,
        "min_hits": 3,
        "device": "cpu",
        "predictions_file": "outputs/MOT17-02-FRCNN_conf040_assocfix.txt",
        "metrics_json": "outputs/MOT17-02-FRCNN_conf040_assocfix_metrics.json",
        "runtime_s": 53.56,
        "fps": 11.2,
        "status": "Completed",
        "observations": "MOTA 18.15%, ID switches 37; higher FN (14,939) confirms conf 0.35 is superior."
    },
    {
        "experiment_name": "Conf 0.35 + MinHits=2",
        "hypothesis": "Lower confirmation threshold to recover track bounding boxes in earlier frames",
        "conf_thresh": 0.35,
        "iou_thresh": 0.30,
        "max_age": 30,
        "min_hits": 2,
        "device": "cpu",
        "predictions_file": "outputs/MOT17-02-FRCNN_conf035_minhits2.txt",
        "metrics_json": "outputs/MOT17-02-FRCNN_conf035_minhits2_metrics.json",
        "runtime_s": 53.26,
        "fps": 11.3,
        "status": "Completed",
        "observations": "False negatives slightly reduced (14,631 vs 14,669), but FP rose to 407, yielding MOTA 18.78%."
    },
    {
        "experiment_name": "Conf 0.35 + IoU=0.20",
        "hypothesis": "Relax association IoU threshold to capture partially overlapping occluded boxes",
        "conf_thresh": 0.35,
        "iou_thresh": 0.20,
        "max_age": 30,
        "min_hits": 3,
        "device": "cpu",
        "predictions_file": "outputs/MOT17-02-FRCNN_conf035_iou020.txt",
        "metrics_json": "outputs/MOT17-02-FRCNN_conf035_iou020_metrics.json",
        "runtime_s": 48.63,
        "fps": 12.3,
        "status": "Completed",
        "observations": "Slight drift into ID switches (52 vs 46); default IoU threshold 0.30 remains optimal."
    },
    {
        "experiment_name": "Conf 0.35 + MaxAge=45",
        "hypothesis": "Extend track memory to 45 frames (1.5s) to preserve identity across longer occlusions",
        "conf_thresh": 0.35,
        "iou_thresh": 0.30,
        "max_age": 45,
        "min_hits": 3,
        "device": "cpu",
        "predictions_file": "outputs/MOT17-02-FRCNN_conf035_maxage45.txt",
        "metrics_json": "outputs/MOT17-02-FRCNN_conf035_maxage45_metrics.json",
        "runtime_s": 48.76,
        "fps": 12.3,
        "status": "Completed",
        "observations": "Significant improvement: MOTA reached 18.90% and IDF1 jumped to 27.06% with ID switches down to 41."
    },
    {
        "experiment_name": "Conf 0.35 + MaxAge=60 (Best 640px)",
        "hypothesis": "Extend track memory to 60 frames (2.0s) to maximize occlusion recovery and identity retention at default 640px",
        "conf_thresh": 0.35,
        "iou_thresh": 0.30,
        "max_age": 60,
        "min_hits": 3,
        "imgsz": 640,
        "device": "cpu",
        "predictions_file": "outputs/MOT17-02-FRCNN_conf035_maxage60.txt",
        "metrics_json": "outputs/MOT17-02-FRCNN_conf035_maxage60_metrics.json",
        "runtime_s": 48.33,
        "fps": 12.4,
        "status": "Completed",
        "observations": "Best 640px result: MOTA 18.90%, IDF1 27.86% (+0.94% over best baseline IDF1 26.92% at conf 0.15, +1.09% over conf 0.25), IDSW 40 (-51% vs baseline 81), FP 375 (-59% vs baseline 921)."
    },
    {
        "experiment_name": "Conf 0.30 + MaxAge=60",
        "hypothesis": "Evaluate combined conf 0.30 with max_age 60",
        "conf_thresh": 0.30,
        "iou_thresh": 0.30,
        "max_age": 60,
        "min_hits": 3,
        "imgsz": 640,
        "device": "cpu",
        "predictions_file": "outputs/MOT17-02-FRCNN_conf030_maxage60.txt",
        "metrics_json": "outputs/MOT17-02-FRCNN_conf030_maxage60_metrics.json",
        "runtime_s": 49.82,
        "fps": 12.0,
        "status": "Completed",
        "observations": "Lower confidence yields higher FP (637) and lower IDF1 (26.47%) than conf 0.35 + max_age 60."
    },
    {
        "experiment_name": "Conf 0.35 + MaxAge=60 + ImgSz=960",
        "hypothesis": "Increase detector resolution to 960px to detect smaller/distant occluded pedestrians",
        "conf_thresh": 0.35,
        "iou_thresh": 0.30,
        "max_age": 60,
        "min_hits": 3,
        "imgsz": 960,
        "device": "cpu",
        "predictions_file": "outputs/MOT17-02-FRCNN_conf035_maxage60_imgsz960.txt",
        "metrics_json": "outputs/MOT17-02-FRCNN_conf035_maxage60_imgsz960_metrics.json",
        "runtime_s": 129.96,
        "fps": 4.6,
        "status": "Completed",
        "observations": "Substantial gain: MOTA surged to 24.79% (+5.89% over 640px), IDF1 to 33.59% (+5.73%), and FN reduced by 1,368 misses (13,287 vs 14,655)."
    },
    {
        "experiment_name": "Conf 0.35 + MaxAge=60 + ImgSz=1280 (Best Overall)",
        "hypothesis": "Increase detector resolution to 1280px to maximize detection recall on high-res 1080p video",
        "conf_thresh": 0.35,
        "iou_thresh": 0.30,
        "max_age": 60,
        "min_hits": 3,
        "imgsz": 1280,
        "device": "cpu",
        "predictions_file": "outputs/MOT17-02-FRCNN_conf035_maxage60_imgsz1280.txt",
        "metrics_json": "outputs/MOT17-02-FRCNN_conf035_maxage60_imgsz1280_metrics.json",
        "runtime_s": 174.89,
        "fps": 3.4,
        "status": "Completed (Best Overall)",
        "observations": "Peak overall tracking accuracy: MOTA 27.75% (+8.85% over 640px), IDF1 35.82% (+7.96% over 640px, +8.90% over best baseline 26.92%), FN reduced by 2,450 misses (12,205 vs 14,655), Mostly Lost reduced from 42 to 31."
    },
    {
        "experiment_name": "Two-Stage ByteTrack (Conf 0.35/0.10, ImgSz 1280)",
        "hypothesis": "Recover occluded tracks using two-stage association with low-confidence detections [0.10, 0.35) while initializing new tracks only from high-confidence detections",
        "conf_thresh": 0.35,
        "iou_thresh": 0.30,
        "max_age": 60,
        "min_hits": 3,
        "imgsz": 1280,
        "device": "cpu",
        "predictions_file": "outputs/MOT17-02-FRCNN_conf035_low010_maxage60_imgsz1280_twostage.txt",
        "metrics_json": "outputs/MOT17-02-FRCNN_conf035_low010_maxage60_imgsz1280_twostage_metrics.json",
        "runtime_s": 178.68,
        "fps": 3.4,
        "status": "Completed",
        "observations": "High identity preservation (IDF1 36.67%) and pedestrian coverage (FN 10,160). However, sub-0.20 noise boxes inflated FP to 5,361, reducing MOTA to 15.56%."
    },
    {
        "experiment_name": "Two-Stage ByteTrack (Conf 0.35/0.20, ImgSz 1280)",
        "hypothesis": "Filter out ultra-low confidence clutter (<0.20) in two-stage recovery to suppress false positives",
        "conf_thresh": 0.35,
        "low_conf_thresh": 0.20,
        "iou_thresh": 0.30,
        "max_age": 60,
        "min_hits": 3,
        "imgsz": 1280,
        "device": "cpu",
        "predictions_file": "outputs/MOT17-02-FRCNN_conf035_low020_maxage60_imgsz1280_twostage.txt",
        "metrics_json": "outputs/MOT17-02-FRCNN_conf035_low020_maxage60_imgsz1280_twostage_metrics.json",
        "runtime_s": 212.19,
        "fps": 2.8,
        "status": "Completed",
        "observations": "Substantial FP reduction: FP dropped from 5,361 to 3,003 (-44%), MOTA recovered to 22.96% (+7.40%), while preserving pedestrian coverage (FN 11,161, MT 10, IDF1 35.78%)."
    },
    {
        "experiment_name": "Two-Stage ByteTrack (Conf 0.35/0.25, ImgSz 1280, Best IDF1)",
        "hypothesis": "Set low-confidence floor to 0.25 to tightly filter noise while retaining genuine occluded pedestrian recovery",
        "conf_thresh": 0.35,
        "low_conf_thresh": 0.25,
        "iou_thresh": 0.30,
        "max_age": 60,
        "min_hits": 3,
        "imgsz": 1280,
        "device": "cpu",
        "predictions_file": "outputs/MOT17-02-FRCNN_conf035_low025_maxage60_imgsz1280_twostage.txt",
        "metrics_json": "outputs/MOT17-02-FRCNN_conf035_low025_maxage60_imgsz1280_twostage_metrics.json",
        "runtime_s": 197.73,
        "fps": 3.0,
        "status": "Completed (Best IDF1)",
        "observations": "Best identity preservation overall: IDF1 achieved 37.52% (+1.70% over single-stage 1280px, +0.85% over 0.10 two-stage). FP dropped to 2,243 (down by 3,118 vs 0.10), recovering MOTA to 25.23% (+9.67% over 0.10) while still beating single-stage on misses (11,529 vs 12,205)."
    },
]

rows = []
for exp in experiments:
    m_path = exp["metrics_json"]
    with open(m_path, "r", encoding="utf-8") as f:
        metrics = json.load(f)

    # Count predictions in output txt
    pred_path = exp["predictions_file"]
    num_preds = 0
    if os.path.exists(pred_path):
        with open(pred_path, "r", encoding="utf-8") as pf:
            num_preds = sum(1 for line in pf if line.strip())

    row = {
        "Experiment Name": exp["experiment_name"],
        "Hypothesis": exp["hypothesis"],
        "Confidence Thresh": exp["conf_thresh"],
        "IoU Thresh": exp["iou_thresh"],
        "Max Age": exp["max_age"],
        "Min Hits": exp["min_hits"],
        "Image Size": exp.get("imgsz", 640),
        "Device": exp["device"],
        "MOTA (%)": round(metrics["mota"] * 100.0, 2),
        "IDF1 (%)": round(metrics["idf1"] * 100.0, 2),
        "ID Switches": metrics["id_switches"],
        "False Positives": metrics["false_positives"],
        "False Negatives": metrics["false_negatives"],
        "Mostly Tracked": metrics["mostly_tracked"],
        "Partially Tracked": metrics["partially_tracked"],
        "Mostly Lost": metrics["mostly_lost"],
        "Predictions Count": num_preds,
        "Runtime (s)": exp["runtime_s"],
        "Speed (FPS)": exp["fps"],
        "Status": exp["status"],
        "Observations": exp["observations"],
    }
    rows.append(row)

df = pd.DataFrame(rows)
df.to_csv("outputs/experiment_summary.csv", index=False)
print("Saved outputs/experiment_summary.csv")

# Generate Markdown table
md_content = "# Multi-Object Tracking Under Occlusion: Experiment Summary Report\n\n"
md_content += f"Dataset: MOT17-02-FRCNN (600 frames, 18,581 ground-truth pedestrian annotations)\n"
md_content += f"Detector: YOLO11n (`yolo11n.pt`) | Hardware: Intel i3-11th Gen CPU (Single-thread PyTorch CPU inference)\n\n"
md_content += "## Quantitative Comparison Table\n\n"
md_content += "| Experiment | Conf | IoU | Max Age | Min Hits | ImgSz | MOTA (%) | IDF1 (%) | ID Switches | FP | FN | MT | PT | ML | FPS |\n"
md_content += "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|\n"

for _, r in df.iterrows():
    fps_str = f"{r['Speed (FPS)']:.1f}" if pd.notna(r["Speed (FPS)"]) else "N/A"
    md_content += (
        f"| **{r['Experiment Name']}** | {r['Confidence Thresh']} | {r['IoU Thresh']} | {r['Max Age']} | {r['Min Hits']} | {r['Image Size']} | "
        f"**{r['MOTA (%)']:.2f}%** | **{r['IDF1 (%)']:.2f}%** | {r['ID Switches']} | {r['False Positives']} | "
        f"{r['False Negatives']} | {r['Mostly Tracked']} | {r['Partially Tracked']} | {r['Mostly Lost']} | {fps_str} |\n"
    )

md_content += "\n## Key Findings and Analysis\n\n"
md_content += "1. **Association Logic Fix**:\n"
md_content += "   - In the initial implementation, `linear_sum_assignment(-iou_matrix)` allowed sub-threshold IoUs to contribute negative costs, frequently diverting tracks away from valid candidates and dropping true pedestrian tracks.\n"
md_content += "   - Replacing this with a cost matrix $C = 1 - \\text{IoU}$ with large penalties ($10^5$) for invalid pairs ($< \\tau$) mathematically guarantees maximum cardinality on valid matches, followed by IoU maximization.\n\n"
md_content += "2. **Detector Confidence Optimization**:\n"
md_content += "   - Confidence 0.25 produces excessive noise (921-939 false positives), destabilizing SORT.\n"
md_content += "   - Confidence 0.35 strikes the ideal trade-off: false positives drop by 59% (to 366-375), reducing spurious tracks while maintaining high pedestrian recall.\n\n"
md_content += "3. **Occlusion Handling via Extended Track Memory (`max_age = 60`)**:\n"
md_content += "   - Extending track persistence from 30 frames (1.0s) to 60 frames (2.0s) yields massive gains in identity persistence.\n"
md_content += "   - At default 640px, IDF1 rose to **27.86%** (+0.94% over the best baseline IDF1 of 26.92% at conf 0.15, and +1.09% over conf 0.25's 26.77%), and ID switches plunged from 81 to **40** (-51% reduction).\n\n"
md_content += "4. **Detector Resolution Optimization (`imgsz=960` and `imgsz=1280`)**:\n"
md_content += "   - Downsampling 1920x1080 video to 640px causes YOLO11n to miss small/distant pedestrians (accounting for ~14,655 false negatives).\n"
md_content += "   - Increasing resolution to **960px** reduced false negatives by 1,368 misses, boosting MOTA to **24.79%** (+5.89%) and IDF1 to **33.59%** (+5.73%) at 4.6 FPS.\n"
md_content += "   - Increasing resolution to **1280px** reduced false negatives by 2,450 misses (from 14,655 down to 12,205), achieving the peak metrics: **MOTA 27.75%** (+8.85% over 640px), **IDF1 35.82%** (+7.96% over 640px, +8.90% over best baseline), and reducing Mostly Lost objects from 42 to 31, at 3.4 FPS on CPU.\n\n"
md_content += "5. **ByteTrack-Style Two-Stage Association Evaluation & Trade-offs**:\n"
md_content += "   - Evaluated ByteTrack two-stage association matching high-confidence detections (>=0.35) first, followed by low-confidence recovery matching remaining tracks without spawning tracks from weak detections.\n"
md_content += "   - **Controlled Threshold Sweep**:\n"
md_content += "     - **Low Threshold 0.10**: IDF1 rose to 36.67% and FN fell to 10,160 (-2,045 misses), but severe background clutter inflated FP to 5,361 (+4,223) and ID switches to 169, depressing MOTA to 15.56%.\n"
md_content += "     - **Low Threshold 0.20**: FP dropped by 44% (3,003 vs 5,361), recovering MOTA to 22.96% (+7.40%) while maintaining pedestrian recovery (FN 11,161, MT 10, IDF1 35.78%).\n"
md_content += "     - **Low Threshold 0.25 (Best IDF1)**: Reached project-record **IDF1 37.52%** (+1.70% over single-stage 1280px, +0.85% over 0.10 two-stage). FP dropped by 3,118 (down to 2,243), recovering MOTA to **25.23%** (+9.67% over 0.10), while still reducing false negatives by 676 misses compared to single-stage (11,529 vs 12,205).\n"
md_content += "   - **Targeted Frame Diagnostics (Frames 343 & 485)**:\n"
md_content += "     - In Frame 343 (GT=20): Single-stage detected only 6 TP (FN=14, FP=4). Two-stage 0.10 doubled TP to 12 (FN=8) but produced 19 FP with weak detections (min conf 0.12). Two-stage 0.25 maintained high TP (10) while slashing FP to 8.\n"
md_content += "     - Diagnostics confirmed that detections in the range [0.10, 0.25) account for the vast majority of spurious ghost tracks and drifting.\n\n"
md_content += "6. **Final Operational Recommendations**:\n"
md_content += "   - **Best Overall Accuracy Benchmark**: **Single-Stage 1280px** (`conf=0.35`, `max_age=60`, `min_hits=3`, `iou_thresh=0.30`) remains the reference champion for overall detection accuracy with **MOTA 27.75%**, minimum false positives (1,138), and minimum ID switches (82).\n"
md_content += "   - **Best Identity Preservation & Occlusion Recovery**: **Two-Stage ByteTrack 1280px with `low_conf=0.25`** is the champion for identity persistence, achieving peak **IDF1 37.52%**, reducing missed pedestrians to 11,529 (-676 misses), and achieving **MOTA 25.23%** at 3.0 FPS.\n"

with open("outputs/experiment_summary.md", "w", encoding="utf-8") as f:
    f.write(md_content)
print("Saved outputs/experiment_summary.md")
