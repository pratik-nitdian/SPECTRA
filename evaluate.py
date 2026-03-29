import os
import time
import json
import glob
import numpy as np
from collections import defaultdict
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor
import cv2

from src import preprocessing
from src.document_engine import DocumentEngine
from src.utils import visualize_layout, compute_iou, process_detections

# -------------------------------
# mAP & AP COMPUTATION
# -------------------------------
def compute_ap(tp, fp, total_gt):
    if total_gt == 0:
        return 0

    scores = np.ones_like(tp) # Fake scores, as we sort by confidence before
    
    # Sort by confidence (descending). Predictions are already sorted.
    idx = np.argsort(-scores)
    tp = np.array(tp)[idx]
    fp = np.array(fp)[idx]

    tp_cum = np.cumsum(tp)
    fp_cum = np.cumsum(fp)

    recall = tp_cum / total_gt if total_gt > 0 else 0
    precision = tp_cum / (tp_cum + fp_cum + 1e-9)

    # VOC-style 11-point AP
    ap = 0
    for t in np.linspace(0, 1, 11):
        if np.any(recall >= t):
            ap += np.max(precision[recall >= t])
    return ap / 11

def run_evaluation(gt_dir, pred_dir, image_dir, vis_dir, save_processed_vis, postprocess_iou_thresh=None):
    """Loads GT and Pred JSONs, computes mAP, and saves processed visualizations."""
    gt_files = glob.glob(os.path.join(gt_dir, "*.json"))
    pred_files = glob.glob(os.path.join(pred_dir, "*.json"))

    gt_dict = {}
    for f in gt_files:
        data = json.load(open(f))
        gt_dict[data["file_name"]] = data["annotations"]

    pred_dict = {}
    for f in tqdm(pred_files, desc="Processing and Visualizing"):
        data = json.load(open(f))
        predictions = data["annotations"]

        # Re-assign category_id -1 to 0 (background class)
        for p in predictions:
            if p["category_id"] == -1:
                p["category_id"] = 0

        if postprocess_iou_thresh is not None:
            processed_predictions = process_detections(predictions, iou_thresh=postprocess_iou_thresh)
        else:
            processed_predictions = process_detections(predictions)
            
        pred_dict[data["file_name"]] = processed_predictions

        if save_processed_vis:
            image_path = os.path.join(image_dir, data["file_name"])
            vis_path = os.path.join(vis_dir, os.path.splitext(data["file_name"])[0] + ".jpg")
            visualize_layout(image_path, processed_predictions, vis_path)


    class_records = defaultdict(lambda: {
        "tp": [], "fp": [], "scores": [], "total_gt": 0
    })

    for img, gt_ann in gt_dict.items():
        preds = pred_dict.get(img, [])

        for g in gt_ann:
            class_records[g["category_id"]]["total_gt"] += 1

        matched_gt = set()
        preds_sorted = sorted(preds, key=lambda x: -x["confidence"])

        for p in preds_sorted:
            best_iou = 0
            best_gt_idx = -1

            for i, g in enumerate(gt_ann):
                if i in matched_gt:
                    continue
                iou = compute_iou(p["bbox"], g["bbox"])
                if iou > best_iou:
                    best_iou = iou
                    best_gt_idx = i

            cls = p["category_id"]
            class_records[cls]["scores"].append(p["confidence"])

            if best_iou >= 0.5:
                # Correct match (TP)
                if best_gt_idx not in matched_gt:
                    class_records[cls]["tp"].append(1)
                    class_records[cls]["fp"].append(0)
                    matched_gt.add(best_gt_idx)
                else:
                    # Duplicate detection (FP)
                    class_records[cls]["tp"].append(0)
                    class_records[cls]["fp"].append(1)
            else:
                # No match (FP)
                class_records[cls]["tp"].append(0)
                class_records[cls]["fp"].append(1)

    aps = {}
    for cls, rec in class_records.items():
        # Sort by score before AP calculation
        sorted_indices = np.argsort(-np.array(rec['scores']))
        tp_sorted = np.array(rec['tp'])[sorted_indices]
        fp_sorted = np.array(rec['fp'])[sorted_indices]
        aps[cls] = compute_ap(tp_sorted, fp_sorted, rec["total_gt"])

    mAP = np.mean(list(aps.values())) if aps else 0
    return mAP, aps


def process_wrapper(args):
    """Wrapper for multiprocessing preprocessing."""
    return preprocessing.process_and_save(*args)

def main():
    # ------------------------------
    # CONFIGURATION
    # ------------------------------
    BATCH_SIZE = 32
    MODEL_PATH = "doclayout_yolo_doclaynet_imgsz1120_docsynth_pretrain.pt"
    
    # --- DIRS (Examples) ---
    GT_DIR = "./benchmark/jsons"
    INPUT_DIR = "./benchmark/images"
    BASE_OUTPUT_DIR = "./evaluation_results"
    
    PARAM_SETS = [
        {'deskew': 1, 'denoise': 0, 'clahe': 0, 'sharpen': 0, 'imgsz': 1120, 'conf': 0.1, 'postprocess_iou_thresh': 0.3, 'save_vis' : 1},
        {'deskew': 1, 'denoise': 0, 'clahe': 0, 'sharpen': 0, 'imgsz': 1120, 'conf': 0.40, 'postprocess_iou_thresh': 0.6, 'save_vis' : 1},
    ]

    best_map = -1
    best_params = None
    results_summary = {}

    print(f"🚀 Loading Model...")
    engine = DocumentEngine(MODEL_PATH)
    
    if not os.path.exists(INPUT_DIR):
        print(f"❌ Input directory '{INPUT_DIR}' not found. Please update GT_DIR and INPUT_DIR in evaluate.py.")
        return

    image_files = [f for f in os.listdir(INPUT_DIR) if f.lower().endswith((".jpg", ".png", ".jpeg"))]

    for params in PARAM_SETS:
        run_name = "_".join(f"{k}{v}" for k, v in sorted(params.items()) if k != 'save_vis')
        print(f"\n{'='*50}\n🔬 TESTING PARAMS: {run_name}\n{'='*50}")

        PREPROCESSED_DIR = os.path.join(BASE_OUTPUT_DIR, run_name, "preprocessed")
        OUTPUT_JSON_DIR = os.path.join(BASE_OUTPUT_DIR, run_name, "json")
        VIS_DIR = os.path.join(BASE_OUTPUT_DIR, run_name, "visualizations")
        PROCESSED_VIS_DIR = os.path.join(BASE_OUTPUT_DIR, run_name, "visualizations_processed")
        os.makedirs(PREPROCESSED_DIR, exist_ok=True)
        os.makedirs(OUTPUT_JSON_DIR, exist_ok=True)
        os.makedirs(VIS_DIR, exist_ok=True)
        os.makedirs(PROCESSED_VIS_DIR, exist_ok=True)

        tasks = []
        for filename in image_files:
            base_name = os.path.splitext(filename)[0]
            raw_path = os.path.join(INPUT_DIR, filename)
            npy_path = os.path.join(PREPROCESSED_DIR, f"{base_name}.npy")
            tasks.append((raw_path, npy_path, params))

        num_cores = os.cpu_count()
        if not all(os.path.exists(t[1]) for t in tasks):
            print(f"⏳ Preprocessing with {num_cores} cores...")
            with ProcessPoolExecutor(max_workers=num_cores) as executor:
                list(tqdm(executor.map(process_wrapper, tasks), total=len(tasks), desc="Preprocessing"))
        else:
            print("⏩ Skipping Preprocessing (Files exist)")

        inference_queue = [(t[1], os.path.basename(t[0])) for t in tasks if os.path.exists(t[1])]
        batches = [inference_queue[i:i + BATCH_SIZE] for i in range(0, len(inference_queue), BATCH_SIZE)]
        
        print(f"⚡ Running Inference on {len(batches)} batches...")
        start_time = time.time()
        
        imgsz = params.get('imgsz', 1120)
        conf = params.get('conf', 0.25)
        save_raw_vis = params.get('save_vis', 0) == 1 

        for batch in tqdm(batches, desc="Inference"):
            engine.run_batch(batch, OUTPUT_JSON_DIR, VIS_DIR, save_vis=save_raw_vis) # Note: run_batch internally uses self.imgsz and self.conf
        
        total_time = time.time() - start_time
        fps = len(inference_queue) / total_time if total_time > 0 else 0
        print(f"⚡ Inference FPS: {fps:.2f}")

        print(f"📊 Evaluating results...")
        postprocess_thresh = params.get('postprocess_iou_thresh')
        save_processed_vis_flag = params.get('save_vis', 0) == 1

        mAP, aps = run_evaluation(
            gt_dir=GT_DIR,
            pred_dir=OUTPUT_JSON_DIR,
            image_dir=INPUT_DIR,
            vis_dir=PROCESSED_VIS_DIR,
            postprocess_iou_thresh=postprocess_thresh,
            save_processed_vis=save_processed_vis_flag
        )
        results_summary[run_name] = mAP
        
        print(f"📈 mAP@0.5 for {run_name}: {mAP:.4f}")

        if mAP > best_map:
            best_map = mAP
            best_params = params

    print(f"\n\n{'='*50}\n🏆 FINAL SUMMARY\n{'='*50}")
    for run_name, map_score in sorted(results_summary.items(), key=lambda item: -item[1]):
        print(f"  - {map_score:.4f} mAP: {run_name}")

    print(f"\n🥇 Best mAP: {best_map:.4f}")
    print(f"🎉 Best parameters: {best_params}")

if __name__ == "__main__":
    main()
