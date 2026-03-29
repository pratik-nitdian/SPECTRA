import os
import time
import json
import argparse
import cv2
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor

from src import preprocessing
from src.document_engine import DocumentEngine
from src.utils import visualize_layout, process_detections

def get_args():
    parser = argparse.ArgumentParser(description="YOLOv10 Document Layout Analysis Pipeline")
    
    parser.add_argument("--source", type=str, required=True, help="Path to input image folder")
    parser.add_argument("--output", type=str, default ="./output", help="Path to save outputs")
    parser.add_argument("--model", type=str, default ="doclayout_yolo_doclaynet_imgsz1120_docsynth_pretrain.pt", help="Path to .pt model file")
    parser.add_argument("--surya-checkpoint", type=str, default ="./my_standalone_ocr/2025_09_23", help="Path to surya checkpoint directory")
    parser.add_argument("--florence-model", type=str, default ="microsoft/Florence-2-base", help="Florence-2 model ID or path")

    parser.add_argument("--imgsz", type=int, default=1120, help="Inference image size")
    parser.add_argument("--conf", type=float, default=0.25, help="Confidence threshold")
    parser.add_argument("--iou-thresh", type=float, default=0.5, help="IOU threshold for post-processing NMS")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size for inference")

    parser.add_argument("--deskew", type=int, default=1, choices=[0, 1], help="Enable Deskew")
    parser.add_argument("--denoise", type=int, default=0, choices=[0, 1], help="Enable Denoise")
    parser.add_argument("--clahe", type=int, default=0, choices=[0, 1], help="Enable CLAHE Contrast")
    parser.add_argument("--sharpen", type=int, default=0, choices=[0, 1], help="Enable Unsharp Mask")
    
    parser.add_argument("--save-vis", type=int, default=1, choices=[0, 1], help="Save visualized images")

    return parser.parse_args()

def process_wrapper(args):
    input_path, output_path, config = args
    return preprocessing.process_and_save(input_path, output_path, config)

def main():
    args = get_args()

    FLAGS = {
        'deskew': args.deskew, 'denoise': args.denoise,
        'clahe': args.clahe, 'sharpen': args.sharpen,
        'save_vis': args.save_vis
    }

    PREPROCESSED_DIR = os.path.join(args.output, "preprocessed")
    OUTPUT_JSON_DIR = os.path.join(args.output, "json")
    OUTPUT_VIS_DIR = os.path.join(args.output, "visualized")

    os.makedirs(PREPROCESSED_DIR, exist_ok=True)
    os.makedirs(OUTPUT_JSON_DIR, exist_ok=True)
    if FLAGS['save_vis']:
        os.makedirs(OUTPUT_VIS_DIR, exist_ok=True)

    if not os.path.exists(args.source):
        print(f"❌ Error: Input directory '{args.source}' not found.")
        return

    image_files = [f for f in os.listdir(args.source) if f.lower().endswith((".jpg", ".png", ".jpeg"))]
    print(f"\n📂 Found {len(image_files)} images in {args.source}")

    tasks, inference_queue = [], []
    for filename in image_files:
        base_name = os.path.splitext(filename)[0]
        raw_path = os.path.join(args.source, filename)
        npy_path = os.path.join(PREPROCESSED_DIR, f"{base_name}.npy")
        tasks.append((raw_path, npy_path, FLAGS))
        inference_queue.append((npy_path, filename))

    files_exist = all(os.path.exists(t[1]) for t in tasks)
    if files_exist:
        print("⏩ All .npy files exist. Skipping Preprocessing.")
    else:
        num_cores = os.cpu_count()
        print(f"\n⏳ Starting Preprocessing ({num_cores} cores)...")
        with ProcessPoolExecutor(max_workers=num_cores) as executor:
            list(tqdm(executor.map(process_wrapper, tasks), total=len(tasks), desc="Preprocessing"))

    print(f"\n🚀 Loading Models...")
    try:
        engine = DocumentEngine(args.model, surya_checkpoint=args.surya_checkpoint, 
                                florence_model_id=args.florence_model,
                                img_size=args.imgsz, conf_threshold=args.conf)
    except Exception as e:
        print(f"❌ Failed to load model: {e}")
        return

    valid_queue = [item for item in inference_queue if os.path.exists(item[0])]
    batches = [valid_queue[i:i + args.batch_size] for i in range(0, len(valid_queue), args.batch_size)]

    print(f"⚡ Running Inference on {len(batches)} batches...")
    start_time = time.time()
    for batch in tqdm(batches, desc="Inference"):
        engine.run_batch(batch, OUTPUT_JSON_DIR, OUTPUT_VIS_DIR, save_vis=0)

    total_time = time.time() - start_time
    fps = len(valid_queue) / total_time if total_time > 0 else 0
    
    json_files = [os.path.join(OUTPUT_JSON_DIR, f) for f in os.listdir(OUTPUT_JSON_DIR) if f.endswith(".json")]

    if args.iou_thresh > 0:
        print(f"\n🔄 Post-processing detections with IOU threshold: {args.iou_thresh}")
        for json_path in tqdm(json_files, desc="Post-processing"):
            with open(json_path, 'r') as f:
                data = json.load(f)
            data['annotations'] = process_detections(data['annotations'], iou_thresh=args.iou_thresh)
            with open(json_path, 'w') as f:
                json.dump(data, f, indent=4)

    if FLAGS['save_vis']:
        print(f"\n🎨 Generating visualizations...")
        for json_path in tqdm(json_files, desc="Visualizing"):
            with open(json_path, 'r') as f:
                data = json.load(f)
            image_filename = data["file_name"]
            original_image_path = os.path.join(args.source, image_filename)
            output_vis_path = os.path.join(OUTPUT_VIS_DIR, os.path.splitext(image_filename)[0] + ".jpg")
            visualize_layout(original_image_path, data['annotations'], output_vis_path)

    print("\n✅ DLA Pipeline Complete!")
    print(f"📁 JSON: {OUTPUT_JSON_DIR}")
    if FLAGS['save_vis']: print(f"🖼️  Vis:  {OUTPUT_VIS_DIR}")
    print(f"⏱️  FPS:  {fps:.2f}")

if __name__ == "__main__":
    main()
