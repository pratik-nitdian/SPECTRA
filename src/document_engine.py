import cv2
import json
import numpy as np
import torch
from PIL import Image
from doclayout_yolo import YOLOv10
from transformers import AutoProcessor, AutoModelForCausalLM

# Import Surya components
from surya.recognition import RecognitionPredictor
from surya.detection import DetectionPredictor
from surya.foundation import FoundationPredictor

class DocumentEngine:
    def __init__(self, model_path, surya_checkpoint="./my_standalone_ocr/2025_09_23", 
                 florence_model_id="microsoft/Florence-2-base", img_size=1120, conf_threshold=0.25):
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        
        # 1. Layout Detection: YOLOv10 (doclayout)
        print(f"🚀 Loading DocLayout model: {model_path}")
        self.model = YOLOv10(model_path)
        self.img_size = img_size
        self.conf_threshold = conf_threshold
        self.class_names = self.model.names if hasattr(self.model, 'names') else {
            0: "Background", 1: "Text", 2: "Title",
            3: "List", 4: "Table", 5: "Figure"
        }

        # 2. OCR: Surya OCR
        print(f"🚀 Loading Surya OCR from: {surya_checkpoint}")
        self.found_pred = FoundationPredictor(checkpoint=surya_checkpoint, device=self.device)
        self.det_pred = DetectionPredictor(device=self.device)
        self.rec_pred = RecognitionPredictor(self.found_pred)

        # 3. Load Florence-2 Model for Descriptions and OCR
        print(f"🚀 Loading Florence-2 model: {florence_model_id}")
        self.florence_processor = AutoProcessor.from_pretrained(florence_model_id, trust_remote_code=True)
        self.florence_model = AutoModelForCausalLM.from_pretrained(florence_model_id, trust_remote_code=True, attn_implementation="eager").to(self.device).eval()


    def run_florence_description(self, image, category):
        # Determine task prompt based on element type
        task_prompt = "<MORE_DETAILED_CAPTION>"
        if category == "Table":
            task_prompt = "<MORE_DETAILED_CAPTION>" # Better for structured/tabular explanation
            
        inputs = self.florence_processor(text=task_prompt, images=image, return_tensors="pt").to(self.device)
        
        if self.florence_model.dtype == torch.float16:
            inputs["pixel_values"] = inputs["pixel_values"].to(torch.float16)
        elif self.florence_model.dtype == torch.bfloat16:
            inputs["pixel_values"] = inputs["pixel_values"].to(torch.bfloat16)

        with torch.no_grad():
            generated_ids = self.florence_model.generate(
                input_ids=inputs["input_ids"],
                pixel_values=inputs["pixel_values"],
                max_new_tokens=512,
                num_beams=3,
                use_cache=False
            )
        generated_text = self.florence_processor.batch_decode(generated_ids, skip_special_tokens=False)[0]
        parsed_answer = self.florence_processor.post_process_generation(
            generated_text, 
            task=task_prompt, 
            image_size=(image.width, image.height)
        )
        return parsed_answer.get(task_prompt, "")

    def run_batch(self, batch_data, json_dir, vis_dir, save_vis=True):
        # 1. Load Data
        images_cv2 = []
        filenames = []
        for npy_path, fname in batch_data:
            try:
                img = np.load(npy_path)
                images_cv2.append(img)
                filenames.append(fname)
            except: continue

        if not images_cv2: return

        # Step 1: Detect Layout with YOLO (DocLayout)
        results = self.model.predict(source=images_cv2, imgsz=self.img_size, conf=self.conf_threshold, device=self.device, verbose=False)

        # Step 2: Run Surya OCR on all images in the batch to find lines
        images_pil = [Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)) for img in images_cv2]
        surya_ocr_results = self.rec_pred(images_pil, det_predictor=self.det_pred, sort_lines=True)

        for i, result in enumerate(results):
            filename = filenames[i]
            img_cv2 = images_cv2[i]
            (h, w) = img_cv2.shape[:2]
            img_pil = images_pil[i]
            
            detections = result.boxes.data.cpu().numpy()
            annotations = []
            
            # Step 3: Initialize annotations and identify elements
            for det in detections:
                x1, y1, x2, y2, conf, cls_id = det
                cat_name = self.class_names.get(int(cls_id), "Unknown")
                
                element_data = {
                    "id": f"block_{i}_{len(annotations)}",
                    "bbox": [float(x1), float(y1), float(x2 - x1), float(y2 - y1)],
                    "category_id": int(cls_id),
                    "category_name": cat_name,
                    "confidence": float(conf),
                    "text": "",
                    "description": "",
                    "languages": ["auto"]
                }
                annotations.append(element_data)

            # Step 4: Map Surya lines back to YOLO layout blocks
            img_ocr = surya_ocr_results[i]
            for ann in annotations:
                if ann["category_name"] in ["Text", "Title", "List"]:
                    bx1, by1, bw, bh = ann["bbox"]
                    bx2, by2 = bx1 + bw, by1 + bh
                    
                    # Find Surya lines whose center is within the YOLO block
                    relevant_lines = []
                    for s_line in img_ocr.text_lines:
                        lx1, ly1, lx2, ly2 = s_line.bbox
                        lcx, lcy = (lx1 + lx2) / 2, (ly1 + ly2) / 2
                        if bx1 <= lcx <= bx2 and by1 <= lcy <= by2:
                            relevant_lines.append(s_line)
                    
                    # Vertical sort and join text
                    relevant_lines.sort(key=lambda l: (l.bbox[1], l.bbox[0]))
                    ann["text"] = "\n".join([l.text for l in relevant_lines])

            # Step 5: Run Florence-2 for descriptions (Table, Figure, Image, Map, Chart)
            for ann in annotations:
                if ann["category_name"] in ["Table", "Figure", "Image", "Map", "Chart"]:
                    # Crop the region for Florence
                    x, y, bw, bh = ann["bbox"]
                    crop_x1, crop_y1, crop_x2, crop_y2 = max(0, int(x)), max(0, int(y)), min(w, int(x+bw)), min(h, int(y+bh))
                    if crop_x2 > crop_x1 and crop_y2 > crop_y1:
                        crop_pil = img_pil.crop((crop_x1, crop_y1, crop_x2, crop_y2))
                        ann["description"] = self.run_florence_description(crop_pil, ann["category_name"])

            # Step 6: Save Outputs
            base_name = filename.rsplit('.', 1)[0]
            with open(f"{json_dir}/{base_name}.json", "w") as f:
                json.dump({"file_name": filename, "annotations": annotations}, f, indent=4)

            self._save_xml(filename, annotations, f"{json_dir}/{base_name}.xml")

            if save_vis:
                self._save_visualization(img_cv2, annotations, f"{vis_dir}/{base_name}.jpg")

    def _save_xml(self, filename, annotations, save_path):
        import xml.etree.ElementTree as ET
        root = ET.Element("document", filename=filename)
        for ann in annotations:
            el = ET.SubElement(root, "element")
            ET.SubElement(el, "category").text = ann["category_name"]
            bbox = ET.SubElement(el, "bbox")
            bbox.set("x", str(ann["bbox"][0])); bbox.set("y", str(ann["bbox"][1]))
            bbox.set("w", str(ann["bbox"][2])); bbox.set("h", str(ann["bbox"][3]))
            ET.SubElement(el, "text").text = ann["text"]
            ET.SubElement(el, "description").text = ann["description"]
        tree = ET.ElementTree(root)
        tree.write(save_path, encoding="utf-8", xml_declaration=True)

    def _save_visualization(self, image, boxes, save_path):
        vis_img = image.copy()
        for box in boxes:
            x, y, w, h = map(int, box["bbox"])
            label_text = box["category_name"]
            color = (0, 0, 255) # Red
            if box["category_name"] == "Table": color = (0, 255, 0) # Green
            if box["category_name"] == "Figure": color = (255, 0, 0) # Blue
            
            cv2.rectangle(vis_img, (x, y), (x + w, y + h), color, 2)
            display = f"{label_text}"
            content = box["description"] or box["text"]
            if content: display += f": {content[:30]}..."
            cv2.putText(vis_img, display, (x, max(15, y - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
        cv2.imwrite(save_path, vis_img)
