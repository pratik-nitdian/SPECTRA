import cv2
import numpy as np

# Class names for document layout visualization
LAYOUT_CLASS_NAMES = {
    0: "Background", 1: "Text", 2: "Title",
    3: "List", 4: "Table", 5: "Figure"
}

def visualize_layout(image_path, boxes, save_path):
    """Visualizes layout detections on an image and saves it."""
    img = cv2.imread(image_path)
    if img is None:
        print(f"⚠️ Could not open image {image_path}")
        return
    for box in boxes:
        x, y, w, h = map(int, box["bbox"])
        cat_id = box["category_id"]
        conf = box["confidence"]
        color = (0, 255, 0) if cat_id in [4, 5] else (0, 0, 255)
        label = f"{LAYOUT_CLASS_NAMES.get(cat_id, f'ID {cat_id}')} ({conf:.2f})"
        cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)
        cv2.putText(img, label, (x, max(15, y - 5)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)
    cv2.imwrite(save_path, img)

def compute_iou(box1, box2):
    """Computes Intersection over Union (IoU) of two bounding boxes."""
    x1, y1, w1, h1 = box1
    x2, y2, w2, h2 = box2

    xa = max(x1, x2)
    ya = max(y1, y2)
    xb = min(x1 + w1, x2 + w2)
    yb = min(y1 + h1, y2 + h2)

    inter = max(0, xb - xa) * max(0, yb - ya)
    area1 = w1 * h1
    area2 = w2 * h2

    union = area1 + area2 - inter
    return inter / union if union > 0 else 0

def is_inside(inner, outer):
    """Returns True if inner bbox is fully inside outer bbox."""
    x1, y1, w1, h1 = inner
    x2, y2, w2, h2 = outer
    return (x1 >= x2 and y1 >= y2 and
            x1 + w1 <= x2 + w2 and
            y1 + h1 <= y2 + h2)

def process_detections(boxes, iou_thresh=0.5):
    """Filters and cleans up layout detections using NMS and containment rules."""
    boxes = sorted(boxes, key=lambda x: x["confidence"], reverse=True)
    keep = []

    for i, b1 in enumerate(boxes):
        remove_flag = False
        for kept in keep:
            iou = compute_iou(b1["bbox"], kept["bbox"])
            if iou > iou_thresh:
                remove_flag = True
                break
        if not remove_flag:
            keep.append(b1)

    final_boxes = []
    for b1 in keep:
        contained = False
        for b2 in keep:
            if b2["category_id"] in [4, 5] and b1["category_id"] != b2["category_id"]:
                if is_inside(b1["bbox"], b2["bbox"]):
                    contained = True
                    break
        if not contained:
            final_boxes.append(b1)

    return final_boxes
