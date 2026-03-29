import cv2
import numpy as np
import time
from jdeskew.estimator import get_angle
from jdeskew.utility import rotate

def process_and_save(input_path, output_path, config):
    """
    Reads image, runs preprocessing steps defined in 'config',
    and saves as .npy. Returns timing statistics.
    """
    try:
        stats = {}
        
        # 1. Read
        t_start = time.time()
        img = cv2.imread(input_path)
        if img is None: return None
        t_read = time.time()
        stats['Reading'] = t_read - t_start

        t_last = t_read

        # 2. Deskew (Using jdeskew)
        if config.get('deskew', 1):
            try:
                angle = get_angle(img)
                img = rotate(img, angle)
            except Exception:
                # Fallback if jdeskew fails on complex images
                pass
            
        t_deskew = time.time()
        stats['Deskew'] = t_deskew - t_last
        t_last = t_deskew

        # 3. Denoise / Grayscale Ops
        # Create grayscale copy if needed
        if config.get('denoise', 1) or config.get('clahe', 1) or config.get('sharpen', 1):
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        if config.get('denoise', 1):
            # This is the slow, high-quality step
            gray = cv2.fastNlMeansDenoising(gray, h=10, templateWindowSize=7, searchWindowSize=21)
        
        t_denoise = time.time()
        stats['Denoise'] = t_denoise - t_last
        t_last = t_denoise

        # 4. Contrast (CLAHE)
        if config.get('clahe', 1):
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            gray = clahe.apply(gray)
        
        t_clahe = time.time()
        stats['CLAHE'] = t_clahe - t_last
        t_last = t_clahe

        # 5. Sharpen
        if config.get('sharpen', 1):
            blur = cv2.GaussianBlur(gray, (5, 5), 1.5)
            sharpened = cv2.addWeighted(gray, 1.5, blur, -0.5, 0)
            gray = np.clip(sharpened, 0, 255).astype(np.uint8)

        t_sharpen = time.time()
        stats['Sharpen'] = t_sharpen - t_last
        t_last = t_sharpen

        # 6. Save
        # Convert back to BGR if we did grayscale ops
        if config.get('denoise', 1) or config.get('clahe', 1) or config.get('sharpen', 1):
            enhanced_bgr = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
        else:
            enhanced_bgr = img

        np.save(output_path, enhanced_bgr)
        
        t_save = time.time()
        stats['Saving'] = t_save - t_last
        stats['Total'] = t_save - t_start
        
        return stats

    except Exception:
        return None