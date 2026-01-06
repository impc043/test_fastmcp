import cv2
import numpy as np

def get_robust_signature_crop(image_path, llm_coords, padding_pct=0.10):
    """
    llm_coords: Dict with {'xmin_pct', 'ymin_pct', 'xmax_pct', 'ymax_pct'}
    padding_pct: 0.10 adds 10% extra space around the predicted box
    """
    # 1. Load Image
    img = cv2.imread(image_path)
    if img is None: return None
    h, w = img.shape[:2]

    # 2. Convert Percentages to Pixels
    x1 = int(llm_coords['xmin_pct'] * w)
    y1 = int(llm_coords['ymin_pct'] * h)
    x2 = int(llm_coords['xmax_pct'] * w)
    y2 = int(llm_coords['ymax_pct'] * h)

    # 3. Apply Dynamic Padding
    box_w = x2 - x1
    box_h = y2 - y1
    
    pad_w = int(box_w * padding_pct)
    pad_h = int(box_h * padding_pct)

    # Calculate new coordinates with clamping to image boundaries
    px1 = max(0, x1 - pad_w)
    py1 = max(0, y1 - pad_h)
    px2 = min(w, x2 + pad_w)
    py2 = min(h, y2 + pad_h)

    # 4. Perform Initial Crop
    initial_crop = img[py1:py2, px1:px2]

    # 5. Content-Aware Refinement (Precision Snap)
    # Convert to gray and use Otsu's threshold to isolate the 'ink'
    gray = cv2.cvtColor(initial_crop, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # Find contours of the ink
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if contours:
        # Get the bounding box of ALL ink found in the padded area
        all_points = np.concatenate(contours)
        rx, ry, rw, rh = cv2.boundingRect(all_points)
        
        # Final Precise Crop
        final_crop = initial_crop[ry:ry+rh, rx:rx+rw]
        
        # Calculate final absolute coordinates for metadata
        final_coords = {
            "abs_x1": px1 + rx,
            "abs_y1": py1 + ry,
            "abs_x2": px1 + rx + rw,
            "abs_y2": py1 + ry + rh
        }
        return final_crop, final_coords

    return initial_crop, {"abs_x1": px1, "abs_y1": py1, "abs_x2": px2, "abs_y2": py2}

# --- Usage Example ---
# coords = {'xmin_pct': 0.72, 'ymin_pct': 0.81, 'xmax_pct': 0.95, 'ymax_pct': 0.92}
# cropped_sign, metadata = get_robust_signature_crop("contract.jpg", coords)
# cv2.imwrite("extracted_sign.png", cropped_sign)