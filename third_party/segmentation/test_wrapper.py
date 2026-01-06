import cv2
import numpy as np
import os
import sys
import time
import torch
from segmentation_wrapper import SegmentationWrapper

def test_wrapper():
    print("=== Testing SegmentationWrapper ===")
    
    # 1. Initialize
    wrapper = SegmentationWrapper()
    # Use FastSAM-s.pt if available, or download/use another
    model_path = "FastSAM-s.pt" 
    if not os.path.exists(model_path):
        print(f"Model {model_path} not found, trying sam2.1_b.pt")
        model_path = "sam2.1_b.pt"
        
    # Initialize with FastSAM (or SAM)
    # Note: "fast_sam" triggers FastSAM logic, "sam" triggers SAM logic
    if "FastSAM" in model_path:
        model_type = "fast_sam"
    elif "yolo" in model_path.lower():
        model_type = "yolo"
    else:
        model_type = "mobile_sam"
        
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading {model_type} from {model_path} on {device}...")
    if not wrapper.initialize(model_type, model_path, device):
        print("Failed to initialize wrapper")
        return

    # 2. Load Image
    # Check command line args
    if len(sys.argv) > 1:
        image_path = sys.argv[1]
    else:
        image_path = "frame1368.jpg"

    if not os.path.exists(image_path):
        # Create dummy image if not exists
        print("Creating dummy test image...")
        image = np.zeros((480, 640, 3), dtype=np.uint8)
        cv2.rectangle(image, (100, 100), (300, 300), (255, 255, 255), -1) # White box
        cv2.circle(image, (400, 200), 50, (0, 0, 255), -1) # Red circle
    else:
        print(f"Loading image {image_path}...")
        image = cv2.imread(image_path)

    # 3. Segment
    print("Running warmup...")
    # Warmup with a dummy image to initialize CUDA context and model
    dummy_image = np.zeros((640, 640, 3), dtype=np.uint8)
    wrapper.segment(dummy_image)
    
    print("Running segmentation...")
    t_start = time.time()
    mask = wrapper.segment(image)
    t_cost = time.time() - t_start
    
    print(f"Segmentation complete. Mask shape: {mask.shape}")
    
    # 4. Analyze Results
    unique_ids = np.unique(mask)
    print(f"Found {len(unique_ids)-1} instances (excluding background).")
    print("Instance IDs found:", unique_ids)
    
    for uid in unique_ids:
        if uid == 0: continue
        name = wrapper.get_class_name(uid)
        print(f"  ID {uid}: {name}")
        
    # 5. Visualize Mask (Colorize instances)
    # Create a color map
    # Fix: Size should be max_id + 1 to accommodate all IDs
    max_id = np.max(unique_ids) if len(unique_ids) > 0 else 0
    colors = np.random.randint(0, 255, (max_id + 1, 3), dtype=np.uint8)
    colors[0] = [128, 128, 128] # Background gray
    
    h, w = mask.shape
    vis_mask = np.zeros((h, w, 3), dtype=np.uint8)
    
    # Draw mask
    for uid in unique_ids:
        # if uid == 0: continue
        vis_mask[mask == uid] = colors[uid]
        
    # Draw text labels (Only ID)
    for uid in unique_ids:
        # if uid == 0: continue
        # Find center of the object
        y_indices, x_indices = np.where(mask == uid)
        if len(y_indices) > 0:
            cy = int(np.mean(y_indices))
            cx = int(np.mean(x_indices))
            
            # Draw text with background for visibility
            text = f"{uid}" # Only ID
            (text_w, text_h), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(vis_mask, (cx, cy - text_h - 5), (cx + text_w, cy + 5), (0,0,0), -1)
            cv2.putText(vis_mask, text, (cx, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
    vis_filename = f"test_instance_mask_vis_{model_type}.png"
    cv2.imwrite(vis_filename, vis_mask)
    print(f"Saved instance mask visualization to {vis_filename}")
    
    # 6. Overlay on original image
    # Blend original image with colored mask
    alpha = 0.5
    overlay = cv2.addWeighted(image, 1 - alpha, vis_mask, alpha, 0)
    
    # Annotate time on top-right
    time_text = f"Time: {t_cost:.3f}s"
    # Get text size to position correctly
    (tw, th), _ = cv2.getTextSize(time_text, cv2.FONT_HERSHEY_SIMPLEX, 1.0, 2)
    # Position: top-right with some padding
    cv2.putText(overlay, time_text, (w - tw - 10, th + 10), 
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
    
    overlay_filename = f"segmentation_overlay_{model_type}.jpg"
    cv2.imwrite(overlay_filename, overlay)
    print(f"Saved overlay visualization to {overlay_filename}")

if __name__ == "__main__":
    test_wrapper()
