import cv2
import numpy as np
import os
import time
import torch
from segmentation_wrapper import SegmentationWrapper

def test_wrapper():
    print("=== Testing SegmentationWrapper ===")
    
    input_dir = "images"
    output_dir = "results"
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    if not os.path.exists(input_dir):
        print(f"Input directory './{input_dir}' not found. Please create it and add images.")
        return

    image_files = sorted([f for f in os.listdir(input_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))])
    
    if not image_files:
        print(f"No images found in {input_dir}")
        return
    print(f"Found {len(image_files)} images in {input_dir}")

    wrapper = SegmentationWrapper()
    
    # --- Configuration ---
    # 0: No PostProcess
    # 1: Original Slow PostProcess (High Quality)
    # 2: New Optimized Fast PostProcess
    POSTPROCESS_MODE = 2
    
    # List of models/methods to test.
    # Supported: Filenames (.pt, .onnx) or "slic"
    # Example: ["FastSAM-s_320.onnx", "FastSAM-s.pt", "sam2.1_b.pt", "slic"]
    MODELS_TO_TEST = [
        "FastSAM-s_320.onnx",
        # "FastSAM-s.pt",
        # "yolo11n-seg.pt",
        # "slic"

    ]
    
    modes_to_test = []

    for item in MODELS_TO_TEST:
        if item == "slic":
            modes_to_test.append(("slic", "", "cpu"))
            continue
            
        if not os.path.exists(item):
            print(f"Warning: Model file '{item}' not found. Skipping...")
            continue
            
        if item.endswith(".onnx"):
            # Assume FastSAM ONNX for now
            modes_to_test.append(("fast_sam", item, "cpu"))
        elif item.endswith(".pt"):
            if "FastSAM" in item:
                 modes_to_test.append(("fast_sam", item, "cuda"))
            elif "sam" in item.lower():
                 modes_to_test.append(("mobile_sam", item, "cuda"))
            elif "yolo" in item.lower():
                 modes_to_test.append(("yolo", item, "cuda"))
            else:
                 modes_to_test.append(("yolo", item, "cuda")) # Default

    for model_type, model_path, device_pref in modes_to_test:
        print(f"\n------------------------------------------------")
        is_onnx = model_path.endswith('.onnx')
        mode_label = f"{model_type} {'(ONNX Accelerated)' if is_onnx else '(PyTorch)'}"
        print(f"Starting Test for Mode: {mode_label}")
        print(f"------------------------------------------------")

        if device_pref == "cuda" and not torch.cuda.is_available():
            print("CUDA not available, switching to CPU")
            device_pref = "cpu"

        # Initialize
        if not wrapper.initialize(model_type, model_path, device_pref):
            print(f"Failed to initialize {model_type}, skipping...")
            continue
        
        # Warmup for DL models
        if model_type != "slic":
            print("Running warmup...")
            dummy_image = np.zeros((640, 640, 3), dtype=np.uint8)
            wrapper.segment(dummy_image, postprocess_mode=POSTPROCESS_MODE)

        for image_file in image_files:
            image_path = os.path.join(input_dir, image_file)
            print(f"Processing {image_file}...")
            
            image = cv2.imread(image_path)
            if image is None:
                print(f"Failed to load {image_path}")
                continue

            # Segment
            t_start = time.time()
            mask = wrapper.segment(image, postprocess_mode=POSTPROCESS_MODE)
            t_cost = time.time() - t_start
            


            # Visualization
            overlay = wrapper.get_visualization(image, mask)
            base_name = os.path.splitext(image_file)[0]
            overlay_filename = os.path.join(output_dir, f"{base_name}_overlay_{model_type}.jpg")
            
            cv2.imwrite(overlay_filename, overlay)
            print(f"Saved result: {overlay_filename}")
            
            # Save ID 0 Visualization (Unassigned/Holes)
            vis_id0 = wrapper.get_single_id_visualization(image, mask, target_id=0)
            id0_filename = os.path.join(output_dir, f"{base_name}_id0_{model_type}.jpg")
            cv2.imwrite(id0_filename, vis_id0)
            print(f"Saved ID=0 visualization: {id0_filename}")
            
            # Save speed stats
            speed_log = os.path.join(output_dir, "speed_benchmark.txt")
            with open(speed_log, "a") as f:
                f.write(f"{mode_label} - {image_file}: {t_cost:.4f}s\n")

if __name__ == "__main__":
    test_wrapper()
