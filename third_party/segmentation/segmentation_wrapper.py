import sys
import os
import numpy as np
import cv2
import torch
import time
from ultralytics import SAM, FastSAM, YOLO

class SegmentationWrapper:
    def __init__(self):
        self.model = None
        self.model_type = ""
        self.device = "cpu"
        self.classes = {}
        self.instance_map = {} # Map Instance ID -> Class Name for the last segmented image

    def initialize(self, model_type, model_path, device):
        self.model_type = model_type
        self.device = device
        print(f"Initializing {model_type} on {device} with weights from {model_path}")

        try:
            if model_type == "mobile_sam":
                # Use ultralytics SAM
                self.model = SAM(model_path)
                # SAM is class agnostic
                self.classes = {0: "object"} 
                
            elif model_type == "fast_sam":
                # Use ultralytics FastSAM
                self.model = FastSAM(model_path)
                self.classes = self.model.names if hasattr(self.model, 'names') else {}
                
            else:
                # Default to YOLO/Generic
                self.model = YOLO(model_path)
                self.classes = self.model.names if hasattr(self.model, 'names') else {}
                
            return True
        except Exception as e:
            print(f"Error initializing model: {e}")
            import traceback
            traceback.print_exc()
            return False

    def segment_from_bytes_return_bytes(self, image_bytes):
        """
        Args:
            image_bytes: bytes object containing encoded image
        Returns:
            tuple: (mask_bytes, rows, cols)
            mask_bytes is raw int32 bytes
        """
        nparr = np.frombuffer(image_bytes, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        mask = self.segment(image)
        return (mask.tobytes(), mask.shape[0], mask.shape[1])

    def segment_from_bytes(self, image_bytes):
        """
        Args:
            image_bytes: bytes object containing encoded image (e.g. jpg/png)
        """
        nparr = np.frombuffer(image_bytes, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        return self.segment(image)

    def segment(self, image):
        """
        Args:
            image: numpy array (H, W, 3) BGR
        Returns:
            mask: numpy array (H, W) with integer IDs.
                  Instance segmentation with merging of similar/low-confidence regions.
                  Pixels not belonging to any mask are assigned ID 0 (background).
        """
        if self.model is None:
            print("Model not initialized")
            return np.zeros(image.shape[:2], dtype=np.int32)

        self.instance_map = {0: 'background'} # Reset map, ID 0 is background

        t_start = time.time()
        try:
            # Run inference
            # conf=0.25: Filter low confidence detections initially
            # iou=0.7: Higher NMS threshold to merge overlapping boxes
            # Use retina_masks=True to get full-size masks with correct aspect ratio.
            # This is usually fast enough on GPU and avoids manual resizing loops.
            t_infer_start = time.time()
            results = self.model(image, device=self.device, retina_masks=True, verbose=False, conf=0.25, iou=0.7)
            t_infer_end = time.time()
            
            if not results:
                return np.zeros(image.shape[:2], dtype=np.int32)

            result = results[0]
            
            # Initialize final mask
            # We will build it on the device first
            
            t_post_start = time.time()
            if result.masks is not None:
                # Keep data on device (GPU if available) to avoid CPU transfer overhead
                masks = result.masks.data # (N, H, W)
                
                # Get classes
                if result.boxes is not None and hasattr(result.boxes, 'cls') and result.boxes.cls is not None:
                    classes = result.boxes.cls # (N,)
                else:
                    classes = torch.zeros(len(masks), device=masks.device)

                # 1. Filter small areas
                # masks is float (0..1) or bool. Usually float.
                binary_masks = masks > 0.5
                areas = binary_masks.sum(dim=(1, 2)) # (N,)
                
                min_area_threshold = 1000
                keep_indices = areas > min_area_threshold
                
                if not keep_indices.any():
                     return np.zeros(image.shape[:2], dtype=np.int32)

                binary_masks = binary_masks[keep_indices]
                classes = classes[keep_indices]
                areas = areas[keep_indices]
                
                # 2. Sort by area (Descending: Large -> Small)
                # We paint Large first, then Small. Small objects will overwrite Large ones (Painter's Algorithm).
                # This correctly handles "object in front of background".
                sorted_indices = torch.argsort(areas, descending=True)
                binary_masks = binary_masks[sorted_indices]
                classes = classes[sorted_indices]

                # Dynamic limit based on total number of detected objects
                # Adjust max number based on total count, allowing more objects if many are detected, but capping to keep total small.
                total_detected = len(binary_masks)
                if total_detected > 15:
                    # Keep roughly 60% of objects, but ensure at least 15 and at most 25
                    keep_count = int(total_detected * 0.6)
                    
                    binary_masks = binary_masks[:keep_count]
                    classes = classes[:keep_count]
                
                # 3. Assign IDs
                num_masks = len(binary_masks)
                
                # Create final mask tensor on device
                final_mask_tensor = torch.zeros((binary_masks.shape[1], binary_masks.shape[2]), dtype=torch.int32, device=self.device)
                
                # Iterate and paint
                # This loop is fast on GPU (kernel launches)
                for i in range(num_masks):
                     # ID starts from 1
                     final_mask_tensor[binary_masks[i]] = i + 1
                
                # Move result to CPU once
                final_mask = final_mask_tensor.cpu().numpy().astype(np.int32)
                
                # Update instance_map
                classes_cpu = classes.cpu().numpy().astype(int)
                names = self.model.names if hasattr(self.model, 'names') else {}
                
                for i, cls_id in enumerate(classes_cpu):
                    instance_id = i + 1
                    if cls_id in names:
                        self.instance_map[instance_id] = names[cls_id]
                    else:
                        self.instance_map[instance_id] = f"class_{cls_id}"
            else:
                final_mask = np.zeros(image.shape[:2], dtype=np.int32)

            t_post_end = time.time()
            
            print(f"Time Stats: Inference={t_infer_end - t_infer_start:.4f}s, Post-process={t_post_end - t_post_start:.4f}s, Total={t_post_end - t_start:.4f}s")
            return final_mask

        except Exception as e:
            print(f"Error during segmentation: {e}")
            import traceback
            traceback.print_exc()
            return np.zeros(image.shape[:2], dtype=np.int32)

    def get_class_name(self, mask_id):
        """
        Get class name for a specific mask ID from the last segmentation.
        """
        if mask_id in self.instance_map:
            return self.instance_map[mask_id]
        return f"unknown_{mask_id}"

# Helper function to create instance
def create_segmentator():
    return SegmentationWrapper()
