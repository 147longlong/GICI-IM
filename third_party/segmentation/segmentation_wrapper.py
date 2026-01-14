import sys
import os
import ctypes
import numpy as np

# Workaround for Qt6/OpenCV freetype symbol issue
# We now rely on opencv-python-headless being installed to avoid Qt/Freetype conflicts entirely.

import os
# Force matplotlib to use non-interactive backend to avoid GUI lib loading
os.environ['MPLBACKEND'] = 'Agg'

conda_lib_path = os.path.join(sys.prefix, 'lib')
libopenjp2_path = os.path.join(conda_lib_path, 'libopenjp2.so.7')
if os.path.exists(libopenjp2_path):
    try:
        ctypes.CDLL(libopenjp2_path, mode=ctypes.RTLD_GLOBAL)
    except Exception as e:
        print(f"[PYTHON] Warning: Failed to preload libopenjp2 from {libopenjp2_path}: {e}")


import cv2
import torch
import torch.nn.functional as F
import time
import math
from ultralytics import SAM, FastSAM, YOLO

try:
    import matplotlib
    matplotlib.use('Agg')
except ImportError:
    pass

try:
    print(f"[PYTHON] Initializing FastSAM wrapper. CV2 version: {cv2.__version__}")
except Exception as e:
    print(f"[PYTHON] Error checking CV2 version: {e}")

try:
    from skimage.segmentation import slic
    SKIMAGE_AVAILABLE = True
except ImportError:
    SKIMAGE_AVAILABLE = False

try:
    # Check for OpenCV Contrib (ximgproc) availability for faster SLIC
    # This is much faster than skimage, ~20ms vs ~200ms
    cv2.ximgproc.createSuperpixelSLIC
    OPENCV_SLIC_AVAILABLE = True
except (AttributeError, ImportError):
    OPENCV_SLIC_AVAILABLE = False


class SegmentationWrapper:
    def __init__(self):
        self.model = None
        self.model_type = ""
        self.device = "cpu"
        self.classes = {}
        self.instance_map = {} # Map Instance ID -> Class Name for the last segmented image

    def initialize(self, model_type, model_path, device, config=None):
        self.config = config if config else {}
        self.model_type = model_type
        self.device = device
        print(f"[PYTHON] Initializing {model_type} on {device} with weights from {model_path}")

        try:
            if model_type == "mobile_sam":
                # Use ultralytics SAM
                self.model = SAM(model_path)
                # SAM is class agnostic
                self.classes = {0: "object"} 
                
            elif model_type == "fast_sam":
                # Use ultralytics FastSAM
                print(f"[PYTHON] Loading FastSAM model: {model_path}")
                # Ultralytics FastSAM handles .pt and .onnx automatically
                self.model = FastSAM(model_path)
                self.classes = self.model.names if hasattr(self.model, 'names') else {}
            
            elif model_type == "slic":
                if not SKIMAGE_AVAILABLE and not OPENCV_SLIC_AVAILABLE:
                    print("[PYTHON] Error: Neither scikit-image nor opencv-contrib installed. Cannot use SLIC.")
                    return False
                self.model = "slic"
                self.classes = {0: "superpixel"}
                if OPENCV_SLIC_AVAILABLE:
                    print("[PYTHON] Using OpenCV ximgproc for fast SLIC.")
                else:
                    print("[PYTHON] Using skimage for SLIC (slower).")
                
            else:
                # Default to YOLO/Generic
                self.model = YOLO(model_path)
                self.classes = self.model.names if hasattr(self.model, 'names') else {}
                
            return True
        except Exception as e:
            print(f"[PYTHON] Error initializing model: {e}")
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

    def segment(self, image, postprocess_mode = 2):
        """
        Runs segmentation on the input image.
        
        Args:
            image: numpy array (H, W, 3) BGR
            postprocess_mode: int
                0: No post-processing (Raw output, fast, no connectivity check)
                1: Original High Quality (Slower, retina_masks=True, large connectivity check)
                2: Optimized Fast (retina_masks=False, small connectivity check, unpadding) - Default
        Returns:
            mask: numpy array (H, W) with integer IDs.
        """
        # ==================================================================================
        # 1. Setup & Configuration
        # ==================================================================================
        enable_postprocess = (postprocess_mode != 0)

        MAX_MASKS_COUNTS = 200         # Maximum number of masks to output, if exceeded, smaller ones are dropped by RATIO_TO_MAX_MASKS
        RATIO_TO_MAX_MASKS = 0.6       # Dynamic ratio to limit masks when too many candidates
        MAX_SINGLE_MASK_RATIO = 0.015  # Maximum ratio of single mask area to image area before splitting
        
        print(f"[PYTHON]  Segmenting image of shape {image.shape} with postprocess_mode = {postprocess_mode}")
        
        if self.model is None:
            print("[PYTHON] Model not initialized")
            return np.zeros(image.shape[:2], dtype=np.int32)

        self.instance_map = {0: 'background'} # Reset map, ID 0 is background

        # Compatibility shim for caller using old boolean arg
        if isinstance(postprocess_mode, bool):
             postprocess_mode = 2 if postprocess_mode else 0

        t_start = time.time()
        
        # ==================================================================================
        # 2. SLIC Superpixel Path (if enabled)
        # ==================================================================================
        if self.model_type == "slic":
            if OPENCV_SLIC_AVAILABLE:
                # OpenCV ximgproc SLIC (Fastest)
                region_size = 30
                ruler = 10.0
                
                # region_size=30 roughly equivalent to 500-600 superpixels on 640x480
                slic_alg = cv2.ximgproc.createSuperpixelSLIC(image, region_size=region_size, ruler=ruler) 
                slic_alg.iterate(10) # 10 iterations usually enough
                
                # Check if we need to enforce min element size? 
                if enable_postprocess and postprocess_mode == 2:
                    slic_alg.enforceLabelConnectivity(min_element_size=50)
                
                masks = slic_alg.getLabels()
                # OpenCV labels start at 0, shift to 1
                masks = masks + 1
                
            else:
                # Skimage SLIC (Fallback)
                # Optimization: Downsample, calculate SLIC, then resize nearest neighbor
                h, w = image.shape[:2]
                target_w = 320
                scale = target_w / float(w)
                if scale < 1.0:
                    small_img = cv2.resize(image, (0,0), fx=scale, fy=scale)
                else:
                    small_img = image
                
                # Convert to RGB for skimage
                img_rgb = cv2.cvtColor(small_img, cv2.COLOR_BGR2RGB)
                
                masks_small = slic(img_rgb, n_segments=70, compactness=10, sigma=1, start_label=1)
                
                if scale < 1.0:
                    masks = cv2.resize(masks_small.astype(np.int32), (w, h), interpolation=cv2.INTER_NEAREST)
                else:
                    masks = masks_small.astype(np.int32)

            # Update instance map
            unique_ids = np.unique(masks)
            if unique_ids[0] == 0: unique_ids = unique_ids[1:]
            
            for uid in unique_ids:
                 self.instance_map[uid] = f"sp_{uid}"

            t_end = time.time()
            print(f"[PYTHON]  SLIC Time ({'OpenCV' if OPENCV_SLIC_AVAILABLE else 'Skimage'}): {t_end - t_start:.4f}s")
            return masks

        # ==================================================================================
        # 3. Model Inference (YOLO / FastSAM)
        # ==================================================================================
        try:
            # Mode specific settings
            use_retina_masks = (postprocess_mode == 1)
            
            # Run inference
            t_infer_start = time.time()
            results = self.model(image, device=self.device, retina_masks=use_retina_masks, verbose=False, conf=0.25, iou=0.7)
            t_infer_end = time.time()
            
            if not results:
                return np.zeros(image.shape[:2], dtype=np.int32)

            result = results[0]
            
            # ==================================================================================
            # 4. Post-Processing & Logic
            # ==================================================================================
            t_post_start = time.time()
            if result.masks is not None:
                # Keep data on device (GPU if available) to avoid CPU transfer overhead
                masks = result.masks.data # (N, H, W)
                
                # Get classes
                if result.boxes is not None:
                     classes = result.boxes.cls
                else:
                     classes = torch.zeros(len(masks), device=masks.device)

                # Convert to binary
                binary_masks = masks > 0.5
                areas = binary_masks.sum(dim=(1, 2)) # (N,)
                
                if enable_postprocess:
                    # -----------------------------------------------------------
                    # 4.1. Initial Filtering (Denoise)
                    # -----------------------------------------------------------
                    # Adjust area threshold based on mode
                    if postprocess_mode == 1:
                        min_area_threshold = 300 
                    else:
                        min_area_threshold = 150 

                    keep_indices = areas > min_area_threshold
                    
                    if keep_indices.any():
                        filtered_masks = binary_masks[keep_indices]
                        filtered_classes = classes[keep_indices]
                        filtered_areas = areas[keep_indices]
                    else:
                        filtered_masks = torch.empty((0, *binary_masks.shape[1:]), device=self.device, dtype=torch.bool)
                        filtered_classes = torch.empty((0,), device=self.device)
                        filtered_areas = torch.empty((0,), device=self.device)

                    # -----------------------------------------------------------
                    # 4.2. Prepare Raw Candidates (FG + BG)
                    # -----------------------------------------------------------
                    raw_candidates = []
                    
                    # Add Foreground
                    for i in range(len(filtered_masks)):
                        raw_candidates.append({
                            'mask': filtered_masks[i],
                            'cls': filtered_classes[i],
                            'area': filtered_areas[i].item() if torch.is_tensor(filtered_areas[i]) else filtered_areas[i],
                            'type': 'fg'
                        })

                    # Calculate Background
                    if len(filtered_masks) > 0:
                        union_mask = torch.any(filtered_masks, dim=0)
                        raw_bg_mask = ~union_mask
                    else:
                        raw_bg_mask = torch.ones((image.shape[0], image.shape[1]), dtype=torch.bool, device=self.device)

                    if raw_bg_mask.sum() > 0:
                         # Connected Components for BG
                         bg_cpu = raw_bg_mask.cpu().numpy().astype(np.uint8)
                         num_labels, labels_im, stats, _ = cv2.connectedComponentsWithStats(bg_cpu, connectivity=4)
                         
                         for lab in range(1, num_labels):
                             # Add valid BG blobs
                             area_comp = stats[lab, cv2.CC_STAT_AREA]
                             if area_comp > min_area_threshold:
                                 m_cpu = (labels_im == lab).astype(np.uint8)
                                 m_gpu = torch.from_numpy(m_cpu).to(self.device).bool()
                                 raw_candidates.append({
                                     'mask': m_gpu,
                                     'cls': torch.tensor(-1, device=self.device),
                                     'area': float(area_comp),
                                     'type': 'bg'
                                 })
                    
                    # -----------------------------------------------------------
                    # 4.3. Assign Base IDs
                    # -----------------------------------------------------------
                    # Sort candidates strictly by area.
                    # We do NOT limit the number of candidates here (raw_candidates),
                    # because we want to allow splitting first. Limiting here might prematurely
                    # drop a large object just because we have many candidates.
                    # (Though sorting by area usually puts small things at the end, it's safer to defer limiting).
                    raw_candidates.sort(key=lambda x: x['area'], reverse=True)
                    raw_candidates = raw_candidates[:99]
                    for idx, item in enumerate(raw_candidates):
                        item['base_id'] = idx + 1
                    # -----------------------------------------------------------
                    # 4.4. Splitting Logic (Recursive / Iterative)
                    # -----------------------------------------------------------
                    # 5% of image area. Using a larger chunk size prevents creating too many fragments
                    # that would exceed MAX_MASKS_COUNTS and result in ID=0 holes.
                    
                    max_single_mask_pixels = image.shape[0] * image.shape[1] * MAX_SINGLE_MASK_RATIO
                    
                    final_split_candidates = []

                    for item in raw_candidates:
                        # Use a stack for recursive splitting (Depth-First) to maintain rough spatial ordering
                        # item keys: 'mask', 'cls', 'area', 'base_id', 'type'
                        item_stack = [item] 
                        
                        # Track assigned offsets for this base_id
                        split_offset_counter = 0

                        while len(item_stack) > 0:
                            curr_item = item_stack.pop() # Pop from end (Stack)
                            curr_area = curr_item['area']
                            if torch.is_tensor(curr_area): curr_area = curr_area.item()
                            
                            # If small enough, finalize
                            if curr_area <= max_single_mask_pixels:
                                curr_item['assigned_id'] = item['base_id'] + (split_offset_counter * 100)
                                curr_item['is_large_split'] = (split_offset_counter > 0)
                                final_split_candidates.append(curr_item)
                                split_offset_counter += 1
                                continue
                                
                            # Need split
                            m = curr_item['mask']
                            x_proj = m.sum(dim=0) > 0 
                            non_zero_cols = torch.where(x_proj)[0]
                            if len(non_zero_cols) == 0: continue

                            min_c = non_zero_cols.min().item()
                            max_c = non_zero_cols.max().item()
                            width = max_c - min_c + 1
                            
                            n_splits = math.ceil(curr_area / max_single_mask_pixels)
                            if n_splits < 2: n_splits = 2
                            
                            step = math.ceil(width / n_splits)
                            if step < 1: step = 1
                            
                            # Check horizontal split feasibility
                            if width < 2 and curr_area > max_single_mask_pixels:
                                # Cannot split horizontally anymore. Force finalize to prevent infinite loop.
                                curr_item['assigned_id'] = item['base_id'] + (split_offset_counter * 100)
                                curr_item['is_large_split'] = (split_offset_counter > 0)
                                final_split_candidates.append(curr_item)
                                split_offset_counter += 1
                                continue
                            
                            # print(f"[PYTHON]    Splitting Huge Mask BaseID {item['base_id']} (area={curr_area:.0f}) into {n_splits} strips")
                            
                            # Generate parts and push to stack in REVERSE order so first part is popped first
                            parts_to_push = []
                            for k in range(n_splits):
                                start = min_c + k * step
                                end = min(min_c + (k + 1) * step, max_c + 1)
                                if start >= max_c + 1: break
                                
                                sub_m = torch.zeros_like(m)
                                sub_m[:, start:end] = m[:, start:end]
                                sub_area = sub_m.sum()
                                
                                if sub_area > 0:
                                    new_part = curr_item.copy()
                                    new_part['mask'] = sub_m
                                    new_part['area'] = sub_area.item()
                                    parts_to_push.append(new_part)
                            
                            # Push to stack in reverse order
                            for p in reversed(parts_to_push):
                                item_stack.append(p)

                    # -----------------------------------------------------------
                    # 4.5. Final Selection
                    # -----------------------------------------------------------
                    # Sort all pieces by area
                    final_split_candidates.sort(key=lambda x: x['area'], reverse=True)
                    
                    # Take top MAX_MASKS_COUNTS
                    # This ensures we always fill the available "slots" with the largest available pieces,
                    # minimizing ID=0 voids.
                    if len(final_split_candidates) > MAX_MASKS_COUNTS:
                        dynamic_max = int(len(final_split_candidates) * RATIO_TO_MAX_MASKS)
                    else:
                        dynamic_max = len(final_split_candidates)
                    
                    extended_masks = final_split_candidates[:dynamic_max]
                    
                    print(f"[PYTHON]  Final Masks: {len(extended_masks)} from {len(raw_candidates)} base objects")
                
                # -----------------------------------------------------------
                # 5. Final Paint & Resize
                # -----------------------------------------------------------
                # Sort to ensure consistent processing order
                extended_masks.sort(key=lambda x: x['assigned_id'])
                
                if postprocess_mode == 1:
                    # Mode 1: Slow / Torch-based accumulation
                    final_mask_tensor = torch.zeros((binary_masks.shape[1], binary_masks.shape[2]), dtype=torch.int32, device=self.device)
                    
                    for item in extended_masks:
                        current_mask = item['mask']
                        obj_id = item['assigned_id']
                            
                        # Connectivity cleanup for small objects (IDs > 10)
                        if enable_postprocess and obj_id > 0:
                            mask_cpu = current_mask.cpu().numpy().astype(np.uint8)
                            num_labels, labels_im, stats, _ = cv2.connectedComponentsWithStats(mask_cpu, connectivity=8)
                            if num_labels > 2:
                                # Keep largest
                                max_area = 0
                                max_label = 1
                                for label_idx in range(1, num_labels):
                                    if stats[label_idx, cv2.CC_STAT_AREA] > max_area:
                                        max_area = stats[label_idx, cv2.CC_STAT_AREA]
                                        max_label = label_idx
                                clean_mask_cpu = (labels_im == max_label)
                                current_mask = torch.from_numpy(clean_mask_cpu).to(self.device)

                        final_mask_tensor[current_mask] = obj_id
                        
                        # Store class name mapping
                        cls_idx = int(item['cls'].item()) if torch.is_tensor(item['cls']) else int(item['cls'])
                        cls_name = "unknown"
                        if cls_idx == -1:
                             cls_name = "background"
                        else:
                             names = self.model.names if hasattr(self.model, 'names') else {}
                             if cls_idx in names:
                                  cls_name = names[cls_idx]
                        
                        if item.get('is_large_split', False):
                            self.instance_map[obj_id] = f"{cls_name}_part"
                        else:
                            self.instance_map[obj_id] = cls_name
                    
                    final_mask = final_mask_tensor.cpu().numpy().astype(np.int32)
                    
                else: 
                    # Mode 2 or 0: Fast CPU Logic
                    mask_h, mask_w = binary_masks.shape[1], binary_masks.shape[2]
                    final_mask_small = np.zeros((mask_h, mask_w), dtype=np.int32)
                    
                    for item in extended_masks:
                        mask_np = item['mask'].cpu().numpy().astype(np.uint8)
                        obj_id = item['assigned_id']
                        
                        if postprocess_mode == 2 and obj_id > 0: 
                             num_labels, labels_im, stats, _ = cv2.connectedComponentsWithStats(mask_np, connectivity=8)
                             if num_labels > 2:
                                 largest_label = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
                                 mask_np = (labels_im == largest_label).astype(np.uint8)
                                
                        final_mask_small[mask_np > 0] = obj_id
                        
                        cls_idx = int(item['cls'].item()) if torch.is_tensor(item['cls']) else int(item['cls'])
                        cls_name = "unknown"
                        if cls_idx == -1:
                             cls_name = "background"
                        else: 
                            if item.get('is_large_split', False):
                                self.instance_map[obj_id] = f"{cls_name}_part"
                            else:
                                self.instance_map[obj_id] = cls_name
                                names = self.model.names if hasattr(self.model, 'names') else {}
                                if cls_idx in names:
                                    cls_name = names[cls_idx]
                        
                        if item.get('is_large_split', False):
                             self.instance_map[obj_id] = f"{cls_name}_part"
                        else:
                             self.instance_map[obj_id] = cls_name
                    
                    # Unpad and Resize if needed
                    if not use_retina_masks:
                        orig_h, orig_w = image.shape[:2]
                        mh, mw = final_mask_small.shape
                        scale = min(mh / orig_h, mw / orig_w)
                        
                        pad_w = (mw - orig_w * scale) / 2
                        pad_h = (mh - orig_h * scale) / 2
                        
                        top = int(round(pad_h - 0.1))
                        bottom = int(round(mh - pad_h + 0.1))
                        left = int(round(pad_w - 0.1))
                        right = int(round(mw - pad_w + 0.1))
                        
                        final_mask_cropped = final_mask_small[top:bottom, left:right]
                        
                        if final_mask_cropped.shape[0] != orig_h or final_mask_cropped.shape[1] != orig_w:
                            final_mask = cv2.resize(final_mask_cropped, (orig_w, orig_h), interpolation=cv2.INTER_NEAREST)
                        else:
                            final_mask = final_mask_cropped
                    else:
                        final_mask = final_mask_small

            else:
                final_mask = np.zeros(image.shape[:2], dtype=np.int32)

            t_post_end = time.time()            
            print(f"[PYTHON]  Time Stats: Inference={t_infer_end - t_infer_start:.4f}s, Post-process={t_post_end - t_post_start:.4f}s, Total={t_post_end - t_start:.4f}s")
            return final_mask

        except Exception as e:
            print(f"[PYTHON]  Error during segmentation: {e}")
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

    def get_visualization_from_bytes(self, image_bytes, mask_bytes, h, w):
        """
        Args:
            image_bytes: bytes object containing encoded image (jpg/png)
            mask_bytes: bytes object containing raw int32 mask data
            h, w: dimensions of the mask/image
        Returns:
            bytes: encoded visualization image (jpg)
        """
        nparr_img = np.frombuffer(image_bytes, np.uint8)
        image = cv2.imdecode(nparr_img, cv2.IMREAD_COLOR)
        
        nparr_mask = np.frombuffer(mask_bytes, np.int32)
        if nparr_mask.size != h * w:
            print(f"[PYTHON] Error: Mask size mismatch. Expected {h*w}, got {nparr_mask.size}")
            return b""
        mask = nparr_mask.reshape((h, w))
        
        vis = self.get_visualization(image, mask)
        success, encoded_vis = cv2.imencode(".jpg", vis)
        if success:
            return encoded_vis.tobytes()
        return b""

    def get_visualization(self, image, mask):
        """
        Generate a visualization of the segmentation mask overlaid on the image.
        Args:
            image: numpy array (H, W, 3) BGR
            mask: numpy array (H, W) int32
        Returns:
            vis_img: numpy array (H, W, 3) BGR
        """
        unique_ids = np.unique(mask)
        max_id = np.max(unique_ids) if len(unique_ids) > 0 else 0
        
        h, w = mask.shape
        vis_mask = np.zeros((h, w, 3), dtype=np.uint8)
        
        # Simple random coloring
        if max_id > 0:
            # Seed random generator for consistent colors for same IDs (optional)
            np.random.seed(42) 
            colors = np.random.randint(0, 255, (max_id + 1, 3), dtype=np.uint8)
            colors[0] = [0, 0, 0] # Background
            vis_mask = colors[mask]

        alpha = 0.3
        # Ensure sizes match
        if image.shape[:2] != mask.shape:
             image = cv2.resize(image, (w, h))
             
        overlay = cv2.addWeighted(image, 1 - alpha, vis_mask, alpha, 0)

        # Text labels (only for DL models usually, SLIC has too many)
        if hasattr(self, 'model_type') and self.model_type != "slic":
            for uid in unique_ids:
                # if uid == 0: continue
                # Performance optimization: calculating mean of indices can be slow for large masks
                # We can approximate or just do it. numpy where is fast enough for typical image sizes.
                y_indices, x_indices = np.where(mask == uid)
                if len(y_indices) > 0:
                    cy = int(np.mean(y_indices))
                    cx = int(np.mean(x_indices))
                    text = f"{uid}" 
                    if cx < w and cy < h:
                         cv2.putText(overlay, text, (cx, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        return overlay

    def get_single_id_visualization(self, image, mask, target_id=0):
        """
        Generate a visualization highlighting only a specific mask ID (e.g., 0 for unassigned).
        Args:
            image: numpy array (H, W, 3) BGR
            mask: numpy array (H, W) int32
            target_id: int, the ID to highlight
        Returns:
            vis_img: numpy array (H, W, 3) BGR
        """
        h, w = mask.shape
        # Create mask for the target ID
        target_mask = (mask == target_id).astype(np.uint8)
        
        # Create a visualization where target is Red, others are Grayscale
        vis_img = image.copy()
        
        # 1. Convert whole image to grayscale for contrast
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        gray_bgr = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
        
        # 2. Define highlight color (Red)
        highlight_color = np.array([0, 0, 255], dtype=np.uint8) # BGR
        
        # 3. Create overlay
        # Alpha blend: 0.5 * original_color + 0.5 * red
        # Actually, let's make it very obvious. 
        # Background (non-target): Grayscale
        # Target: Original Color mixed with Red? Or just Red?
        # Let's do: Target = Red overlay on Grayscale.
        
        # Base is grayscale
        vis_img = gray_bgr 
        
        # Where mask is target
        roi = (target_mask > 0)
        
        # Create colored ROI
        colored_roi = np.zeros_like(image)
        colored_roi[:] = highlight_color
        
        # Blend in the ROI
        alpha = 0.4
        if roi.any():
             # Part of image that is target
             img_part = image[roi]
             color_part = colored_roi[roi]
             blended = cv2.addWeighted(img_part, 1-alpha, color_part, alpha, 0)
             vis_img[roi] = blended
             
        # Add text
        text = f"ID={target_id} Count={np.sum(target_mask)}"
        cv2.putText(vis_img, text, (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
        
        return vis_img

# Helper function to create instance
def create_segmentator():
    return SegmentationWrapper()
