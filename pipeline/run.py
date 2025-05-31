import cv2
import numpy as np
from ultralytics import YOLO
import os
from pathlib import Path
import argparse
import torch
import traceback
import sys
import torchvision.transforms.functional as F
import torch.nn.functional as TF           # add along with the other imports

try:
    from cutie.model.cutie import CUTIE
    from cutie.inference.inference_core import InferenceCore
    from cutie.utils.get_default_model import get_default_model
    CUTIE_AVAILABLE = True
    print("Cutie available")
except ImportError as e:
    print(e)
    traceback.print_exc()
    print("Warning: Cutie not available. Install Cutie for tracking functionality.")
    CUTIE_AVAILABLE = False

import gc, contextlib

def gpu_mb():
    return torch.cuda.memory_allocated() / 1024**2

def show_delta(load_fn, name="model"):
    torch.cuda.empty_cache(); gc.collect()
    start = gpu_mb()
    obj = load_fn().to("cuda")          # ← load_fn returns the model
    torch.cuda.synchronize()
    print(f"{name:15s}: +{gpu_mb()-start:8.1f} MB parameters")
    return obj                           # keep it or del obj to free

# CUTIE_AVAILABLE = False
class YOLOCutiePipeline:
    def __init__(self, detection_model_path, segmentation_model_path, cutie_model_path=None, confidence_threshold=0.5, show_boxes=False, low_mem=False):
        """
        Initialize the YOLO-Cutie hybrid pipeline.
        
        Args:
            detection_model_path (str): Path to the fine-tuned YOLO detection model
            segmentation_model_path (str): Path to the YOLO segmentation model
            cutie_model_path (str): Path to the Cutie model (optional)
            confidence_threshold (float): Threshold for detection confidence
        """
        self.detection_model = show_delta(lambda: YOLO(detection_model_path), "detection")
        self.segmentation_model = show_delta(lambda: YOLO(segmentation_model_path), "segmentation")
        self.confidence_threshold = confidence_threshold
        self.show_boxes = show_boxes
        self.low_mem = low_mem
        
        # Cutie tracking setup
        self.cutie_model = None
        self.cutie_processor = None
        if CUTIE_AVAILABLE:
            # self.cutie_model = self._load_cutie_model(cutie_model_path)
            # if self.low_mem:
            #     self.cutie_model = self.cutie_model.half()

            self.cutie_model = self._load_cutie_model(cutie_model_path)
            if self.low_mem and self.cutie_model is not None:
                self.cutie_model = self.cutie_model.half()  # FP16 weights
                self.cutie_model.eval()                     # inference-only
        
        # Tracking state
        self.is_tracking = False
        self.last_instance_count = 0
        self.tracking_masks = None
        self.frame_idx = 0

    def _load_cutie_model(self, model_path):
        """Load and initialize the Cutie model."""
        try:
            # Adjust this based on your Cutie model loading requirements
            # model = CUTIE.from_pretrained(model_path)
            # model.eval()
            # return model
            cutie = show_delta(lambda: get_default_model(), "Cutie")
            return cutie
        except Exception as e:
            print(f"Error loading Cutie model: {e}")
            return None
    
    def _draw_detection_boxes(self, frame, yolo_results):
        """Overlay YOLO bounding boxes (in-place)."""
        for r in yolo_results:
            if r.boxes is None:
                continue
            for b in r.boxes:
                x1, y1, x2, y2 = map(int, b.xyxy[0])
                conf = float(b.conf[0])
                cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 255, 0), 2)
                cv2.putText(frame, f"{conf:.2f}", (x1, y1 - 6),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)


    # helper: centralise resize + dtype choice
    # def _cv2_to_tensor(self, frame_bgr):
    #     # ↓ optional resize for 720-long-side
    #     if self.low_mem:
    #         h, w = frame_bgr.shape[:2]
    #         scale = 720.0 / max(h, w)
    #         if scale < 1.0:
    #             frame_bgr = cv2.resize(frame_bgr,
    #                                 (int(w * scale), int(h * scale)),
    #                                 interpolation=cv2.INTER_AREA)

    #     # ↓ pad H, W to multiples of 32
    #     h, w = frame_bgr.shape[:2]
    #     pad_h, pad_w = (-h) % 32, (-w) % 32
    #     if pad_h or pad_w:
    #         frame_bgr = cv2.copyMakeBorder(frame_bgr, 0, pad_h, 0, pad_w,
    #                                     cv2.BORDER_CONSTANT, value=0)

    #     # ↓ BGR→RGB → tensor (0-1) → send to same device *and* dtype as model
    #     frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    #     tensor = F.to_tensor(frame_rgb)                          # FP32 here

    #     if self.cutie_model is not None:                         # always true in tracking
    #         p = next(self.cutie_model.parameters())
    #         tensor = tensor.to(device=p.device, dtype=p.dtype)   # ← key line

    #     return tensor
    def _cv2_to_tensor(self, frame_bgr):
        # optional low‑mem resize (long side ≤ 720 px)
        if self.low_mem:
            h, w = frame_bgr.shape[:2]
            scale = 720.0 / max(h, w)
            if scale < 1.0:
                frame_bgr = cv2.resize(
                    frame_bgr, (int(w * scale), int(h * scale)),
                    interpolation=cv2.INTER_AREA)

        # pad H, W to multiples of 32 (Cutie & YOLO assumption)

        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        tensor = F.to_tensor(frame_rgb)                 # FP32

        # ↓↓↓  add THIS two-liner  ↓↓↓
        if self.low_mem:                                # --low_mem implies FP16
            tensor = tensor.half()
        return tensor.to(self.cutie_model.device)




    # ---------------------------------------------------------------------------

    def _initialize_cutie_tracking(self, frame, masks):
        """Bootstrap Cutie on the first detection frame."""
        if not (CUTIE_AVAILABLE and self.cutie_model):
            return False

        try:
            self.cutie_processor = InferenceCore(self.cutie_model,
                                                cfg=self.cutie_model.cfg)
            self.cutie_processor.max_internal_size = 240    # tweak if you like

            # 1️⃣  build an instance-label mask (H₀×W₀, int)
            combined_mask = self._prepare_masks_for_cutie(masks).squeeze(0).cpu().numpy()
            objects       = np.unique(combined_mask)
            objects       = objects[objects > 0].tolist()

            # 2️⃣  make the frame tensor – this already handles scale/pad
            frame_t       = self._cv2_to_tensor(frame)           # C×H×W
            H, W          = frame_t.shape[-2:]                   # 416 × 736 in low-mem

            # 3️⃣  resize mask → (H×W) w/ nearest-neighbour to keep IDs intact
            combined_mask = cv2.resize(combined_mask,
                                    (W, H),
                                    interpolation=cv2.INTER_NEAREST)

            # 4️⃣  pad mask exactly like the image (it is already correct in W/H but
            #     if you ever change padding logic keep this in sync)
            combined_mask = torch.from_numpy(combined_mask).to(frame_t.device)

            # print("[DEBUG] frame_t:", frame_t.dtype, frame_t.device,
            #     "| model:", next(self.cutie_model.parameters()).dtype,
            #     next(self.cutie_model.parameters()).device)

            # 5️⃣  seed Cutie’s memory bank
            self.cutie_processor.step(frame_t,
                                    combined_mask,
                                    objects=objects)


            self.is_tracking = True
            self.frame_idx = 0
            self.last_instance_count = len(objects)
            return True

        except Exception:
            traceback.print_exc()
            print("[Cutie] failed to initialise — falling back to detection mode")
            return False


    def _track_frame(self, frame):
        """Track an already initialised sequence."""
        if not (self.is_tracking and CUTIE_AVAILABLE and self.cutie_processor):
            return None, 0

        try:
            frame_t = self._cv2_to_tensor(frame)
            with torch.inference_mode():                          # saves VRAM
                output_prob = self.cutie_processor.step(frame_t)  # 🌟 one‑liner    
                mask_int   = self.cutie_processor.output_prob_to_mask(output_prob)
                masks, n   = self._process_cutie_output(mask_int)
                self.frame_idx += 1
                return masks, n

        except Exception:
            traceback.print_exc()
            self.is_tracking = False
            return None, 0
    
    def _prepare_masks_for_cutie(self, yolo_masks):
        """Convert YOLO segmentation masks to Cutie format."""
        # This function needs to be implemented based on your specific
        # YOLO and Cutie mask formats
        if len(yolo_masks) == 0:
            return None
            
        # Combine all masks into a single tensor with instance IDs
        combined_mask = np.zeros(yolo_masks[0].shape, dtype=np.uint8)
        for i, mask in enumerate(yolo_masks):
            combined_mask[mask > 0.5] = i + 1  # Instance IDs start from 1
            
        return torch.from_numpy(combined_mask).unsqueeze(0)
    
    def _reset_cutie(self):
        if self.cutie_processor is not None:
            if hasattr(self.cutie_processor, "clear_memory"):
                self.cutie_processor.clear_memory()  # drop feature bank
            del self.cutie_processor
            torch.cuda.empty_cache()
        self.cutie_processor = None
        self.is_tracking = False
        self.last_instance_count = 0


    def _process_cutie_output(self, cutie_mask):
        """Process Cutie output mask and count instances."""
        if cutie_mask is None:
            return [], 0
            
        # Convert tensor to numpy if needed
        if torch.is_tensor(cutie_mask):
            mask_np = cutie_mask.squeeze().cpu().numpy()
        else:
            mask_np = cutie_mask
            
        # Count unique instances (excluding background=0)
        unique_ids = np.unique(mask_np)
        unique_ids = unique_ids[unique_ids > 0]  # Remove background
        instance_count = len(unique_ids)
        
        # Convert back to individual masks
        individual_masks = []
        for instance_id in unique_ids:
            individual_mask = (mask_np == instance_id).astype(np.float32)
            individual_masks.append(individual_mask)
            
        return individual_masks, instance_count
    
    def process_video(self, video_path, output_dir="out", save_original=False, save_all_frames=False, 
                     output_video=True):
        """
        Process video frames with hybrid detection-tracking pipeline.
        
        Args:
            video_path (str): Path to input video
            output_dir (str): Directory to save output images
            save_original (bool): Whether to save original frame alongside segmented result
            save_all_frames (bool): Whether to save all tracked frames or just detection frames
            output_video (bool): Whether to generate output video
            output_video_path (str): Path for output video (auto-generated if None)
        
        Returns:
            tuple: (list of saved image paths, output video path if created)
        """
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Create frames subdirectory if save_all_frames is enabled
        frames_dir = None
        if save_all_frames:
            frames_dir = os.path.join(output_dir, "frames")
            os.makedirs(frames_dir, exist_ok=True)
        
        # Open video
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Could not open video: {video_path}")
        
        frame_count = 0
        saved_images = []
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # Video properties for output
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        # Setup video writer if needed
        video_writer = None
        output_video_full_path = None
        if output_video:
            video_name = Path(video_path).stem
            
            # Try different codecs and formats in order of preference
            codecs_to_try = [
                # (fourcc, extension, description)
                ('H264', '.mp4', 'H.264'),
                ('avc1', '.mp4', 'AVC1'),
                ('mp4v', '.mp4', 'MP4V'),
                ('XVID', '.avi', 'XVID'),
                ('MJPG', '.avi', 'Motion JPEG'),
                ('X264', '.mp4', 'X264'),
                (-1, '.avi', 'Default codec')  # -1 lets OpenCV choose
            ]
            
            video_writer = None
            for fourcc_code, ext, desc in codecs_to_try:
                try:
                    test_path = os.path.join(output_dir, f"{video_name}_processed{ext}")
                    
                    if fourcc_code == -1:
                        fourcc = -1
                    else:
                        fourcc = cv2.VideoWriter_fourcc(*fourcc_code)
                    
                    # Ensure dimensions are even numbers (required by many codecs)
                    adj_width = width if width % 2 == 0 else width - 1
                    adj_height = height if height % 2 == 0 else height - 1
                    
                    video_writer = cv2.VideoWriter(test_path, fourcc, fps, (adj_width, adj_height))
                    
                    # Test if the writer was initialized successfully
                    if video_writer.isOpened():
                        output_video_full_path = test_path
                        width, height = adj_width, adj_height  # Update dimensions
                        print(f"Using codec: {desc} ({fourcc_code}), format: {ext}")
                        print(f"Video dimensions adjusted to: {width}x{height}")
                        break
                    else:
                        video_writer.release()
                        video_writer = None
                        
                except Exception as e:
                    print(f"Failed to initialize codec {desc}: {e}")
                    if video_writer:
                        video_writer.release()
                        video_writer = None
            
            if video_writer is None:
                print("Warning: Could not initialize video writer with any codec. Video output disabled.")
                output_video = False
        
        print(f"Processing video: {video_path}")
        print(f"Total frames: {total_frames}")
        print(f"Video properties: {width}x{height} @ {fps:.2f} FPS")
        print(f"Detection threshold: {self.confidence_threshold}")
        print(f"Cutie tracking: {'Enabled' if CUTIE_AVAILABLE and self.cutie_model else 'Disabled'}")
        print(f"Save all frames: {'Enabled' if save_all_frames else 'Disabled'}")
        if save_all_frames:
            print(f"Frames directory: {frames_dir}")
        if output_video:
            print(f"Output video: {output_video_full_path}")
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                    
                frame_count += 1
                segmented_image = None
                current_confidence = 0.0
                source_type = "none"
                display_frame = frame.copy()  # Frame to write to video

                if self.show_boxes:
                    self._draw_detection_boxes(display_frame, detection_results)
                
                if not self.is_tracking:
                    # Detection mode: Look for objects to start tracking
                    detection_results = self.detection_model(frame, verbose=False)
                    max_confidence = self._get_max_confidence(detection_results)
                    current_confidence = max_confidence
                    
                    if max_confidence >= self.confidence_threshold:
                        print(f"Frame {frame_count}: Detection triggered (conf: {max_confidence:.3f})")
                        
                        # Run segmentation
                        segmentation_results = self.segmentation_model(frame, verbose=False)
                        masks = self._extract_masks_from_segmentation(segmentation_results)
                        
                        if masks:
                            # Create segmented image
                            segmented_image = self._create_segmented_image(frame, masks)
                            display_frame = segmented_image.copy()
                            self.last_instance_count = len(masks)
                            source_type = "detection+segmentation"
                            
                            # Initialize tracking if Cutie is available
                            if CUTIE_AVAILABLE and self.cutie_model:
                                success = self._initialize_cutie_tracking(frame, masks)
                                if success:
                                    print(f"  -> Tracking initialized with {len(masks)} instances")
                                else:
                                    print("  -> Failed to initialize tracking, continuing with detection mode")
                        
                else:
                    # Tracking mode: Use Cutie to track existing objects
                    tracked_masks, instance_count = self._track_frame(frame)
                    
                    if instance_count == 0:
                        # Lost all tracks - switch back to detection mode
                        print(f"Frame {frame_count}: All tracks lost, switching to detection mode")
                        self.is_tracking = False
                        self.last_instance_count = 0
                        self._reset_cutie()
                        
                    elif instance_count != self.last_instance_count:
                        # Instance count changed - re-run segmentation and restart tracking
                        print(f"Frame {frame_count}: Instance count changed ({self.last_instance_count} -> {instance_count})")
                        print("  -> Re-running segmentation")
                        
                        segmentation_results = self.segmentation_model(frame, verbose=False)
                        masks = self._extract_masks_from_segmentation(segmentation_results)
                        
                        if masks:
                            segmented_image = self._create_segmented_image(frame, masks)
                            display_frame = segmented_image.copy()
                            self.last_instance_count = len(masks)
                            source_type = "re-segmentation"
                            
                            # Restart tracking with new masks
                            success = self._initialize_cutie_tracking(frame, masks)
                            if success:
                                print(f"  -> Tracking restarted with {len(masks)} instances")
                            else:
                                print("  -> Failed to restart tracking")
                        else:
                            # No masks found, switch to detection mode
                            self.is_tracking = False
                            self.last_instance_count = 0
                            
                    else:
                        # Normal tracking - use tracked masks
                        if tracked_masks:
                            segmented_image = self._create_segmented_image_from_masks(frame, tracked_masks)
                            display_frame = segmented_image.copy()
                            source_type = "tracking"
                
                # Add status overlay to the frame
                display_frame = self._add_status_overlay(display_frame, frame_count, source_type, 
                                                       current_confidence, self.last_instance_count)
                
                # Write frame to output video
                if video_writer is not None and video_writer.isOpened():
                    # Ensure frame dimensions match expected output
                    if display_frame.shape[1] != width or display_frame.shape[0] != height:
                        display_frame = cv2.resize(display_frame, (width, height))
                    
                    # Ensure the frame is in the correct format (BGR, uint8)
                    if display_frame.dtype != np.uint8:
                        display_frame = display_frame.astype(np.uint8)
                    
                    # success = video_writer.write(display_frame)
                    # if not success and frame_count == 1:
                    #     print("Warning: Failed to write first frame. Video may be corrupted.")
                    video_writer.write(display_frame)
                
                # Save individual frames only if save_all_frames is True
                if save_all_frames and segmented_image is not None:
                    video_name = Path(video_path).stem
                    output_filename = f"{video_name}_frame_{frame_count:06d}_{source_type}_conf_{current_confidence:.3f}.jpg"
                    output_path = os.path.join(frames_dir, output_filename)
                    
                    cv2.imwrite(output_path, segmented_image)
                    saved_images.append(output_path)
                    
                    # Optionally save original frame
                    if save_original:
                        original_filename = f"original_{output_filename}"
                        original_path = os.path.join(frames_dir, original_filename)
                        cv2.imwrite(original_path, frame)
                    
                    print(f"  -> Saved: {output_filename}")
                
                # Progress indicator
                if frame_count % 100 == 0:
                    mode = "tracking" if self.is_tracking else "detection"
                    print(f"Processed {frame_count}/{total_frames} frames (mode: {mode})")
        
        finally:
            cap.release()
            if video_writer is not None:
                video_writer.release()
                
                # Wait a moment for file to be fully written
                import time
                time.sleep(0.5)
                
                if output_video_full_path and os.path.exists(output_video_full_path):
                    file_size = os.path.getsize(output_video_full_path)
                    if file_size > 1000:  # At least 1KB
                        print(f"Output video saved: {output_video_full_path}")
                        print(f"Video file size: {file_size / (1024*1024):.2f} MB")
                        
                        # Try to verify the video is readable
                        test_cap = cv2.VideoCapture(output_video_full_path)
                        if test_cap.isOpened():
                            test_frame_count = int(test_cap.get(cv2.CAP_PROP_FRAME_COUNT))
                            test_cap.release()
                            print(f"Video verification: {test_frame_count} frames detected")
                            if test_frame_count == 0:
                                print("Warning: Video appears to have 0 frames. May be corrupted.")
                        else:
                            print("Warning: Created video file cannot be opened by OpenCV.")
                    else:
                        print(f"Warning: Video file seems too small ({file_size} bytes). May be corrupted.")
                else:
                    print("Warning: Video file was not created successfully.")
        
        if save_all_frames:
            print(f"\nProcessing complete! Saved {len(saved_images)} segmented images to {frames_dir}")
        else:
            print(f"\nProcessing complete! Frame saving was disabled (save_all_frames=False)")
        
        if output_video:
            return saved_images, output_video_full_path
        else:
            return saved_images, None
    
    def _extract_masks_from_segmentation(self, segmentation_results):
        """Extract individual masks from YOLO segmentation results."""
        masks = []
        for result in segmentation_results:
            if result.masks is not None:
                mask_data = result.masks.data.cpu().numpy()
                for mask in mask_data:
                    masks.append(mask)
        return masks
    
    def _create_segmented_image_from_masks(self, original_frame, masks):
        """Create segmented image from a list of mask arrays."""
        result_image = original_frame.copy()
        
        if not masks:
            return result_image
            
        colors = self._generate_colors(len(masks))
        
        for i, mask in enumerate(masks):
            # Resize mask to match frame dimensions if needed
            if mask.shape[:2] != original_frame.shape[:2]:
                mask_resized = cv2.resize(mask, (original_frame.shape[1], original_frame.shape[0]))
            else:
                mask_resized = mask
            
            # Create colored mask
            colored_mask = np.zeros_like(original_frame)
            colored_mask[mask_resized > 0.5] = colors[i]
            
            # Blend with original image
            alpha = 0.6
            result_image = cv2.addWeighted(result_image, 1, colored_mask, alpha, 0)
            
            # Add contours for better visibility
            contours, _ = cv2.findContours(
                (mask_resized > 0.5).astype(np.uint8), 
                cv2.RETR_EXTERNAL, 
                cv2.CHAIN_APPROX_SIMPLE
            )
            cv2.drawContours(result_image, contours, -1, colors[i], 2)
        
        return result_image
    
    def _get_max_confidence(self, detection_results):
        """Extract maximum confidence from detection results."""
        max_conf = 0.0
        for result in detection_results:
            if result.boxes is not None and len(result.boxes) > 0:
                confidences = result.boxes.conf.cpu().numpy()
                max_conf = max(max_conf, np.max(confidences))
        return max_conf
    
    def _create_segmented_image(self, original_frame, masks):
        """Create segmented image from mask list (compatibility method)."""
        return self._create_segmented_image_from_masks(original_frame, masks)
    
    def _add_status_overlay(self, frame, frame_num, source_type, confidence, instance_count):
        """Add status information overlay to the frame."""
        overlay_frame = frame.copy()
        
        # Create semi-transparent overlay area
        overlay = overlay_frame.copy()
        cv2.rectangle(overlay, (10, 10), (400, 120), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.7, overlay_frame, 0.3, 0, overlay_frame)
        
        # Add text information
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.6
        color = (255, 255, 255)
        thickness = 1
        
        # Frame number
        cv2.putText(overlay_frame, f"Frame: {frame_num}", (20, 35), 
                   font, font_scale, color, thickness)
        
        # Mode and source
        # mode = "TRACKING" if source_type == "tracking" else "DETECTION"
        mode = {
            "tracking":          "TRACKING",
            "detection+segmentation": "DETECTION",
            "re-segmentation":   "RE-SEGMENT"
        }.get(source_type, "DETECTION")

        mode_color = (0, 255, 0) if source_type == "tracking" else (0, 255, 255)
        cv2.putText(overlay_frame, f"Mode: {mode}", (20, 55), 
                   font, font_scale, mode_color, thickness)
        
        # Source type
        cv2.putText(overlay_frame, f"Source: {source_type}", (20, 75), 
                   font, font_scale, color, thickness)
        
        # Confidence (if available)
        if confidence > 0:
            cv2.putText(overlay_frame, f"Confidence: {confidence:.3f}", (20, 95), 
                       font, font_scale, color, thickness)
        
        # Instance count
        cv2.putText(overlay_frame, f"Instances: {instance_count}", (200, 35), 
                   font, font_scale, color, thickness)
        
        return overlay_frame
    
    def _generate_colors(self, num_colors):
        """Generate distinct colors for different mask instances."""
        colors = []
        for i in range(num_colors):
            hue = i * 180 // num_colors
            color = cv2.cvtColor(np.uint8([[[hue, 255, 255]]]), cv2.COLOR_HSV2BGR)[0][0]
            colors.append([int(c) for c in color])
        return colors


def main():
    parser = argparse.ArgumentParser(description='YOLO-Cutie Hybrid Pipeline')
    parser.add_argument('--detection_model', required=True, help='Path to detection model')
    parser.add_argument('--segmentation_model', required=True, help='Path to segmentation model')
    parser.add_argument('--cutie_model', help='Path to Cutie tracking model')
    parser.add_argument('--video', required=True, help='Path to input video')
    parser.add_argument('--output_dir', default='out', help='Output directory')
    parser.add_argument('--threshold', type=float, default=0.5, help='Detection confidence threshold')
    parser.add_argument('--save_all_frames', action='store_true', default=False, help='Save all frames including tracked ones')
    parser.add_argument('--output_video', action='store_true', default=True, help='Generate output video')
    parser.add_argument('--show_boxes', action='store_true', help='Draw YOLO detection boxes (default: off)')# CLI
    parser.add_argument('--low_mem', action='store_true',
                        help='Run Cutie in half-precision at 720 p (uses <8 GB VRAM)')

    
    args = parser.parse_args()
    
    # Initialize pipeline
    pipeline = YOLOCutiePipeline(
        detection_model_path=args.detection_model,
        segmentation_model_path=args.segmentation_model,
        cutie_model_path=args.cutie_model,
        confidence_threshold=args.threshold,
        show_boxes=args.show_boxes,
        low_mem=args.low_mem
    )
    
    # Process video
    results = pipeline.process_video(
        video_path=args.video,
        output_dir=args.output_dir,
        save_all_frames=args.save_all_frames,
        output_video=args.output_video,
    )
    
    if isinstance(results, tuple):
        saved_images, output_video_path = results
        if output_video_path:
            print(f"\nPipeline complete! Generated output video: {output_video_path}")
    else:
        saved_images = results

    if saved_images:
        print(f"\nSaved {len(saved_images)} segmented frames.")
    print(f"\nPipeline complete!")


if __name__ == "__main__":
    main()


# Example usage:
"""
# Initialize the hybrid pipeline
pipeline = YOLOCutiePipeline(
    detection_model_path="path/to/detection_model.pt",
    segmentation_model_path="path/to/segmentation_model.pt",
    cutie_model_path="path/to/cutie_model.pth",
    confidence_threshold=0.7
)

# Process a video with tracking
saved_images = pipeline.process_video(
    video_path="input_video.mp4",
    output_dir="hybrid_outputs",
    save_original=True,
    save_all_frames=True  # Now frames will be saved to hybrid_outputs/frames/
)
"""

"""

python run.py --detection_model ../train/runs/detect/medium/weights/best.pt --segmentation_model ../train/runs/segment/initial/weights/best.pt --video inputs/bloomberg.mp4 --threshold 0.5 --output_dir out/bloomberg --save_all_frames

# Cutie
python run.py --detection_model ../train/runs/detect/medium/weights/best.pt \
            --segmentation_model ../train/runs/segment/initial/weights/best.pt \
            --cutie_model ../../Cutie/weights/cutie-base-mega.pth \
            --video inputs/bloomberg.mp4 \
            --threshold 0.3 \
            --output_dir out/bloomberg \
            --low_mem

"""