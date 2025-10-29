#!/usr/bin/env python3
"""
MMPose Keypoint Detection Script

This script performs 2D human pose estimation using MMPose with ViTPose models.
It processes videos frame by frame, using pre-computed bounding boxes to detect
and extract keypoint coordinates for multiple individuals.

Usage:
    python mmpose_inference.py POSE_CONFIG POSE_CHECKPOINT [options]

Example:
    python mmpose_inference.py \
        configs/body_2d_keypoint/topdown_heatmap/coco/td-hm_ViTPose-huge_8xb64-210e_coco-256x192.py \
        td-hm_ViTPose-huge_8xb64-210e_coco-256x192-e32adcd4_20230314.pth \
        --input video.mp4 \
        --bboxes bboxes.json \
        --output-root ./output \
        --save-predictions

Requirements:
    - MMPose with custom modification (see README)
    - ViTPose model weights
    - CUDA-capable GPU (recommended)
"""

import logging
import mimetypes
import os
import sys
import time

import cv2
import tqdm
import json_tricks as json
import mmcv
import mmengine
import numpy as np
import argparse

import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from mmengine.logging import print_log

from mmpose.apis import inference_topdown
from mmpose.apis import init_model as init_pose_estimator
from mmpose.evaluation.functional import nms
from mmpose.registry import VISUALIZERS
from mmpose.structures import merge_data_samples, split_instances
from mmpose.utils import adapt_mmdet_pipeline

try:
    from mmdet.apis import inference_detector, init_detector
    has_mmdet = True
except (ImportError, ModuleNotFoundError):
    has_mmdet = False

mmengine.registry.init_default_scope('mmpose')



class MMPoseInference:
    """
    MMPose inference class for 2D human pose estimation.
    
    This class handles video processing, bounding box mapping, keypoint detection,
    and result visualization using MMPose framework with ViTPose models.
    """

    def __init__(self, pose_config: str, pose_checkpoint: str, det_config: str = None, det_checkpoint: str = None, device: str = 'cuda:0', mode: str = 'mmdet'):
        """
        Initialize the MMPose inference system.
        
        Args:
            pose_config (str): Path to MMPose configuration file
            pose_checkpoint (str): Path to model checkpoint file
            det_config (str): Path to detection model config file (optional)
            det_checkpoint (str): Path to detection model checkpoint file (optional)
            device (str): Device to use for inference ('cuda:0', 'cpu', etc.)
            mode (str): Mode of operation ('mmdet' for detection model, 'bbox' for pre-computed bboxes)
        """
        self.pose_config = pose_config
        self.pose_checkpoint = pose_checkpoint
        self.det_config = det_config
        self.det_checkpoint = det_checkpoint
        self.device = device
        self.mode = mode
        
        # Model components
        self.pose_estimator = None
        self.visualizer = None
        self.detector = None
        
        # Processing data
        self.bboxes_data = None
        self.pred_instances_list = []
        
        print_log("Initializing MMPose inference system...")
        self._validate_inputs()
        if mode == 'mmdet':
            self._init_detector()

        self._init_pose_estimator()


    
    def _validate_inputs(self):
        """Validate input files and configurations."""
        if not os.path.exists(self.pose_config):
            raise FileNotFoundError(f"Pose config file not found: {self.pose_config}")
        
        if not os.path.exists(self.pose_checkpoint):
            raise FileNotFoundError(f"Pose checkpoint file not found: {self.pose_checkpoint}")
        
        print_log(f"Using pose config: {self.pose_config}")
        print_log(f"Using pose checkpoint: {self.pose_checkpoint}")
        print_log(f"Using device: {self.device}")
    
    def _init_pose_estimator(self):
        """Initialize the pose estimation model."""
        print_log("Loading pose estimation model...")
        
        try:
            self.pose_estimator = init_pose_estimator(
                self.pose_config,
                self.pose_checkpoint,
                device=self.device,
                cfg_options=dict(
                    model=dict(test_cfg=dict(output_heatmaps=False))
                )
            )
            print_log("Pose estimator loaded successfully")
            
        except Exception as e:
            raise RuntimeError(f"Failed to initialize pose estimator: {e}")
        
    def _init_detector(self):
        """Initialize the detection model if provided."""
        if self.det_config and self.det_checkpoint:
            if not has_mmdet:
                raise ImportError("mmdet is not installed. Please install mmdet to use detection model.")
            
            print_log("Loading detection model...")
            try:
                self.detector = init_detector(
                    self.det_config,
                    self.det_checkpoint,
                    device=self.device,
                )
                self.detector.cfg = adapt_mmdet_pipeline(self.detector.cfg)
                print_log("Detection model loaded successfully")
            except Exception as e:
                raise RuntimeError(f"Failed to initialize detection model: {e}")
    
    def setup_visualizer(self, args):
        """
        Setup the visualization components.
        
        Args:
            args: Command line arguments containing visualization parameters
        """
        print_log("Setting up visualizer...")
        
        # Clean any existing visualizer first
        # self.clean_visualizer()
        
        # Configure visualizer parameters
        self.pose_estimator.cfg.visualizer.radius = args.radius
        self.pose_estimator.cfg.visualizer.alpha = args.alpha
        self.pose_estimator.cfg.visualizer.line_width = args.thickness

        self.visualizer = VISUALIZERS.build(self.pose_estimator.cfg.visualizer)
        self.visualizer.set_dataset_meta(
            self.pose_estimator.dataset_meta, skeleton_style=args.skeleton_style)
        
        print_log(f"Visualizer configured with skeleton style: {args.skeleton_style}")
    
    # def clean_visualizer(self):
    #     """
    #     Clean and reset the visualizer to prevent state accumulation between runs.
    #     """
    #     if self.visualizer is not None:
    #         print_log("Cleaning existing visualizer...")
            
    #         # Clear any accumulated drawings or state
    #         if hasattr(self.visualizer, 'clean_image'):
    #             self.visualizer.clean_image()
            
    #         # Reset the image buffer if it exists
    #         if hasattr(self.visualizer, '_image'):
    #             self.visualizer._image = None
            
    #         # Clear any drawing state
    #         if hasattr(self.visualizer, '_drawn_data'):
    #             self.visualizer._drawn_data = []
            
    #         # Reset any accumulated visualization data
    #         if hasattr(self.visualizer, '_vis_buffer'):
    #             self.visualizer._vis_buffer = None
                
    #         print_log("Visualizer cleaned successfully")
        
    #     # Set to None to ensure fresh creation
    #     self.visualizer = None
    
    # def cleanup(self):
    #     """
    #     Cleanup resources and reset the inference system for the next run.
    #     Call this method between multiple inference runs.
    #     """
    #     print_log("Cleaning up inference system...")
        
    #     # Clean visualizer
    #     self.clean_visualizer()
        
    #     # Clear any prediction data
    #     self.pred_instances_list = []
        
    #     # Clear bounding box data if it exists
    #     if hasattr(self, 'bboxes_data'):
    #         self.bboxes_data = None
        
    #     # Force garbage collection to free up memory
    #     import gc
    #     gc.collect()
        
    #     print_log("Cleanup completed")
        
    #     # # Build visualizer using mmpose registry
    #     # try:
    #     #     # Ensure mmpose scope is set
    #     #     mmengine.registry.init_default_scope('mmpose')
            
    #     #     # Import and build visualizer from mmpose registry
    #     #     from mmpose.registry import VISUALIZERS
    #     #     self.visualizer = VISUALIZERS.build(self.pose_estimator.cfg.visualizer)
    #     #     print_log("Visualizer built successfully from mmpose registry")
            
    #     # except Exception as e:
    #     #     print_log(f"Failed to build visualizer from mmpose registry: {e}")
            
    #     #     # Fallback approach: try to build using mmengine directly
    #     #     try:
    #     #         # Try building with mmengine's build function
    #     #         import mmengine
    #     #         self.visualizer = mmengine.build_from_cfg(
    #     #             self.pose_estimator.cfg.visualizer,
    #     #             mmengine.VISUALIZERS
    #     #         )
    #     #         print_log("Visualizer built using mmengine fallback")
                
    #     #     except Exception as e2:
    #     #         print_log(f"mmengine fallback failed: {e2}")
                
    #     #         # Final fallback: create a simple visualizer
    #     #         try:
    #     #             from mmpose.visualization import PoseLocalVisualizer
    #     #             self.visualizer = PoseLocalVisualizer()
    #     #             print_log("Using PoseLocalVisualizer as final fallback")
    #     #         except Exception as e3:
    #     #             print_log(f"All visualizer creation methods failed: {e3}")
    #     #             # Set to None - visualization will be skipped
    #     #             self.visualizer = None
    #     #             print_log("Visualizer disabled - continuing without visualization")
    #     #             return
        
    #     # # Set dataset metadata
    #     # if self.visualizer and hasattr(self.visualizer, 'set_dataset_meta'):
    #     #     try:
    #     #         self.visualizer.set_dataset_meta(
    #     #             self.pose_estimator.dataset_meta, 
    #     #             skeleton_style=args.skeleton_style
    #     #         )
    #     #         print_log(f"Visualizer configured with skeleton style: {args.skeleton_style}")
    #     #     except Exception as e:
    #     #         print_log(f"Failed to set dataset metadata: {e}")
    #     # else:
    #     #     print_log("Visualizer does not support set_dataset_meta method")
    
    def load_bboxes(self, bbox_file: str):
        """
        Load bounding boxes from JSON file.
        
        Args:
            bbox_file (str): Path to the bounding box JSON file
        """
        print_log(f"Loading bounding boxes from: {bbox_file}")
        
        if not os.path.exists(bbox_file):
            raise FileNotFoundError(f"Bounding box file not found: {bbox_file}")
        
        try:
            with open(bbox_file, 'r') as f:
                self.bboxes_data = json.load(f)
            
            if 'meta' in self.bboxes_data.keys():
                meta = self.bboxes_data.pop('meta')

            num_frames = len(self.bboxes_data)
            print_log(f"Loaded bounding boxes for {num_frames} frames")
            
            # Log sample of bbox data for verification
            if num_frames > 0:
                sample_frame = list(self.bboxes_data.keys())[0]
                sample_bboxes = self.bboxes_data[sample_frame]
                print_log(f"Sample frame {sample_frame} has {len(sample_bboxes)} bounding boxes")
                
        except Exception as e:
            raise RuntimeError(f"Error loading bounding boxes: {e}")
    
    def process_one_image(self, img, frame_idx: Dict, args) -> Optional[object]:
        """
        Process a single image/frame for keypoint detection.
        
        Args:
            img (np.ndarray): Input image
            frame_bboxes (Dict): Bounding boxes for this frame
            args: Command line arguments
            
        Returns:
            Predicted instances with keypoints
        """

        if self.mode == 'bbox':
            if isinstance(img, str):
                tmp_img = mmcv.imread(img, channel_order='rgb')
                height, width, _ = tmp_img.shape
            elif isinstance(img, np.ndarray):
                height, width, _ = img.shape
            else:
                raise TypeError("Input img must be a file path or numpy array")

            # Get bounding boxes for current frame
            frame_bboxes = self.bboxes_data.get(str(frame_idx), {})

                    # Convert normalized bboxes to pixel coordinates
            bboxes = []
            bbox_labels = []
            
            for bbox_id, bbox_coords in frame_bboxes.items():
                # Convert normalized coordinates to pixel coordinates
                x_min = int(bbox_coords[0] * width)
                y_min = int(bbox_coords[1] * height)
                x_max = int(bbox_coords[2] * width)
                y_max = int(bbox_coords[3] * height)
                
                bboxes.append([x_min, y_min, x_max, y_max])
                bbox_labels.append(int(bbox_id))
            
            bboxes = np.array(bboxes, dtype=np.int16)
            bbox_labels = np.array(bbox_labels, dtype=np.int16)

            print_log(f"Frame {frame_idx}: Loaded {len(bboxes)} bounding boxes from pre-computed data : {bboxes}")

        else:
                # predict bbox
            det_result = inference_detector(self.detector, img)
            pred_instance = det_result.pred_instances.cpu().numpy()
            bboxes = np.concatenate(
                (pred_instance.bboxes, pred_instance.scores[:, None]), axis=1)
            bboxes = bboxes[np.logical_and(pred_instance.labels == args.det_cat_id,
                                        pred_instance.scores > args.bbox_thr)]
            bboxes = bboxes[nms(bboxes, args.nms_thr), :4]
            bbox_labels = np.ones(len(bboxes), dtype=np.int16) * -1  # Unknown labels
        
        if len(bboxes) == 0:
            print_log("No valid bounding boxes")
            return None
        
        try:
            # Perform pose estimation
            pose_results = inference_topdown(self.pose_estimator, img, bboxes)
            data_samples = merge_data_samples(pose_results)
            
            # Add keypoint labels
            pred_inst = data_samples.get('pred_instances', None)
            
            if pred_inst is not None:
                # Handle case where bbox count doesn't match prediction count
                if len(bbox_labels) != len(pred_inst.get('bbox_scores', [])):
                    print_log(f"Bbox count mismatch: {len(bbox_labels)} vs {len(pred_inst.get('bbox_scores', []))}")
                    bbox_labels = np.ones(len(pred_inst.get('bbox_scores', [])), dtype=np.int16) * -1
                
                pred_inst.set_data({'keypoints_label': bbox_labels})
            
            
            # show the results
            if isinstance(img, str):
                img = mmcv.imread(img, channel_order='rgb')
            elif isinstance(img, np.ndarray):
                img = mmcv.bgr2rgb(img)

            if args.dark_background:
                H, W = img.shape[:2]
                img = np.zeros((H, W, 3), dtype=np.uint8)  # black background
                
            # Visualize if required
            if self.visualizer is not None:               
                self.visualizer.add_datasample(
                    'result',
                    img,
                    data_sample=data_samples,
                    draw_gt=False,
                    draw_heatmap=args.draw_heatmap,
                    draw_bbox=args.draw_bbox,
                    show_kpt_idx=args.show_kpt_idx,
                    skeleton_style=args.skeleton_style,
                    show=args.show,
                    wait_time=args.show_interval,
                    kpt_thr=args.kpt_thr
                )
            
            return data_samples.get('pred_instances', None)
            
        except Exception as e:
            print_log(f"Error processing frame: {e}")
            return None
    
    def process_video(self, input_path: str, args) -> List[Dict]:
        """
        Process a video file for keypoint detection.
        
        Args:
            input_path (str): Path to input video
            args: Command line arguments
            
        Returns:
            List of prediction results for each frame
        """
        print_log(f"Processing video: {input_path}")
        
        if not os.path.exists(input_path):
            raise FileNotFoundError(f"Input video not found: {input_path}")
        
        # Open video capture
        cap = cv2.VideoCapture(input_path)
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open video: {input_path}")
        
        # Get video properties
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        # Determine max frames based on mode
        if self.mode == 'bbox' and self.bboxes_data:
            max_frame = self.bboxes_data.keys()
            max_frame = [int(x) for x in max_frame]
            max_frame = max(max_frame)
            total_frames_to_process = max_frame + 1
        else:
            # Use all frames from video when using detection mode
            max_frame = total_frames - 1
            total_frames_to_process = total_frames

        print_log(f"Video properties:")
        if self.mode == 'bbox':
            print_log(f"  - Total frames: {total_frames} / {total_frames_to_process} (using {total_frames_to_process} frames based on bbox data)")
        else:
            print_log(f"  - Total frames: {total_frames} (processing all frames with detection)")
        print_log(f"  - FPS: {fps}")
        print_log(f"  - Resolution: {frame_width}x{frame_height}")
        
        # Setup video writer if needed
        video_writer = None
        output_file = None
        
        if args.save_video and args.output_root:
            mmengine.mkdir_or_exist(args.output_root)
            output_file = os.path.join(args.output_root, os.path.basename(input_path))
            if output_file.endswith('.mp4'):
                output_file = output_file[:-4] + '_keypoints.mp4'
            else:
                output_file += '_keypoints.mp4'
        
        # Process frames
        pred_instances_list = []
        frame_idx = 0


        
        pbar = tqdm.tqdm(desc='Processing frames', total=total_frames_to_process)
        
        try:
            while cap.isOpened():
                success, frame = cap.read()
                
                if not success:
                    break
                

                
                # Process frame
                pred_instances = self.process_one_image(frame, frame_idx, args)
                
                # Save predictions
                if args.save_predictions:
                    frame_data = {
                        'frame_id': int(frame_idx),
                        'instances': split_instances(pred_instances) if pred_instances else []
                    }
                    pred_instances_list.append(frame_data)
                
                # Handle video output
                if output_file and pred_instances and self.visualizer:
                    frame_vis = self.visualizer.get_image()
                    
                    if video_writer is None:
                        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                        video_writer = cv2.VideoWriter(
                            output_file,
                            fourcc,
                            fps,
                            (frame_vis.shape[1], frame_vis.shape[0])
                        )
                    
                    video_writer.write(mmcv.rgb2bgr(frame_vis))
                
                # Handle display
                if args.show:
                    if cv2.waitKey(5) & 0xFF == 27:  # ESC key
                        break
                    time.sleep(args.show_interval)
                
                frame_idx += 1
                pbar.update(1)
                if frame_idx >= total_frames_to_process:
                    break

        finally:
            pbar.close()
            cap.release()
            
            if video_writer:
                video_writer.release()
                print_log(f"Output video saved to: {output_file}")
            
            if args.show:
                cv2.destroyAllWindows()
        
        print_log(f"Processed {frame_idx} frames")
        return pred_instances_list
    

    def save_predictions(self, predictions: List[Dict], output_path: str):
        """
        Save prediction results to JSON file.
        
        Args:
            predictions (List[Dict]): List of prediction results
            output_path (str): Path to save the results
        """
        print_log(f"Saving predictions to: {output_path}")
        
        # Ensure output directory exists
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # Prepare output data
        output_data = {
            'meta_info': self.pose_estimator.dataset_meta,
            'instance_info': predictions
        }
        
        try:
            def convert_ndarray(obj):
                if isinstance(obj, np.ndarray):
                    return obj.astype(float).tolist()
                elif isinstance(obj, dict):
                    return {k: convert_ndarray(v) for k, v in obj.items()}
                elif isinstance(obj, list):
                    return [convert_ndarray(v) for v in obj]
                elif isinstance(obj, np.float32):
                    return float(obj)
                else:
                    return obj

            output_data = convert_ndarray(output_data)
            with open(output_path, 'w') as f:
                json.dump(output_data, f, indent=2)
            
            # Log statistics
            total_instances = sum(len(frame['instances']) for frame in predictions)
            print_log(f"Saved predictions for {len(predictions)} frames")
            print_log(f"Total instances detected: {total_instances}")
            
        except Exception as e:
            raise RuntimeError(f"Error saving predictions: {e}")


def create_parser():
    """Create argument parser with all required options."""
    parser = argparse.ArgumentParser(
        description='MMPose 2D human pose estimation',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage
  python mmpose_inference.py config.py checkpoint.pth --input video.mp4 --bboxes bboxes.json --output-root ./output

  # With visualization
  python mmpose_inference.py config.py checkpoint.pth --input video.mp4 --bboxes bboxes.json --output-root ./output --save-video --draw-bbox

  # Save predictions only
  python mmpose_inference.py config.py checkpoint.pth --input video.mp4 --bboxes bboxes.json --output-root ./output --save-predictions

Important Notes:
  - This script requires a custom modification to MMPose (see documentation)
  - Bounding boxes should be in normalized coordinates (0-1 range)
  - CUDA device is recommended for better performance
        """
    )
    
    # Required arguments
    parser.add_argument('pose_config', help='Path to MMPose configuration file')
    parser.add_argument('pose_checkpoint', help='Path to model checkpoint file')

    # Possiblity to use mmdet for bbox detection
    parser.add_argument('--det_config', help='Config file for detection', default=r"/home/shares/schaerm/schaer2/idtracking_keypoint/vitpose_sam2/mmpose/demo/mmdetection_cfg/rtmdet_m_640-8xb32_coco-person.py")
    parser.add_argument('--det_checkpoint', help='Checkpoint file for detection', default=r"/home/shares/schaerm/schaer2/idtracking_keypoint/pose_model_weight/cspnext-m_8xb256-rsb-a1-600e_in1k-ecb3bbd9.pth")
    parser.add_argument(
        '--det-cat-id',
        type=int,
        default=0,
        help='Category id for bounding box detection model')
    parser.add_argument(
        '--bbox-thr',
        type=float,
        default=0.3,
        help='Bounding box score threshold')
    parser.add_argument(
        '--nms-thr',
        type=float,
        default=0.3,
        help='IoU threshold for bounding box NMS')
    
    # Input/Output
    parser.add_argument('--input', type=str, required=True, help='Path to input video/image file')
    parser.add_argument('--bboxes', type=str, required=False, help='Path to JSON file containing bounding boxes')
    parser.add_argument('--output-root', type=str, default='', help='Root directory for output files')
    parser.add_argument('--suffix', type=str, default='', help='Suffix for output files if needed')
    
    # Model settings
    parser.add_argument('--device', default='cuda:0', help='Device for inference (cuda:0, cpu, etc.)')
    parser.add_argument('--kpt-thr', type=float, default=0.3, help='Keypoint confidence threshold')
    
    # Visualization settings
    parser.add_argument('--skeleton-style', default='mmpose', choices=['mmpose', 'openpose'], 
                       help='Skeleton style for visualization')
    parser.add_argument('--radius', type=int, default=3, help='Keypoint radius for visualization')
    parser.add_argument('--thickness', type=int, default=1, help='Line thickness for visualization')
    parser.add_argument('--alpha', type=float, default=0.8, help='Transparency of bounding boxes')
    parser.add_argument('--dark_background', action='store_true', help='Transparency of bounding boxes')
    
    # Output options
    parser.add_argument('--save-predictions', action='store_true', help='Save prediction results to JSON')
    parser.add_argument('--save-video', action='store_true', help='Save visualization video')
    parser.add_argument('--show', action='store_true', help='Display results in real-time')
    parser.add_argument('--show-interval', type=int, default=0, help='Display interval in milliseconds')
    
    # Drawing options
    parser.add_argument('--draw-heatmap', action='store_true', help='Draw prediction heatmaps')
    parser.add_argument('--draw-bbox', action='store_true', help='Draw bounding boxes')
    parser.add_argument('--show-kpt-idx', action='store_true', help='Show keypoint indices')
    
    # Logging
    parser.add_argument('--verbose', '-v', action='store_true', help='Enable verbose logging')
    
    return parser


# def clean_global_state():
#     """
#     Clean global state that might accumulate between runs.
#     Useful for notebook environments where the same process runs multiple times.
#     """
#     print_log("Cleaning global state...")
    
#     # Reset mmengine registry scope
#     mmengine.registry.init_default_scope('mmpose')
    
#     # Force garbage collection
#     import gc
#     gc.collect()
    
#     # Clear CUDA cache if available
#     try:
#         import torch
#         if torch.cuda.is_available():
#             torch.cuda.empty_cache()
#             print_log("CUDA cache cleared")
#     except ImportError:
#         pass
    
#     print_log("Global state cleaned")


def main(args=None):
    """Main function to run MMPose inference."""
    parser = create_parser()
    args = parser.parse_args(args)
    
    # Set logging level
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Validate arguments
    if not (args.show or args.output_root):
        parser.error("Either --show or --output-root must be specified")
    
    if not args.input:
        parser.error("--input is required")

    if not (args.bboxes):
        print_log("--bboxes not provided, falling back to detection model")
    
    # Log startup information
    print_log("Starting MMPose inference")
    print_log(f"Configuration:")
    print_log(f"  - Pose config: {args.pose_config}")
    print_log(f"  - Pose checkpoint: {args.pose_checkpoint}")
    print_log(f"  - Input video: {args.input}")
    print_log(f"  - Bounding boxes: {args.bboxes}")
    print_log(f"  - Detection config: {args.det_config}")
    print_log(f"  - Detection checkpoint: {args.det_checkpoint}")
    print_log(f"  - Output root: {args.output_root}")
    print_log(f"  - Device: {args.device}")

    if args.bboxes:
        print_log(f"  - Using pre-computed bounding boxes from: {args.bboxes}")
        mode = 'bbox'
    else:
        print_log(f"  - Using detection model for bounding boxes")
        mode = 'mmdet'
    
    try:
        # Initialize inference system
        inference_system = MMPoseInference(
            args.pose_config, 
            args.pose_checkpoint, 
            args.det_config,
            args.det_checkpoint,
            args.device,
            mode=mode
        )
        
        # Setup visualizer
        inference_system.setup_visualizer(args)

        if mode == 'bbox':
            # Load bounding boxes
            inference_system.load_bboxes(args.bboxes)

        # Determine input type
        input_type = mimetypes.guess_type(args.input)[0]
        if input_type is None:
            print_log(f"Cannot determine file type for {args.input}, assuming video")
            input_type = 'video/mp4'
        
        input_category = input_type.split('/')[0]
        
        input_name = Path(args.input).stem
        if args.suffix:
            input_name += args.suffix
        pred_output_file = os.path.join(args.output_root, f'results_skeleton_{input_name}.json')

        if input_category == 'image':
            # Process frame
            pred_instances = inference_system.process_one_image(args.input, 0, args)

            if args.output_root and args.save_predictions:
                pred_instances_list = split_instances(pred_instances) if pred_instances else []

                img_name = Path(args.input).stem
                
                # Save predictions to JSON
                inference_system.save_predictions([{
                    'frame_id': 0,
                    'instances': pred_instances_list
                }], pred_output_file)

                # Save visualization image if visualizer exists
                if inference_system.visualizer and hasattr(inference_system.visualizer, 'get_image'):
                    try:
                        img_vis = inference_system.visualizer.get_image()
                        img_output_file = os.path.join(args.output_root, f'{img_name}_vis.jpg')
                        mmcv.imwrite(mmcv.rgb2bgr(img_vis), img_output_file)
                        print_log(f"Visualization saved to: {img_output_file}")
                    except Exception as e:
                        print_log(f"Failed to save visualization: {e}")
                else:
                    print_log("No visualizer available for saving image visualization")

        elif input_category == 'video':
            # Process video
            predictions = inference_system.process_video(args.input, args)
            
            # Save predictions if requested
            if args.save_predictions and args.output_root:
                inference_system.save_predictions(predictions, pred_output_file)
        else:
            print_log(f"Unsupported input type: {input_type}")
            sys.exit(1)
        
        print_log("MMPose inference completed successfully!")
        
        # Cleanup resources for next run
        # inference_system.cleanup()
        
    except KeyboardInterrupt:
        print_log("Process interrupted by user")
        # Cleanup on interrupt as well
        # if 'inference_system' in locals():
        #     inference_system.cleanup()
    except Exception as e:
        print_log(f"Error during inference: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        # # Cleanup on error as well
        # if 'inference_system' in locals():
        #     inference_system.cleanup()
        sys.exit(1)


if __name__ == "__main__":
    main()
