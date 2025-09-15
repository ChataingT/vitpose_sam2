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

import argparse
import cv2
import json
import logging
import mimetypes
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import mmcv
import mmengine
import numpy as np
from mmengine.logging import print_log
from tqdm import tqdm

try:
    from mmpose.apis import inference_topdown
    from mmpose.apis import init_model as init_pose_estimator
    from mmpose.registry import VISUALIZERS
    from mmpose.structures import merge_data_samples, split_instances
    MMPOSE_AVAILABLE = True
except ImportError:
    MMPOSE_AVAILABLE = False
    print("ERROR: MMPose is not installed or not properly configured.")
    print("Please install MMPose and ensure the custom modification is applied.")
    sys.exit(1)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        # logging.FileHandler('mmpose_inference.log')
    ]
)
logger = logging.getLogger(__name__)


class MMPoseInference:
    """
    MMPose inference class for 2D human pose estimation.
    
    This class handles video processing, bounding box mapping, keypoint detection,
    and result visualization using MMPose framework with ViTPose models.
    """
    
    def __init__(self, pose_config: str, pose_checkpoint: str, device: str = 'cuda:0'):
        """
        Initialize the MMPose inference system.
        
        Args:
            pose_config (str): Path to MMPose configuration file
            pose_checkpoint (str): Path to model checkpoint file
            device (str): Device to use for inference ('cuda:0', 'cpu', etc.)
        """
        self.pose_config = pose_config
        self.pose_checkpoint = pose_checkpoint
        self.device = device
        
        # Model components
        self.pose_estimator = None
        self.visualizer = None
        
        # Processing data
        self.bboxes_data = None
        self.pred_instances_list = []
        
        logger.info("Initializing MMPose inference system...")
        self._validate_inputs()
        self._init_pose_estimator()
    
    def _validate_inputs(self):
        """Validate input files and configurations."""
        if not os.path.exists(self.pose_config):
            raise FileNotFoundError(f"Pose config file not found: {self.pose_config}")
        
        if not os.path.exists(self.pose_checkpoint):
            raise FileNotFoundError(f"Pose checkpoint file not found: {self.pose_checkpoint}")
        
        logger.info(f"Using pose config: {self.pose_config}")
        logger.info(f"Using pose checkpoint: {self.pose_checkpoint}")
        logger.info(f"Using device: {self.device}")
    
    def _init_pose_estimator(self):
        """Initialize the pose estimation model."""
        logger.info("Loading pose estimation model...")
        
        try:
            self.pose_estimator = init_pose_estimator(
                self.pose_config,
                self.pose_checkpoint,
                device=self.device,
                cfg_options=dict(
                    model=dict(test_cfg=dict(output_heatmaps=False))
                )
            )
            logger.info("Pose estimator loaded successfully")
            
        except Exception as e:
            raise RuntimeError(f"Failed to initialize pose estimator: {e}")
    
    def setup_visualizer(self, args):
        """
        Setup the visualization components.
        
        Args:
            args: Command line arguments containing visualization parameters
        """
        logger.info("Setting up visualizer...")
        
        # Configure visualizer parameters
        self.pose_estimator.cfg.visualizer.radius = args.radius
        self.pose_estimator.cfg.visualizer.alpha = args.alpha
        self.pose_estimator.cfg.visualizer.line_width = args.thickness
        
        # Build visualizer
        self.visualizer = VISUALIZERS.build(self.pose_estimator.cfg.visualizer)
        
        # Set dataset metadata
        self.visualizer.set_dataset_meta(
            self.pose_estimator.dataset_meta, 
            skeleton_style=args.skeleton_style
        )
        
        logger.info(f"Visualizer configured with skeleton style: {args.skeleton_style}")
    
    def load_bboxes(self, bbox_file: str):
        """
        Load bounding boxes from JSON file.
        
        Args:
            bbox_file (str): Path to the bounding box JSON file
        """
        logger.info(f"Loading bounding boxes from: {bbox_file}")
        
        if not os.path.exists(bbox_file):
            raise FileNotFoundError(f"Bounding box file not found: {bbox_file}")
        
        try:
            with open(bbox_file, 'r') as f:
                self.bboxes_data = json.load(f)
            
            num_frames = len(self.bboxes_data)
            logger.info(f"Loaded bounding boxes for {num_frames} frames")
            
            # Log sample of bbox data for verification
            if num_frames > 0:
                sample_frame = list(self.bboxes_data.keys())[0]
                sample_bboxes = self.bboxes_data[sample_frame]
                logger.info(f"Sample frame {sample_frame} has {len(sample_bboxes)} bounding boxes")
                
        except Exception as e:
            raise RuntimeError(f"Error loading bounding boxes: {e}")
    
    def process_one_image(self, img: np.ndarray, frame_bboxes: Dict, args) -> Optional[object]:
        """
        Process a single image/frame for keypoint detection.
        
        Args:
            img (np.ndarray): Input image
            frame_bboxes (Dict): Bounding boxes for this frame
            args: Command line arguments
            
        Returns:
            Predicted instances with keypoints
        """
        if not frame_bboxes:
            logger.debug("No bounding boxes for this frame")
            return None
        
        height, width, _ = img.shape
        
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
        
        if len(bboxes) == 0:
            logger.debug("No valid bounding boxes after conversion")
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
                    logger.warning(f"Bbox count mismatch: {len(bbox_labels)} vs {len(pred_inst.get('bbox_scores', []))}")
                    bbox_labels = np.ones(len(pred_inst.get('bbox_scores', [])), dtype=np.int16) * -1
                
                pred_inst.set_data({'keypoints_label': bbox_labels})
            
            # Visualize if required
            if self.visualizer is not None and (args.show or args.output_root):
                if isinstance(img, np.ndarray):
                    img_rgb = mmcv.bgr2rgb(img)
                else:
                    img_rgb = mmcv.imread(img, channel_order='rgb')
                
                self.visualizer.add_datasample(
                    'result',
                    img_rgb,
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
            logger.error(f"Error processing frame: {e}")
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
        logger.info(f"Processing video: {input_path}")
        
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
        
        logger.info(f"Video properties:")
        logger.info(f"  - Total frames: {total_frames}")
        logger.info(f"  - FPS: {fps}")
        logger.info(f"  - Resolution: {frame_width}x{frame_height}")
        
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
        
        pbar = tqdm(desc='Processing frames', total=total_frames)
        
        try:
            while cap.isOpened():
                success, frame = cap.read()
                
                if not success:
                    break
                
                # Get bounding boxes for current frame
                frame_bboxes = self.bboxes_data.get(str(frame_idx), {})
                
                # Process frame
                pred_instances = self.process_one_image(frame, frame_bboxes, args)
                
                # Save predictions
                if args.save_predictions:
                    frame_data = {
                        'frame_id': int(frame_idx),
                        'instances': split_instances(pred_instances) if pred_instances else []
                    }
                    pred_instances_list.append(frame_data)
                
                # Handle video output
                if output_file and pred_instances:
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
                
        finally:
            pbar.close()
            cap.release()
            
            if video_writer:
                video_writer.release()
                logger.info(f"Output video saved to: {output_file}")
            
            if args.show:
                cv2.destroyAllWindows()
        
        logger.info(f"Processed {frame_idx} frames")
        return pred_instances_list
    
    def save_predictions(self, predictions: List[Dict], output_path: str):
        """
        Save prediction results to JSON file.
        
        Args:
            predictions (List[Dict]): List of prediction results
            output_path (str): Path to save the results
        """
        logger.info(f"Saving predictions to: {output_path}")
        
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
            logger.info(f"Saved predictions for {len(predictions)} frames")
            logger.info(f"Total instances detected: {total_instances}")
            
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
    
    # Input/Output
    parser.add_argument('--input', type=str, required=True, help='Path to input video file')
    parser.add_argument('--bboxes', type=str, required=True, help='Path to JSON file containing bounding boxes')
    parser.add_argument('--output-root', type=str, default='', help='Root directory for output files')
    
    # Model settings
    parser.add_argument('--device', default='cuda:0', help='Device for inference (cuda:0, cpu, etc.)')
    parser.add_argument('--kpt-thr', type=float, default=0.3, help='Keypoint confidence threshold')
    
    # Visualization settings
    parser.add_argument('--skeleton-style', default='mmpose', choices=['mmpose', 'openpose'], 
                       help='Skeleton style for visualization')
    parser.add_argument('--radius', type=int, default=3, help='Keypoint radius for visualization')
    parser.add_argument('--thickness', type=int, default=1, help='Line thickness for visualization')
    parser.add_argument('--alpha', type=float, default=0.8, help='Transparency of bounding boxes')
    
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


def main():
    """Main function to run MMPose inference."""
    parser = create_parser()
    args = parser.parse_args()
    
    # Check MMPose availability
    if not MMPOSE_AVAILABLE:
        logger.error("MMPose is not available. Please install MMPose and apply required modifications.")
        sys.exit(1)
    
    # Set logging level
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Validate arguments
    if not (args.show or args.output_root):
        parser.error("Either --show or --output-root must be specified")
    
    if not args.input:
        parser.error("--input is required")
    
    # Log startup information
    logger.info("Starting MMPose inference")
    logger.info(f"Configuration:")
    logger.info(f"  - Pose config: {args.pose_config}")
    logger.info(f"  - Pose checkpoint: {args.pose_checkpoint}")
    logger.info(f"  - Input video: {args.input}")
    logger.info(f"  - Bounding boxes: {args.bboxes}")
    logger.info(f"  - Output root: {args.output_root}")
    logger.info(f"  - Device: {args.device}")
    
    try:
        # Initialize inference system
        inference_system = MMPoseInference(
            args.pose_config, 
            args.pose_checkpoint, 
            args.device
        )
        
        # Setup visualizer
        inference_system.setup_visualizer(args)
        
        # Load bounding boxes
        inference_system.load_bboxes(args.bboxes)
        
        # Determine input type
        input_type = mimetypes.guess_type(args.input)[0]
        if input_type is None:
            logger.warning(f"Cannot determine file type for {args.input}, assuming video")
            input_type = 'video/mp4'
        
        input_category = input_type.split('/')[0]
        
        if input_category == 'image':
            logger.error("Image input is not currently supported. Please use video input.")
            sys.exit(1)
        elif input_category == 'video':
            # Process video
            predictions = inference_system.process_video(args.input, args)
            
            # Save predictions if requested
            if args.save_predictions and args.output_root:
                video_name = Path(args.input).stem
                pred_save_path = os.path.join(args.output_root, f'results_skeleton_{video_name}.json')
                inference_system.save_predictions(predictions, pred_save_path)
        else:
            logger.error(f"Unsupported input type: {input_type}")
            sys.exit(1)
        
        logger.info("MMPose inference completed successfully!")
        
    except KeyboardInterrupt:
        logger.info("Process interrupted by user")
    except Exception as e:
        logger.error(f"Error during inference: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
