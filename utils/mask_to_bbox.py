#!/usr/bin/env python3
"""
Mask to Bounding Box Converter

This script converts mask data from a JSON file to bounding box coordinates.
It processes video frames, extracts masks, converts them to bounding boxes,
and saves the results in a normalized format.

Usage:
    python mask_to_bbox.py --video_path VIDEO --mask_json MASK_JSON --output_dir OUTPUT_DIR [options]

Example:
    python mask_to_bbox.py \
        --video_path /path/to/video.mp4 \
        --mask_json /path/to/mask_data.json \
        --output_dir /path/to/output \
        --resize \
        --expected_ids 0 1 2
"""

import argparse
import cv2
import json
import logging
import numpy as np
import os
import sys
from pathlib import Path
from tqdm import tqdm
import decord

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        # logging.FileHandler('mask_to_bbox.log')
    ]
)
logger = logging.getLogger(__name__)


class MaskToBBoxConverter:
    """
    A class to convert mask data to bounding box coordinates.
    
    This class handles the conversion of Run-Length Encoded (RLE) masks
    to bounding boxes with proper scaling and normalization.
    """
    
    def __init__(self, video_path, mask_json_path, output_dir, resize=True):
        """
        Initialize the converter with input parameters.
        
        Args:
            video_path (str): Path to the video file
            mask_json_path (str): Path to the JSON file containing mask data
            output_dir (str): Directory to save the output JSON file
            resize (bool): Whether to resize masks to match video dimensions
        """
        self.video_path = video_path
        self.mask_json_path = mask_json_path
        self.output_dir = output_dir
        self.resize = resize
        
        # Initialize data containers
        self.bbox_dict = {}
        self.missing_data = []
        self.expected_ids = []  # Will be populated from mask data
        
        # Video properties
        self.vr = None
        self.vr_width = None
        self.vr_height = None
        # self.src_width = None
        # self.src_height = None
        # self.ratio_x = None
        # self.ratio_y = None
        
        # Mask data
        self.mask_data = None
        
    def load_video(self):
        """Load the video file and extract its properties."""
        logger.info(f"Loading video from: {self.video_path}")
        
        if not os.path.exists(self.video_path):
            raise FileNotFoundError(f"Video file not found: {self.video_path}")
            
        try:
            self.vr = decord.VideoReader(self.video_path)
            self.vr_width = self.vr[0].shape[1]
            self.vr_height = self.vr[0].shape[0]
            
            logger.info(f"Video loaded successfully:")
            logger.info(f"  - Frames: {len(self.vr)}")
            logger.info(f"  - Dimensions: {self.vr_width} x {self.vr_height}")
            
        except Exception as e:
            raise RuntimeError(f"Error loading video: {e}")
    
    def load_mask_data(self):
        """Load mask data from JSON file and extract available target IDs."""
        logger.info(f"Loading mask data from: {self.mask_json_path}")
        
        if not os.path.exists(self.mask_json_path):
            raise FileNotFoundError(f"Mask JSON file not found: {self.mask_json_path}")
            
        try:
            with open(self.mask_json_path, 'r') as f:
                self.mask_data = json.load(f)
            
            num_frames = len(self.mask_data.get('frames', []))
            logger.info(f"Mask data loaded successfully with {num_frames} frames")
            
            # Extract all unique target IDs from the mask data
            self._extract_target_ids()
            
        except Exception as e:
            raise RuntimeError(f"Error loading mask data: {e}")
    
    def _extract_target_ids(self):
        """Extract all unique target IDs from the mask data."""
        target_ids_set = set()
        frames = self.mask_data.get('frames', [])
        
        for frame in frames:
            masks = frame.get('masks', {})
            if masks and 'target_ids' in masks:
                target_ids_set.update(masks['target_ids'])
        
        self.expected_ids = sorted(list(target_ids_set))
        
        if self.expected_ids:
            logger.info(f"Extracted target IDs from mask data: {self.expected_ids}")
        else:
            logger.warning("No target IDs found in mask data")
            self.expected_ids = []
    
    # def compute_scaling_ratios(self, source_video_path):
        """
        Compute scaling ratios between source video and target video.
        
        Args:
            source_video_path (str): Path to the source video used for mask generation
        """
        # logger.info(f"Computing scaling ratios using source video: {source_video_path}")
        
        # if not os.path.exists(source_video_path):
        #     logger.warning(f"Source video not found: {source_video_path}")
        #     logger.warning("Using 1:1 scaling ratios")
        #     # self.ratio_x = 1.0
        #     # self.ratio_y = 1.0
        #     return
            
        # try:
            # cap = cv2.VideoCapture(source_video_path)
            # self.src_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            # self.src_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            # cap.release()
            
            # self.ratio_x = self.vr_width / self.src_width
            # self.ratio_y = self.vr_height / self.src_height
            
            # logger.info(f"Source video dimensions: {self.src_width} x {self.src_height}")
            # logger.info(f"Scale ratios - X: {self.ratio_x:.4f}, Y: {self.ratio_y:.4f}")
            
        # except Exception as e:
        #     logger.error(f"Error computing scaling ratios: {e}")
        #     logger.warning("Using 1:1 scaling ratios")
        #     self.ratio_x = 1.0
        #     self.ratio_y = 1.0
    
    @staticmethod
    def mask_to_bbox(mask):
        """
        Convert a binary mask to a bounding box.
        
        Args:
            mask (np.ndarray): Binary mask array
            
        Returns:
            list: Bounding box coordinates [x_min, y_min, x_max, y_max]
        """
        rows = np.any(mask, axis=1)
        cols = np.any(mask, axis=0)
        
        if rows.any():
            y_min, y_max = np.where(rows)[0][[0, -1]]
            x_min, x_max = np.where(cols)[0][[0, -1]]
            return [int(x_min), int(y_min), int(x_max), int(y_max)]
        
        return [0, 0, 0, 0]
    
    @staticmethod
    def rle_to_mask(rle):
        """
        Convert Run-Length Encoded data to a binary mask.
        
        Args:
            rle (dict): RLE data containing 'size' and 'counts'
            
        Returns:
            np.ndarray: Binary mask array
        """
        h, w = rle["size"]
        mask = np.empty(h * w, dtype=bool)
        idx = 0
        parity = False
        
        for count in rle["counts"]:
            mask[idx : idx + count] = parity
            idx += count
            parity ^= True
            
        mask = mask.reshape(w, h)
        return mask.transpose()  # Put in C order
    
    def get_mask(self, frame_number):
        """
        Retrieve the masks associated with a specific frame.
        
        Args:
            frame_number (int): Frame number to retrieve masks for
            
        Returns:
            tuple: (points, bboxs, masks) data for the frame
        """
        frames = self.mask_data.get('frames')
        
        if frame_number < len(frames):
            frame = frames[frame_number]
            points = frame.get('points', {})
            bboxs = frame.get('bboxs', {})
            masks = frame.get('masks', [])
            return points, bboxs, masks
            
        return None, None, None
    
    def process_frames(self):
        """Process all frames in the video and convert masks to bounding boxes."""
        logger.info("Starting frame processing...")
        logger.info(f"Expected target IDs: {self.expected_ids}")
        logger.info(f"Resize masks: {self.resize}")
        
        total_frames = len(self.vr)
        processed_frames = 0
        frames_with_masks = 0
        
        # Process each frame
        for frame_id in tqdm(range(total_frames), desc="Processing frames"):
            _, _, masks = self.get_mask(frame_id)
            
            if masks:
                frames_with_masks += 1
                self.bbox_dict[str(frame_id)] = {}
                
                # Process each mask in the frame
                for target_id, rle_mask in zip(masks.get('target_ids'), masks.get('rle_masks')):
                    try:
                        # Convert RLE to mask
                        mask = self.rle_to_mask(rle_mask)
                        
                        # Resize mask if requested
                        if self.resize:
                            mask = cv2.resize(
                                mask.astype(np.uint8), 
                                (self.vr_width, self.vr_height), 
                                interpolation=cv2.INTER_NEAREST
                            )
                        
                        # Convert mask to bounding box
                        bbox = self.mask_to_bbox(mask)
                        
                        if bbox == [0, 0, 0, 0]:
                            self.missing_data.append((frame_id, target_id))
                            logger.debug(f"Empty bbox for frame {frame_id}, target {target_id}")
                        else:
                            # Normalize bounding box coordinates
                            normalized_bbox = [
                                bbox[0] / self.vr_width,
                                bbox[1] / self.vr_height,
                                bbox[2] / self.vr_width,
                                bbox[3] / self.vr_height
                            ]
                            
                            # Save bounding box in dictionary
                            self.bbox_dict[str(frame_id)][target_id] = normalized_bbox
                            
                    except Exception as e:
                        logger.error(f"Error processing mask for frame {frame_id}, target {target_id}: {e}")
                        self.missing_data.append((frame_id, target_id))
            else:
                # No masks found for this frame, mark all expected IDs as missing
                for expected_id in self.expected_ids:
                    self.missing_data.append((frame_id, expected_id))
            
            processed_frames += 1
        
        logger.info(f"Frame processing completed:")
        logger.info(f"  - Total frames: {total_frames}")
        logger.info(f"  - Frames with masks: {frames_with_masks}")
        logger.info(f"  - Missing data entries: {len(self.missing_data)}")
    
    def save_results(self):
        """Save the bounding box results to a JSON file."""
        # Create output directory if it doesn't exist
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Generate output filename based on mask JSON filename
        mask_name = Path(self.mask_json_path).stem
        # Replace 'mask' with 'bbox' in the filename
        if 'mask' in mask_name:
            output_filename = mask_name.replace('mask', 'bbox') + '.json'
        else:
            output_filename = f"{mask_name}_bbox.json"
        output_path = os.path.join(self.output_dir, output_filename)
        
        logger.info(f"Saving results to: {output_path}")
        
        try:
            with open(output_path, 'w') as outfile:
                json.dump(self.bbox_dict, outfile, indent=2)
            
            # Save missing data report
            missing_data_filename = mask_name.replace('mask', 'missing_data') + '.json' if 'mask' in mask_name else f"{mask_name}_missing_data.json"
            missing_data_path = os.path.join(self.output_dir, missing_data_filename)
            with open(missing_data_path, 'w') as outfile:
                json.dump(self.missing_data, outfile, indent=2)
            
            logger.info(f"Results saved successfully:")
            logger.info(f"  - Bounding boxes: {output_path}")
            logger.info(f"  - Missing data report: {missing_data_path}")
            logger.info(f"  - Total bounding boxes: {sum(len(frame_data) for frame_data in self.bbox_dict.values())}")
            
        except Exception as e:
            raise RuntimeError(f"Error saving results: {e}")
    
    def run(self, source_video_path=None):
        """
        Run the complete mask-to-bbox conversion process.
        
        Args:
            source_video_path (str, optional): Path to source video for scaling computation
        """
        logger.info("Starting mask-to-bbox conversion process")
        
        try:
            # Load video and mask data
            self.load_video()
            self.load_mask_data()
            
            # Compute scaling ratios if source video is provided
            # if source_video_path:
            #     self.compute_scaling_ratios(source_video_path)
            
            # Process frames
            self.process_frames()
            
            # Save results
            self.save_results()
            
            logger.info("Mask-to-bbox conversion completed successfully!")
            
        except Exception as e:
            logger.error(f"Error during conversion: {e}")
            raise


def main():
    """Main function to handle command line arguments and run the converter."""
    parser = argparse.ArgumentParser(
        description="Convert mask data to bounding box coordinates",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage
  python mask_to_bbox.py --video_path video.mp4 --mask_json masks.json --output_dir ./output

  # Without resizing
  python mask_to_bbox.py --video_path video.mp4 --mask_json masks.json --output_dir ./output --no_resize
        """
    )
    
    parser.add_argument(
        '--video_path',
        type=str,
        required=True,
        help='Path to the input video file'
    )
    
    parser.add_argument(
        '--mask_json',
        type=str,
        required=True,
        help='Path to the JSON file containing mask data'
    )
    
    parser.add_argument(
        '--output_dir',
        type=str,
        required=True,
        help='Directory to save the output JSON file'
    )
    
    # parser.add_argument(
    #     '--source_video',
    #     type=str,
    #     help='Path to the source video used for mask generation (for scaling computation)'
    # )
    
    parser.add_argument(
        '--no_resize',
        action='store_true',
        help='Do not resize masks to match video dimensions'
    )
    
    parser.add_argument(
        '--verbose',
        '-v',
        action='store_true',
        help='Enable verbose logging'
    )
    
    args = parser.parse_args()
    
    # Set logging level
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Log the arguments
    logger.info("Starting with arguments:")
    logger.info(f"  - Video path: {args.video_path}")
    logger.info(f"  - Mask JSON: {args.mask_json}")
    logger.info(f"  - Output directory: {args.output_dir}")
    # logger.info(f"  - Source video: {args.source_video}")
    logger.info(f"  - Resize: {not args.no_resize}")
    
    # Create converter and run
    converter = MaskToBBoxConverter(
        video_path=args.video_path,
        mask_json_path=args.mask_json,
        output_dir=args.output_dir,
        resize=not args.no_resize
    )
    
    # converter.run(source_video_path=args.source_video)
    converter.run()


if __name__ == "__main__":
    main()
