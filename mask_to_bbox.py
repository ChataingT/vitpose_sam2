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
        --resize
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
    ]
)
logger = logging.getLogger(__name__)


class MaskToBBoxConverter:
    """Convert RLE mask data (produced by the segmentation app) to bounding boxes."""

    def __init__(self, video_path, mask_json_path, output_dir, resize=True):
        """
        Args:
            video_path (str): Path to the video file.
            mask_json_path (str): Path to the merged mask JSON produced by the app.
            output_dir (str): Directory to save the output JSON file.
            resize (bool): Resize masks to match video dimensions before bbox extraction.
        """
        self.video_path = video_path
        self.mask_json_path = mask_json_path
        self.output_dir = output_dir
        self.resize = resize

        self.bbox_dict = {}
        self.missing_data = []
        self.expected_ids = []

        self.vr = None
        self.vr_width = None
        self.vr_height = None

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

            logger.info(f"Video loaded: {len(self.vr)} frames, {self.vr_width}x{self.vr_height}")

        except Exception as e:
            raise RuntimeError(f"Error loading video: {e}")

    def load_mask_data(self):
        """Load mask data from the app's merged JSON and extract available target IDs."""
        logger.info(f"Loading mask data from: {self.mask_json_path}")

        if not os.path.exists(self.mask_json_path):
            raise FileNotFoundError(f"Mask JSON file not found: {self.mask_json_path}")

        try:
            with open(self.mask_json_path, 'r') as f:
                self.mask_data = json.load(f)

            # The app outputs frames as a dict keyed by string frame indices.
            frames = self.mask_data.get('frames', {})
            logger.info(f"Mask data loaded: {len(frames)} frames with mask data")

            self._extract_target_ids()

        except Exception as e:
            raise RuntimeError(f"Error loading mask data: {e}")

    def _extract_target_ids(self):
        """Extract all unique target IDs from the mask data."""
        target_ids_set = set()
        frames = self.mask_data.get('frames', {})

        for frame in frames.values():
            masks = frame.get('masks', {})
            if masks and 'target_ids' in masks:
                target_ids_set.update(masks['target_ids'])

        self.expected_ids = sorted(list(target_ids_set))

        if self.expected_ids:
            logger.info(f"Extracted target IDs: {self.expected_ids}")
        else:
            logger.warning("No target IDs found in mask data")

    @staticmethod
    def mask_to_bbox(mask):
        """
        Convert a binary mask to a bounding box.

        Returns:
            list: [x_min, y_min, x_max, y_max], or [0,0,0,0] if mask is empty.
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
        Decode a column-major RLE mask produced by the segmentation app.

        The app encodes masks in Fortran (column-major) order matching pycoco tools:
          size = [height, width]
          counts = run-length counts starting from the top-left pixel, column by column.

        Args:
            rle (dict): {'size': [h, w], 'counts': [...]}

        Returns:
            np.ndarray: Boolean mask of shape (h, w).
        """
        h, w = rle["size"]
        mask = np.empty(h * w, dtype=bool)
        idx = 0
        parity = False

        for count in rle["counts"]:
            mask[idx: idx + count] = parity
            idx += count
            parity ^= True

        # Encoded in column-major order: reshape as (w, h) then transpose to (h, w).
        return mask.reshape(w, h).T

    def get_mask(self, frame_number):
        """
        Retrieve mask data for a specific frame from the loaded JSON.

        The app stores frames as a dict with string keys ("0", "1", ...).

        Returns:
            tuple: (points, bboxs, masks) or (None, None, None) if frame absent.
        """
        frames = self.mask_data.get('frames', {})
        frame = frames.get(str(frame_number))
        if frame is None:
            return None, None, None

        points = frame.get('points', {})
        bboxs = frame.get('bboxs', {})
        masks = frame.get('masks', {})
        return points, bboxs, masks

    def process_frames(self):
        """Process all video frames and convert masks to bounding boxes."""
        logger.info("Starting frame processing...")
        logger.info(f"Expected target IDs: {self.expected_ids}")
        logger.info(f"Resize masks: {self.resize}")

        total_frames = len(self.vr)
        frames_with_masks = 0

        for frame_id in tqdm(range(total_frames), desc="Processing frames"):
            _, _, masks = self.get_mask(frame_id)

            if not masks:
                # Frame not in the mask JSON — record all expected IDs as missing.
                for expected_id in self.expected_ids:
                    self.missing_data.append((frame_id, expected_id))
                continue

            target_ids = masks.get('target_ids')
            rle_masks = masks.get('rle_masks')

            if not target_ids or not rle_masks:
                for expected_id in self.expected_ids:
                    self.missing_data.append((frame_id, expected_id))
                continue

            frames_with_masks += 1
            self.bbox_dict[str(frame_id)] = {}

            for target_id, rle_mask in zip(target_ids, rle_masks):
                try:
                    mask = self.rle_to_mask(rle_mask)

                    if self.resize:
                        mask = cv2.resize(
                            mask.astype(np.uint8),
                            (self.vr_width, self.vr_height),
                            interpolation=cv2.INTER_NEAREST,
                        )

                    bbox = self.mask_to_bbox(mask)

                    if bbox == [0, 0, 0, 0]:
                        self.missing_data.append((frame_id, target_id))
                        logger.debug(f"Empty bbox for frame {frame_id}, target {target_id}")
                    else:
                        normalized_bbox = [
                            bbox[0] / self.vr_width,
                            bbox[1] / self.vr_height,
                            bbox[2] / self.vr_width,
                            bbox[3] / self.vr_height,
                        ]
                        # Use str keys throughout so the output JSON is consistent.
                        self.bbox_dict[str(frame_id)][str(target_id)] = normalized_bbox

                except Exception as e:
                    logger.error(f"Error processing mask for frame {frame_id}, target {target_id}: {e}")
                    self.missing_data.append((frame_id, target_id))

        logger.info(
            f"Frame processing completed: {total_frames} total, "
            f"{frames_with_masks} with masks, {len(self.missing_data)} missing entries"
        )

    def save_results(self):
        """Save bounding box results and a missing-data report to JSON files."""
        os.makedirs(self.output_dir, exist_ok=True)

        mask_name = Path(self.mask_json_path).stem
        if 'mask' in mask_name:
            bbox_filename = mask_name.replace('mask', 'bbox') + '.json'
            missing_filename = mask_name.replace('mask', 'missing_data') + '.json'
        else:
            bbox_filename = f"{mask_name}_bbox.json"
            missing_filename = f"{mask_name}_missing_data.json"

        bbox_path = os.path.join(self.output_dir, bbox_filename)
        missing_path = os.path.join(self.output_dir, missing_filename)

        logger.info(f"Saving bounding boxes to: {bbox_path}")
        try:
            with open(bbox_path, 'w') as f:
                json.dump(self.bbox_dict, f, indent=2)
            with open(missing_path, 'w') as f:
                json.dump(self.missing_data, f, indent=2)

            total_bboxes = sum(len(v) for v in self.bbox_dict.values())
            logger.info(f"Saved {total_bboxes} bounding boxes")
            logger.info(f"Missing data report: {missing_path}")

        except Exception as e:
            raise RuntimeError(f"Error saving results: {e}")

    def run(self):
        """Run the full mask-to-bbox conversion pipeline."""
        logger.info("Starting mask-to-bbox conversion")
        try:
            self.load_video()
            self.load_mask_data()
            self.process_frames()
            self.save_results()
            logger.info("Conversion completed successfully!")
        except Exception as e:
            logger.error(f"Error during conversion: {e}")
            raise


def main():
    parser = argparse.ArgumentParser(
        description="Convert segmentation app mask JSON to bounding box coordinates.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python mask_to_bbox.py --video_path video.mp4 --mask_json masks.json --output_dir ./output
  python mask_to_bbox.py --video_path video.mp4 --mask_json masks.json --output_dir ./output --no_resize
        """,
    )

    parser.add_argument('--video_path', type=str, required=True, help='Path to the input video file')
    parser.add_argument('--mask_json', type=str, required=True, help='Path to the merged mask JSON from the app')
    parser.add_argument('--output_dir', type=str, required=True, help='Directory to save output files')
    parser.add_argument('--no_resize', action='store_true', help='Do not resize masks to video dimensions')
    parser.add_argument('--verbose', '-v', action='store_true', help='Enable debug logging')

    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    logger.info(f"video_path:  {args.video_path}")
    logger.info(f"mask_json:   {args.mask_json}")
    logger.info(f"output_dir:  {args.output_dir}")
    logger.info(f"resize:      {not args.no_resize}")

    MaskToBBoxConverter(
        video_path=args.video_path,
        mask_json_path=args.mask_json,
        output_dir=args.output_dir,
        resize=not args.no_resize,
    ).run()


if __name__ == "__main__":
    main()
