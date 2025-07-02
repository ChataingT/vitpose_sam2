#!/bin/bash

# Example script to run mmpose_inference.py with the files from the current workspace
# This demonstrates how to use the script with the specific files available

echo "Running MMPose inference with workspace files..."

python mmpose_inference.py \
    /home/share/schaer2/idtracking_keypoint/vitpose_sam2/mmpose/configs/body_2d_keypoint/topdown_heatmap/coco/td-hm_ViTPose-huge_8xb64-210e_coco-256x192.py \
    /home/share/schaer2/idtracking_keypoint/vitpose_config/td-hm_ViTPose-huge_8xb64-210e_coco-256x192-e32adcd4_20230314.pth \
    --input /home/share/schaer2/idtracking_keypoint/input/20200505155636_20200505174320_0_converted_small.mp4 \
    --bboxes /home/share/schaer2/idtracking_keypoint/output/20200505155636_20200505174320_0_converted_small_bbox.json \
    --output-root /home/share/schaer2/idtracking_keypoint/output \
    --device cuda \
    --skeleton-style openpose \
    --save-predictions \
    --verbose

echo "MMPose inference completed! Check the output directory for results."
