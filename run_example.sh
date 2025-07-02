#!/bin/bash
#SBATCH --cpus-per-task=24
#SBATCH --job-name=vitpose
#SBATCH --chdir=/home/share/schaer2/idtracking_keypoint
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --time=0-12:00:00
#SBATCH --partition=shared-gpu,private-schaer-gpu
#SBATCH --mem=64G
#SBATCH --gpus=1
#SBATCH --output=logs/slurm-%J.out
#SBATCH --mail-type=END,FAIL

echo `pwd -P`
echo `nvidia-smi -L`

# Execution parameters
PROJROOT=/home/share/schaer2/idtracking_keypoint

# Load environment
module load GCCcore/10.2.0 Python/3.8.6 cuDNN/8.9.2.26-CUDA-12.1.1
source $PROJROOT/vitpose_env/bin/activate

##### Parameters
# Adapt those to your setup

# Common paths
OUTPUT_DIR="/home/share/schaer2/idtracking_keypoint/output"

# Full paths to video files
VIDEO_PATH_USED_IN_SAM2=/home/share/schaer2/idtracking_keypoint/input/bedroom.mp4   # Here is the video used to extract the mask. We need it to normalize the bounding boxes
INPUT_VIDEO_PATH_FOR_SKELETON=/home/share/schaer2/idtracking_keypoint/input/bedroom.mp4  # Here is the video used for skeleton extraction. The size of frame can be different but the fps NOT.

# Mask to BBox parameters
MASK_JSON="/home/share/schaer2/idtracking_keypoint/output/bedroom_mask_0-end.json"



















################## Do not modify below this line ##################

# Generate bbox filename from mask filename (replace 'mask' with 'bbox')
MASK_BASENAME=$(basename "${MASK_JSON}" .json)
BBOX_FILENAME="${MASK_BASENAME/mask/bbox}.json"
BBOX_OUTPUT_FILE="${OUTPUT_DIR}/${BBOX_FILENAME}"


VITPOSE_SAM2_DIR="${PROJROOT}/vitpose_sam2"
VITPOSE_CONFIG_DIR="${PROJROOT}/vitpose_config"

# MMPose parameters
POSE_CONFIG="${VITPOSE_SAM2_DIR}/mmpose/configs/body_2d_keypoint/topdown_heatmap/coco/td-hm_ViTPose-huge_8xb64-210e_coco-256x192.py"
POSE_CHECKPOINT="${VITPOSE_CONFIG_DIR}/td-hm_ViTPose-huge_8xb64-210e_coco-256x192-e32adcd4_20230314.pth"
DEVICE="cuda"
SKELETON_STYLE="openpose"

# Script paths
MASK_TO_BBOX_SCRIPT="${VITPOSE_SAM2_DIR}/mask_to_bbox.py"
MMPOSE_SCRIPT="${VITPOSE_SAM2_DIR}/mmpose_inference.py"

# Logging options
VERBOSE_FLAG="--verbose"

echo "=== Configuration ==="
echo "Project root: ${PROJROOT}"
echo "Input video used in sam2: ${VIDEO_PATH_USED_IN_SAM2}"
echo "Input video for the skeleton: ${INPUT_VIDEO_PATH_FOR_SKELETON}"
echo "Mask JSON: ${MASK_JSON}"
echo "Generated bbox file: ${BBOX_OUTPUT_FILE}"
echo "Output directory: ${OUTPUT_DIR}"
echo "Device: ${DEVICE}"
echo "Skeleton style: ${SKELETON_STYLE}"
echo "====================="

echo "Running mask_to_bbox.py with workspace files..."

python "${MASK_TO_BBOX_SCRIPT}" \
    --video_path "${VIDEO_PATH_USED_IN_SAM2}" \
    --mask_json "${MASK_JSON}" \
    --output_dir "${OUTPUT_DIR}" \
    ${VERBOSE_FLAG}

echo "Conversion completed! Check the output directory for results."

echo "Running MMPose inference with workspace files..."

python "${MMPOSE_SCRIPT}" \
    "${POSE_CONFIG}" \
    "${POSE_CHECKPOINT}" \
    --input "${INPUT_VIDEO_PATH_FOR_SKELETON}" \
    --bboxes "${BBOX_OUTPUT_FILE}" \
    --output-root "${OUTPUT_DIR}" \
    --device "${DEVICE}" \
    --skeleton-style "${SKELETON_STYLE}" \
    --save-predictions \
    ${VERBOSE_FLAG}

echo "MMPose inference completed! Check the output directory for results."