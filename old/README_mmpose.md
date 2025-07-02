# MMPose Inference Script

This script performs 2D human pose estimation using MMPose with ViTPose models. It processes videos frame by frame using pre-computed bounding boxes to detect and extract keypoint coordinates for multiple individuals.

## Important Prerequisites

### MMPose Custom Modification Required

**CRITICAL**: This script requires a modification to MMPose's source code to work properly. You must modify the file `mmpose/structures/utils.py` at line 128.

**Original code:**
```python
def split_instances(instances: InstanceData) -> List[InstanceData]:
    """Convert instances into a list where each element is a dict that contains
    information about one instance."""
    results = []

    # return an empty list if there is no instance detected by the model
    if instances is None:
        return results

    for i in range(len(instances.keypoints)):
        result = dict(
            keypoints=instances.keypoints[i].tolist(),
            keypoint_scores=instances.keypoint_scores[i].tolist(),
        )
        if 'bboxes' in instances:
            result['bbox'] = instances.bboxes[i].tolist(),
            if 'bbox_scores' in instances:
                result['bbox_score'] = instances.bbox_scores[i]
        results.append(result)

    return results
```

**Modified code (add the keypoints_label line):**
```python
def split_instances(instances: InstanceData) -> List[InstanceData]:
    """Convert instances into a list where each element is a dict that contains
    information about one instance."""
    results = []

    # return an empty list if there is no instance detected by the model
    if instances is None:
        return results

    for i in range(len(instances.keypoints)):
        result = dict(
            keypoints=instances.keypoints[i].tolist(),
            keypoint_scores=instances.keypoint_scores[i].tolist(),
            keypoints_label=instances.keypoints_label[i].tolist()  # ADD THIS LINE
        )
        if 'bboxes' in instances:
            result['bbox'] = instances.bboxes[i].tolist(),
            if 'bbox_scores' in instances:
                result['bbox_score'] = instances.bbox_scores[i]
        results.append(result)

    return results
```

## Features

- **Video Processing**: Processes video files frame by frame for pose estimation
- **Bounding Box Integration**: Uses pre-computed bounding boxes for targeted pose detection
- **Multi-Person Support**: Handles multiple individuals per frame with ID tracking
- **Flexible Output**: Supports prediction saving, video output, and real-time display
- **Skeleton Styles**: Supports different skeleton visualization styles (MMPose, OpenPose)
- **GPU Acceleration**: CUDA support for faster inference
- **Progress Tracking**: Real-time progress bars and detailed logging
- **Error Handling**: Robust error handling with detailed logging

## Installation

1. Install MMPose and dependencies:
```bash
# Install MMPose (follow official installation guide)
pip install openmim
mim install mmengine
mim install "mmcv>=2.0.1"
mim install "mmdet>=3.1.0"
mim install "mmpose>=1.1.0"

# Install additional requirements
pip install -r requirements_mmpose.txt
```

2. Apply the required MMPose modification (see above)

3. Download ViTPose model weights if needed

## Usage

### Basic Usage
```bash
python mmpose_inference.py CONFIG CHECKPOINT --input VIDEO --bboxes BBOXES --output-root OUTPUT
```

### Advanced Usage
```bash
python mmpose_inference.py \
    configs/body_2d_keypoint/topdown_heatmap/coco/td-hm_ViTPose-huge_8xb64-210e_coco-256x192.py \
    td-hm_ViTPose-huge_8xb64-210e_coco-256x192-e32adcd4_20230314.pth \
    --input /path/to/video.mp4 \
    --bboxes /path/to/bboxes.json \
    --output-root ./output \
    --device cuda \
    --skeleton-style openpose \
    --save-predictions \
    --save-video \
    --draw-bbox \
    --verbose
```

### Example with Workspace Files
```bash
./run_mmpose_example.sh
```

### Command Line Options

#### Required Arguments
- `pose_config`: Path to MMPose configuration file
- `pose_checkpoint`: Path to model checkpoint file
- `--input`: Path to input video file
- `--bboxes`: Path to JSON file containing bounding boxes
- `--output-root`: Root directory for output files

#### Model Settings
- `--device`: Device for inference (default: cuda:0)
- `--kpt-thr`: Keypoint confidence threshold (default: 0.3)

#### Visualization Settings
- `--skeleton-style`: Skeleton style (mmpose/openpose, default: mmpose)
- `--radius`: Keypoint radius for visualization (default: 3)
- `--thickness`: Line thickness for visualization (default: 1)
- `--alpha`: Transparency of bounding boxes (default: 0.8)

#### Output Options
- `--save-predictions`: Save prediction results to JSON
- `--save-video`: Save visualization video
- `--show`: Display results in real-time
- `--show-interval`: Display interval in milliseconds

#### Drawing Options
- `--draw-heatmap`: Draw prediction heatmaps
- `--draw-bbox`: Draw bounding boxes
- `--show-kpt-idx`: Show keypoint indices

## Input Format

### Bounding Boxes JSON
The bounding boxes file should contain normalized coordinates (0-1 range):
```json
{
  "0": {
    "0": [x_min, y_min, x_max, y_max],
    "1": [x_min, y_min, x_max, y_max]
  },
  "1": {
    "0": [x_min, y_min, x_max, y_max]
  }
}
```

Where:
- Frame IDs are string keys ("0", "1", "2", ...)
- Object IDs are string keys within each frame
- Coordinates are normalized (0.0 to 1.0)

## Output Files

### Prediction Results (`results_*.json`)
Contains keypoint predictions for each frame:
```json
{
  "meta_info": {
    "dataset_name": "coco",
    "keypoint_info": {...}
  },
  "instance_info": [
    {
      "frame_id": 0,
      "instances": [
        {
          "keypoints": [[x1, y1], [x2, y2], ...],
          "keypoint_scores": [s1, s2, ...],
          "keypoints_label": [id],
          "bbox": [x_min, y_min, x_max, y_max],
          "bbox_score": 0.95
        }
      ]
    }
  ]
}
```

### Visualization Video
If `--save-video` is enabled, generates an MP4 file with:
- Keypoint overlays
- Skeleton connections
- Optional bounding boxes
- Configurable visualization style

## Logging

The script provides comprehensive logging:
- **Console Output**: Real-time progress and status updates
- **Log File**: Detailed log saved to `mmpose_inference.log`
- **Progress Bars**: Visual progress tracking for video processing

### Example Log Output
```
2025-07-01 14:30:15,123 - INFO - Starting MMPose inference
2025-07-01 14:30:15,124 - INFO - Loading pose estimation model...
2025-07-01 14:30:16,456 - INFO - Pose estimator loaded successfully
2025-07-01 14:30:16,789 - INFO - Loaded bounding boxes for 1500 frames
2025-07-01 14:30:17,012 - INFO - Processing video: /path/to/video.mp4
Processing frames: 100%|██████████| 1500/1500 [02:15<00:00, 11.08it/s]
2025-07-01 14:32:32,234 - INFO - Processed 1500 frames
2025-07-01 14:32:32,567 - INFO - Saved predictions for 1500 frames
2025-07-01 14:32:32,567 - INFO - Total instances detected: 4350
```

## Performance Tips

1. **GPU Usage**: Use CUDA device for significantly faster processing
2. **Batch Processing**: Process multiple videos in sequence
3. **Memory Management**: Monitor GPU memory usage for large videos
4. **Model Selection**: Choose appropriate ViTPose model size based on accuracy/speed requirements

## Troubleshooting

### Common Issues

1. **MMPose Import Error**: Ensure MMPose is properly installed and the custom modification is applied
2. **CUDA Out of Memory**: Reduce batch size or use a smaller model
3. **Bounding Box Format Error**: Ensure bboxes are in normalized coordinates
4. **Model Loading Error**: Verify checkpoint file path and compatibility

### Dependencies
- MMPose >= 1.0.0
- MMCV >= 2.0.0
- MMEngine >= 0.7.0
- OpenCV >= 4.5.0
- NumPy >= 1.20.0
- TQDM >= 4.60.0

## Integration with Pipeline

This script is designed to work with the output from `mask_to_bbox.py`:
1. Generate masks using SAM2
2. Convert masks to bounding boxes using `mask_to_bbox.py`
3. Perform pose estimation using `mmpose_inference.py`
4. Further processing and analysis

## Model Support

Currently tested with:
- ViTPose-Base
- ViTPose-Large
- ViTPose-Huge

Other MMPose models should work with appropriate configuration files.
