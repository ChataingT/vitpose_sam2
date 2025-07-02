# Mask to Bounding Box Converter

This script converts mask data from JSON files to normalized bounding box coordinates. It's a command-line version of the functionality originally implemented in the `mask_to_bbox.ipynb` notebook.

## Features

- **Video Processing**: Loads video files using decord for efficient frame access
- **Mask Conversion**: Converts Run-Length Encoded (RLE) masks to binary masks
- **Bounding Box Extraction**: Computes tight bounding boxes around masked regions
- **Scaling Support**: Handles videos with different resolutions through scaling ratios
- **Normalization**: Outputs normalized bounding box coordinates (0-1 range)
- **Missing Data Tracking**: Reports frames/targets with missing or invalid data
- **Progress Tracking**: Shows progress bars and detailed logging
- **Flexible Configuration**: Supports various command-line options

## Installation

1. Install the required dependencies:
```bash
pip install -r requirements_mask_to_bbox.txt
```

## Usage

### Basic Usage
```bash
python mask_to_bbox.py \
    --video_path /path/to/video.mp4 \
    --mask_json /path/to/mask_data.json \
    --output_dir ./output
```

### Advanced Usage
```bash
# With source video for proper scaling
python mask_to_bbox.py \
    --video_path /home/share/schaer2/idtracking_keypoint/input/20200505155636_20200505174320_0_converted_25fps.mp4 \
    --mask_json /home/share/schaer2/idtracking_keypoint/output/20200505155636_20200505174320_0_converted_small_mask_0-end.json \
    --output_dir /home/share/schaer2/idtracking_keypoint/output \
    --verbose
```

### Command Line Options

- `--video_path`: Path to the input video file (required)
- `--mask_json`: Path to the JSON file containing mask data (required)
- `--output_dir`: Directory to save the output files (required)
- `--no_resize`: Don't resize masks to match video dimensions
- `--verbose`, `-v`: Enable verbose logging

**Note**: Target IDs are automatically extracted from the mask data, eliminating the need to specify them manually.

## Output Files

The script generates two output files in the specified output directory:

1. **Bounding Box File**: Contains normalized bounding box coordinates
   - **Naming**: Based on mask JSON filename with 'mask' replaced by 'bbox'
   - **Example**: `video_mask_0-end.json` → `video_bbox_0-end.json`
   - **Format**:
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

2. **Missing Data Report**: Lists frames and target IDs with missing data
   - **Naming**: Based on mask JSON filename with 'mask' replaced by 'missing_data'
   - **Example**: `video_mask_0-end.json` → `video_missing_data_0-end.json`
   - **Format**:
   ```json
   [
     [frame_id, target_id],
     [frame_id, target_id]
   ]
   ```

## Logging

The script provides comprehensive logging:

- **Console Output**: Real-time progress and status updates
- **Log File**: Detailed log saved to `mask_to_bbox.log`
- **Progress Bars**: Visual progress tracking for frame processing

### Example Log Output
```
2025-07-01 10:30:15,123 - INFO - Starting mask-to-bbox conversion process
2025-07-01 10:30:15,124 - INFO - Loading video from: /path/to/video.mp4
2025-07-01 10:30:15,456 - INFO - Video loaded successfully:
2025-07-01 10:30:15,456 - INFO -   - Frames: 1500
2025-07-01 10:30:15,456 - INFO -   - Dimensions: 1920 x 1080
2025-07-01 10:30:15,789 - INFO - Mask data loaded successfully with 1500 frames
2025-07-01 10:30:16,012 - INFO - Starting frame processing...
Processing frames: 100%|██████████| 1500/1500 [00:45<00:00, 33.12it/s]
2025-07-01 10:31:01,234 - INFO - Frame processing completed:
2025-07-01 10:31:01,234 - INFO -   - Total frames: 1500
2025-07-01 10:31:01,234 - INFO -   - Frames with masks: 1450
2025-07-01 10:31:01,234 - INFO -   - Missing data entries: 150
2025-07-01 10:31:01,567 - INFO - Results saved successfully:
2025-07-01 10:31:01,567 - INFO -   - Total bounding boxes: 4350
```

## Algorithm Details

1. **Video Loading**: Uses decord for efficient video frame access
2. **Mask Data Loading**: Loads mask data and automatically extracts available target IDs
3. **Target ID Extraction**: Scans all frames to find unique target IDs present in the data
4. **Mask Decoding**: Converts RLE format to binary masks
5. **Scaling**: Optionally resizes masks to match target video dimensions
6. **Bbox Extraction**: Finds tight bounding rectangles around mask regions
7. **Normalization**: Converts pixel coordinates to normalized (0-1) range
8. **Data Validation**: Tracks and reports missing or invalid data

## Key Improvements

- **Automatic ID Detection**: No need to manually specify expected target IDs
- **Robust Processing**: Handles variable numbers of targets across frames
- **Data-Driven**: Works with any mask data regardless of target ID configuration

## Error Handling

The script includes comprehensive error handling:
- File existence validation
- Video loading error recovery
- Mask processing error handling
- Output directory creation
- Detailed error logging

## Performance

- **Memory Efficient**: Processes frames sequentially to minimize memory usage
- **Progress Tracking**: Real-time progress updates with estimated completion time
- **Batch Processing**: Handles large videos efficiently
- **Error Recovery**: Continues processing even if individual frames fail

## Dependencies

- `opencv-python`: Image processing and video handling
- `numpy`: Numerical computations
- `tqdm`: Progress bars
- `decord`: Efficient video reading
