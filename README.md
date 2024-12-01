# Brain-Detection

## Overview

This repository contains scripts for image processing and analysis, specifically for detecting and analyzing brain images. The main functionalities include mask prediction using the Segment Anything Model (SAM) and subsequent analysis of the predicted masks.

## Requirements

### Python Script (`main_mask_detection.py`)

- Python 3.6+
- PyTorch with CUDA support
- OpenCV
- NumPy
- Matplotlib
- tqdm

### MATLAB Script (`main_analize.m`)

- MATLAB R2018b or later
- Image Processing Toolbox

## Installation

### Python Environment

1. **Install PyTorch with CUDA**:
    ```sh
    pip install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu117
    ```

2. **Install other dependencies**:
    ```sh
    pip install opencv-python numpy matplotlib tqdm
    ```

### MATLAB Environment

Ensure you have MATLAB installed with the Image Processing Toolbox.

## Usage

### Python Script (`main_mask_detection.py`)

This script uses the Segment Anything Model (SAM) to predict masks for images in the `Materials` directory and saves the masks in the `Result` directory.

1. **Prepare the environment**:
    - Ensure you have the SAM model checkpoint file (`sam_vit_h.pth`) in the `models` directory.

2. **Run the script**:
    ```sh
    python main_mask_detection.py
    ```

3. **Script Workflow**:
    - Loads the SAM model.
    - Processes each image in the `Materials` directory.
    - Predicts the mask for each image.
    - Saves the mask in the `Result` directory.

### MATLAB Script (`main_analize.m`)

This script analyzes images and their corresponding masks, calculates measurements such as Weber contrast and area, and generates visualizations.

1. **Prepare the environment**:
    - Ensure you have images in the `Materials` directory and corresponding masks in the `Results` directory.

2. **Run the script**:
    - Open MATLAB.
    - Navigate to the directory containing `main_analize.m`.
    - Run the script:
      ```matlab
      main_analize
      ```

3. **Script Workflow**:
    - Finds and sorts image-mask pairs.
    - Displays images with mask contours.
    - Calculates and displays measurements such as Weber contrast and area.
    - Generates plots for the measurements.
    - Creates an animation showing the processing progress.

## Functions

### Python Script (`main_mask_detection.py`)

- **SamPredictorWrapper**: A wrapper class for the SAM model.
  - `__init__`: Initializes the SAM wrapper with model type, checkpoint path, and device.
  - `_load_model`: Loads the SAM model based on the specified model type and checkpoint path.
  - `_prepare_points`: Prepares and processes positive and negative points for further use.
  - `predict_mask`: Predicts a mask for the given RGB image based on positive and negative points.
- **ImagePathUtility**: Utility class for handling image paths.
  - `get_image_paths`: Retrieves image paths from the specified directory.
  - `save_mask_as_image`: Saves the logical mask as an image at the given path.
- **ImageProcessor**: Class for processing images.
  - `load_image`: Loads an image from the specified path.
  - `rescale`: Rescales the image by the given factor.
  - `invert_mask`: Inverts the given mask.
  - `get_biggest_object_from_mask`: Retrieves the biggest object from the mask.
- **MaskDetector**: Main class for detecting masks.
  - `process_images`: Main processing pipeline for images.
  - `process_single_image`: Processes a single image.
  - `_get_save_path`: Generates the save path for the mask.

### MATLAB Script (`main_analize.m`)

- **findImageMaskPairs**: Finds pairs of image and mask files.
- **processMask**: Processes the mask by converting it to grayscale, resizing, and binarizing.
- **displayImageWithMaskContour**: Displays an image with overlaid mask contours.
- **weberContrast**: Calculates the Weber contrast.
- **sort_image_mask_struct**: Sorts a structure array based on numerical values in the image path.
- **calcMetricsOfAllImages**: Calculates metrics for all images.
- **displayMetrics**: Displays calculated metrics.
- **createAnimationOfObjectDetection**: Creates an animation showing the object detection process.

## Example Usage

### Python Script

```python
from src.mask_detector import MaskDetectorBuilder

def transform_source_path_to_save_path(path_source: str) -> str:
    return path_source.replace("Materials", "Result")

if __name__ == "__main__":
    builder = MaskDetectorBuilder()
    builder.folderpath_source = r".\Materials\250um brain\skrawek 2"
    builder.folderpath_save = transform_source_path_to_save_path(builder.folderpath_source)
    builder.is_display = True
    
    detector = builder.build()
    detector.process_images()