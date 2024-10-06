# Brain-Detection

## Overview

This repository contains two main scripts for image processing and analysis:

1. **main_mask_detection.py**: A Python script that uses the Segment Anything Model (SAM) to predict and save masks for images.
2. **main_analize.m**: A MATLAB script that analyzes images and their corresponding masks, calculates measurements, and generates visualizations.

## Requirements

### Python Script (main_mask_detection.py)

- Python 3.6+
- PyTorch with CUDA support
- OpenCV
- NumPy
- Matplotlib
- tqdm

### MATLAB Script (main_analize.m)

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

### 

main_mask_detection.py



This script uses the Segment Anything Model (SAM) to predict masks for images in the 

Materials

 directory and saves the masks in the 

Results

 directory.

1. **Prepare the environment**:
    - Ensure you have the SAM model checkpoint file (

sam_vit_h.pth

) in the working directory.

2. **Run the script**:
    ```sh
    python main_mask_detection.py
    ```

3. **Script Workflow**:
    - Loads the SAM model.
    - Processes each image in the 

Materials

 directory.
    - Predicts the mask for each image.
    - Saves the mask in the 

Results

 directory.

### 

main_analize.m



This script analyzes images and their corresponding masks, calculates measurements such as Weber contrast and area, and generates visualizations.

1. **Prepare the environment**:
    - Ensure you have images in the 

Materials

 directory and corresponding masks in the 

Results

 directory.

2. **Run the script**:
    - Open MATLAB.
    - Navigate to the directory containing 

main_analize.m

.
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

### Python Script (main_mask_detection.py)

- **SamPredictorWrapper**: A wrapper class for the SAM model.
  - [`__init__`](command:_github.copilot.openSymbolFromReferences?%5B%22%22%2C%5B%7B%22uri%22%3A%7B%22scheme%22%3A%22file%22%2C%22authority%22%3A%22%22%2C%22path%22%3A%22%2Fc%3A%2FPraca%2FQCI%20lab%2Ftemp%2FBrain-Detection%2Fmain_mask_detection.py%22%2C%22query%22%3A%22%22%2C%22fragment%22%3A%22%22%7D%2C%22pos%22%3A%7B%22line%22%3A7%2C%22character%22%3A8%7D%7D%5D%2C%22c856d913-9791-4c4b-9d45-743db57c158b%22%5D "Go to definition"): Initializes the model.
  - [`_load_model`](command:_github.copilot.openSymbolFromReferences?%5B%22%22%2C%5B%7B%22uri%22%3A%7B%22scheme%22%3A%22file%22%2C%22authority%22%3A%22%22%2C%22path%22%3A%22%2Fc%3A%2FPraca%2FQCI%20lab%2Ftemp%2FBrain-Detection%2Fmain_mask_detection.py%22%2C%22query%22%3A%22%22%2C%22fragment%22%3A%22%22%7D%2C%22pos%22%3A%7B%22line%22%3A11%2C%22character%22%3A24%7D%7D%5D%2C%22c856d913-9791-4c4b-9d45-743db57c158b%22%5D "Go to definition"): Loads the SAM model.
  - [`predict_mask`](command:_github.copilot.openSymbolFromReferences?%5B%22%22%2C%5B%7B%22uri%22%3A%7B%22scheme%22%3A%22file%22%2C%22authority%22%3A%22%22%2C%22path%22%3A%22%2Fc%3A%2FPraca%2FQCI%20lab%2Ftemp%2FBrain-Detection%2Fmain_mask_detection.py%22%2C%22query%22%3A%22%22%2C%22fragment%22%3A%22%22%7D%2C%22pos%22%3A%7B%22line%22%3A19%2C%22character%22%3A8%7D%7D%5D%2C%22c856d913-9791-4c4b-9d45-743db57c158b%22%5D "Go to definition"): Predicts the mask for a given image.
  - [`display_mask`](command:_github.copilot.openSymbolFromReferences?%5B%22%22%2C%5B%7B%22uri%22%3A%7B%22scheme%22%3A%22file%22%2C%22authority%22%3A%22%22%2C%22path%22%3A%22%2Fc%3A%2FPraca%2FQCI%20lab%2Ftemp%2FBrain-Detection%2Fmain_mask_detection.py%22%2C%22query%22%3A%22%22%2C%22fragment%22%3A%22%22%7D%2C%22pos%22%3A%7B%22line%22%3A32%2C%22character%22%3A8%7D%7D%5D%2C%22c856d913-9791-4c4b-9d45-743db57c158b%22%5D "Go to definition"): Displays the mask on the image.
- **save_mask_as_image**: Saves the mask as an image file.
- **save_mask**: Saves the mask as a `.npy` file.

### MATLAB Script (main_analize.m)

- **findImageMaskPairs**: Finds pairs of image and mask files.
- **processMask**: Processes the mask by converting it to grayscale, resizing, and binarizing.
- **displayImageWithMaskContour**: Displays an image with overlaid mask contours.
- **weberContrast**: Calculates the Weber contrast.
- **sort_image_mask_struct**: Sorts a structure array based on numerical values in the image path.

## Example Usage

### Python Script

```python
from segment_anything import SamPredictorWrapper

sam_predictor = SamPredictorWrapper(model_type="vit_h", checkpoint_path="sam_vit_h.pth")
image_path = "./Materials/sample_image.png"
image_rgb, masks, point = sam_predictor.predict_mask(image_path)
sam_predictor.display_mask(image_rgb, masks, point)
```

### MATLAB Script

```matlab
clc; clear; close all;
addpath(genpath("./"));

paths = findImageMaskPairs("./Materials", "./Results");
paths = sort_image_mask_struct(paths);

for ind = 1 : length(paths)
    path_img = paths(ind).image_path;
    path_mask = paths(ind).mask_path;
    displayImageWithMaskContour(path_img, path_mask)
end
```

## License

This project is licensed under the MIT License. See the LICENSE file for details.

## Acknowledgements

- The Segment Anything Model (SAM) is used for mask prediction.
- MATLAB Image Processing Toolbox is used for image analysis and visualization.
