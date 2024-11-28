import cv2
import numpy as np
import os
from tqdm import tqdm
from src.SAM_wrapper import SamPredictorWrapper
from src.utility import save_mask_as_image, get_click_coordinates, get_negative_points

class MaskDetector:
    DOWNSCALE_FACTOR = 4
    IMAGE_EXTENSIONS = ('.tiff', '.tif', '.png', '.jpg', '.jpeg', '.bmp')
    
    def __init__(self, model_type="vit_h", checkpoint_path="sam_vit_h.pth"):
        self.sam_predictor = SamPredictorWrapper(
            model_type=model_type, 
            checkpoint_path=checkpoint_path
        )

    def _setup_directories(self, input_dir):
        """Create results directory if it doesn't exist"""
        results_dir = input_dir.replace("Materials", "Results")
        os.makedirs(results_dir, exist_ok=True)
        return results_dir

    def _get_image_paths(self, input_dir):
        """Get all valid image paths from input directory"""
        image_paths = [
            os.path.join(input_dir, f) for f in os.listdir(input_dir)
            if f.lower().endswith(self.IMAGE_EXTENSIONS)
        ]
        if not image_paths:
            raise ValueError(
                f"No images found in {input_dir} with supported extensions: {self.IMAGE_EXTENSIONS}"
            )
        
        print(f"Found {len(image_paths)} images in {input_dir}")
        return image_paths

    def _preprocess_image(self, image_path):
        """Load and preprocess image"""
        image = cv2.imread(image_path)
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        return cv2.resize(
            image_rgb, 
            (image_rgb.shape[1] // self.DOWNSCALE_FACTOR, 
             image_rgb.shape[0] // self.DOWNSCALE_FACTOR)
        )

    def _get_output_path(self, results_dir, image_path):
        """Generate output path for mask"""
        return os.path.join(
            results_dir,
            f"{os.path.splitext(os.path.basename(image_path))[0]}_mask.png"
        )

    def process_images(self, input_dir):
        """Main processing pipeline"""
        results_dir = self._setup_directories(input_dir)
        image_paths = self._get_image_paths(input_dir)

        # Initialize with first image
        first_image = self._preprocess_image(image_paths[0])
        point_center = get_click_coordinates(
            cv2.cvtColor(first_image, cv2.COLOR_RGB2BGR)
        )
        points_negative = None

        # Process all images
        for image_path in tqdm(image_paths, desc="Processing images"):
            image_rgb = self._preprocess_image(image_path)
            
            if points_negative is None:
                masks, point = self.sam_predictor.predict_mask(image_rgb, point_center)
                points_negative = get_negative_points(masks[1], num_points=10, min_distance=50)
            else:
                masks, point_center = self.sam_predictor.predict_mask(
                    image_rgb, 
                    point_center, 
                    points_negative
                )
                points_negative = get_negative_points(masks[1], num_points=5, min_distance=50)

            # Save mask and update center
            output_path = self._get_output_path(results_dir, image_path)
            save_mask_as_image(masks[1], output_path)
            point_center = np.array([
                np.mean(np.where(masks[1] == 1)[1]), 
                np.mean(np.where(masks[1] == 1)[0])
            ])
            
def main():
    detector = MaskDetector()
    detector.process_images(r".\Materials\250um brain\skrawek 2")

if __name__ == "__main__":
    main()