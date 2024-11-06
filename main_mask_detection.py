import torch
from segment_anything import SamPredictor, sam_model_registry
import cv2
import numpy as np
import matplotlib.pyplot as plt

class SamPredictorWrapper:
    def __init__(self, model_type="vit_h", checkpoint_path="sam_vit_h.pth", device=None):
        self.model_type = model_type
        self.checkpoint_path = checkpoint_path
        self.device = device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        self.sam = self._load_model()
        self.predictor = SamPredictor(self.sam)

    def _load_model(self):
        sam = sam_model_registry[self.model_type](checkpoint=self.checkpoint_path)
        sam.to(device=self.device)
        return sam

    def predict_mask(self, image_rgb, point=None):
        
        if point is None:
            pos_center = (image_rgb.shape[1] // 2, image_rgb.shape[0] // 2)
            point = np.array([pos_center])
        else:
            point = np.array([point])

        self.predictor.set_image(image_rgb)
        masks, _, _ = self.predictor.predict(point_coords=point, point_labels=np.array([1]))
        
        return image_rgb, masks, point

    def display_mask(self, image_rgb, masks, point, mask_index=1):
        plt.figure(figsize=(10, 10))
        plt.imshow(image_rgb)
        plt.imshow(masks[mask_index], alpha=0.5)
        plt.scatter(point[:, 0], point[:, 1], color='red', s=100, marker='x')
        plt.axis('off')
        plt.show()

def save_mask_as_image(mask, output_path):
    """
    Saves the logical mask as an image at the given path.

    :param mask: Logical mask (numpy array)
    :param output_path: Path to the output file
    """
    mask = mask.astype(np.uint8) * 255
    cv2.imwrite(output_path, mask)
    print(f"Saved mask to {output_path}")

def save_mask(mask, output_path):
    """
    Saves the logical mask as a .npy file at the given path.

    :param mask: Logical mask (numpy array)
    :param output_path: Path to the output file
    """
    np.save(output_path, mask)
    print(f"Saved mask to {output_path}")

def get_click_coordinates(image):
    coordinates = []
    
    def mouse_callback(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            coordinates.append((x, y))
            cv2.destroyAllWindows()
    
    cv2.imshow('Click to select point', image)
    cv2.setMouseCallback('Click to select point', mouse_callback)
    cv2.waitKey(0)
    return coordinates[0] if coordinates else None

# Example usage of the class
if __name__ == "__main__":
    import os
    from tqdm import tqdm

    # Configuration
    model_type = "vit_h"
    checkpoint_path = "sam_vit_h.pth"
    path_dir_images = r".\Materials\1_2mm_brain"
    
    # Supported image formats
    IMAGE_EXTENSIONS = ('.tiff', '.tif', '.png', '.jpg', '.jpeg', '.bmp')
    
    sam_predictor = SamPredictorWrapper(model_type=model_type, checkpoint_path=checkpoint_path)

    # Create Results directory
    path_dir_results = path_dir_images.replace("Materials", "Results")
    if not os.path.exists(path_dir_results):
        os.makedirs(path_dir_results)

    # Get all image files with supported extensions
    image_paths = []
    for file in os.listdir(path_dir_images):
        if file.lower().endswith(IMAGE_EXTENSIONS):
            image_paths.append(os.path.join(path_dir_images, file))
    
    if not image_paths:
        raise ValueError(f"No images found in {path_dir_images} with supported extensions: {IMAGE_EXTENSIONS}")

    # Get first image and let user select point
    first_image = cv2.imread(image_paths[0])
    first_image_rgb = cv2.cvtColor(first_image, cv2.COLOR_BGR2RGB)
    first_image_rgb = cv2.resize(first_image_rgb, (first_image_rgb.shape[1] // 4, first_image_rgb.shape[0] // 4))
    
    # Get click coordinates for the first image
    mask_center = get_click_coordinates(cv2.cvtColor(first_image_rgb, cv2.COLOR_RGB2BGR))
    
    for image_path in tqdm(image_paths, desc="Processing images"):
        image = cv2.imread(image_path)
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image_rgb = cv2.resize(image_rgb, (image_rgb.shape[1] // 4, image_rgb.shape[0] // 4))
        
        image_rgb, masks, point = sam_predictor.predict_mask(image_rgb, mask_center)
        
        # Create output path with original extension replaced by _mask.png
        output_path = os.path.join(
            path_dir_results,
            os.path.splitext(os.path.basename(image_path))[0] + "_mask.png"
        )
        
        save_mask_as_image(masks[1], output_path)
        
        # Update mask center for next frame
        mask_center = np.array([np.mean(np.where(masks[1] == 1)[1]), np.mean(np.where(masks[1] == 1)[0])])