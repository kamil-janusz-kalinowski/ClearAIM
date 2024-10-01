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

    def predict_mask(self, image_path, point=None):
        image = cv2.imread(image_path)
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
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

# Example usage of the class
if __name__ == "__main__":
    import os
    from tqdm import tqdm

    model_type = "vit_h"
    checkpoint_path = "sam_vit_h.pth"
    path_dir_images = r".\Materials\13.08"
    
    sam_predictor = SamPredictorWrapper(model_type=model_type, checkpoint_path=checkpoint_path)

    # Create Results directory
    path_dir_results = path_dir_images.replace("Materials", "Results")
    if not os.path.exists(path_dir_results):
        os.makedirs(path_dir_results)

    image_paths = [os.path.join(path_dir_images, file) for file in os.listdir(path_dir_images) if file.endswith(".png")]

    mask_center = None
    for image_path in tqdm(image_paths, desc="Processing images"):
        if mask_center is not None:
            image_rgb, masks, point = sam_predictor.predict_mask(image_path, mask_center)
        else:
            image_rgb, masks, point = sam_predictor.predict_mask(image_path)

        save_mask_as_image(masks[1], image_path.replace("Materials", "Results").replace(".png", "_mask.png"))

        # Get center of mask
        mask_center = np.array([np.mean(np.where(masks[1] == 1)[1]), np.mean(np.where(masks[1] == 1)[0])])
