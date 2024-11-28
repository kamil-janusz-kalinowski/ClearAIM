import torch
from segment_anything import SamPredictor, sam_model_registry
import numpy as np


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

    def prepare_points(self, points_positive = None, points_negative=None):
        # Ensure points are Nx2

        points_positive = np.array(points_positive).reshape(-1, 2)  # Reshape to Nx2

        if points_negative is not None:
            points_negative = np.array(points_negative).reshape(-1, 2)  # Reshape to Nx2
            point_coords = np.concatenate([points_positive, points_negative], axis=0)
            point_labels = np.concatenate([np.ones(len(points_positive)), np.zeros(len(points_negative))], axis=0)
        else:
            point_coords = points_positive
            point_labels = np.ones(len(points_positive))

        return point_coords, point_labels, points_positive

    def predict_mask(self, image_rgb, points_positive=None, points_negative=None):
        point_coords, point_labels, points_positive = self.prepare_points(points_positive, points_negative)
        self.predictor.set_image(image_rgb)
        masks, _, _ = self.predictor.predict(point_coords=point_coords, point_labels=point_labels)

        return masks, points_positive

