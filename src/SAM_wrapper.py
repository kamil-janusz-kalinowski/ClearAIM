import torch
from segment_anything import SamPredictor, sam_model_registry
import numpy as np
import cv2


class SamPredictorWrapper:
    def __init__(self, model_type="vit_h", checkpoint_path="sam_vit_h.pth", device=None):
        """
        Initializes the SAM_wrapper class.

        Args:
            model_type (str): The type of model to use. Default is "vit_h".
            checkpoint_path (str): The path to the model checkpoint file. Default is "sam_vit_h.pth".
            device (str, optional): The device to run the model on. If not specified, it will use "cuda" if available, otherwise "cpu".

        Attributes:
            model_type (str): The type of model being used.
            checkpoint_path (str): The path to the model checkpoint file.
            device (str): The device to run the model on.
            predictor (SamPredictor): The predictor object initialized with the loaded model.
        """
        self.model_type = model_type
        self.checkpoint_path = checkpoint_path
        self.device = device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        sam = self._load_model()
        self.predictor = SamPredictor(sam)

    def _load_model(self) -> torch.nn.Module:
        """
        Loads the SAM model based on the specified model type and checkpoint path.

        This method retrieves the SAM model from the model registry using the provided
        model type and loads the model weights from the specified checkpoint path. The
        model is then moved to the specified device (e.g., CPU or GPU).

        Returns:
            sam: The loaded SAM model.
        """
        sam = sam_model_registry[self.model_type](checkpoint=self.checkpoint_path)
        sam.to(device=self.device)
        return sam

    def _prepare_points(self, points_positive: np.ndarray = None, points_negative: np.ndarray = None) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Prepares and processes positive and negative points for further use.
        Args:
            points_positive (np.ndarray, optional): An array of positive points. Defaults to None.
            points_negative (np.ndarray, optional): An array of negative points. Defaults to None.
        Returns:
            tuple[np.ndarray, np.ndarray, np.ndarray]: A tuple containing:
                - point_coords (np.ndarray): Concatenated array of positive and negative points.
                - point_labels (np.ndarray): Array of labels where 1 represents positive points and 0 represents negative points.
                - points_positive (np.ndarray): Array of positive points reshaped to Nx2.
        """
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

    def predict_mask(self, image_rgb: np.ndarray, points_positive: np.ndarray = None, points_negative: np.ndarray = None, box: np.ndarray = None, mask_input: np.ndarray = None) -> torch.Tensor:
        """
        Predicts a mask for the given RGB image based on positive and negative points.
        Args:
            image_rgb (np.ndarray): The input image in RGB format.
            points_positive (np.ndarray, optional): Array of positive points. Defaults to None.
            points_negative (np.ndarray, optional): Array of negative points. Defaults to None.
        Returns:
            torch.Tensor: The predicted mask as a tensor.
        """
        point_coords, point_labels, points_positive = self._prepare_points(points_positive, points_negative)
        # Rescale mask_input to 1x256x256
        if mask_input is not None:
            mask_input = cv2.resize(mask_input, (256, 256), interpolation=cv2.INTER_LINEAR)
            mask_input = mask_input[np.newaxis, :, :]
        
        self.predictor.set_image(image_rgb)
        masks, _, _ = self.predictor.predict(point_coords=point_coords, point_labels=point_labels, box=box, mask_input=mask_input)

        return masks

