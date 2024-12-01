import cv2
import numpy as np
import os
from tqdm import tqdm
from src.SAM_wrapper import SamPredictorWrapper
from sklearn.cluster import KMeans
from src.utility import save_mask_as_image, get_click_coordinates, get_negative_points

class MaskDetectorConfig:
    def __init__(self):
        # Domyślne wartości
        self.model_type = "vit_h"
        self.checkpoint_path = "models/sam_vit_h.pth"
        self.is_display = True
        self.downscale_factor = 5
        self.image_extensions = ('.tiff', '.tif', '.png', '.jpg', '.jpeg', '.bmp')
        self.folderpath_source = None
        self.folderpath_save = None
        self.num_positive_points = 2
        self.num_negative_points = 12
    
class MaskDetectorBuilder:
    def __init__(self):
        self._config = MaskDetectorConfig()
    
    @property
    def model_type(self):
        return self._config.model_type
    
    @model_type.setter
    def model_type(self, value):
        self._config.model_type = value
    
    @property
    def checkpoint_path(self):
        return self._config.checkpoint_path
    
    @checkpoint_path.setter
    def checkpoint_path(self, value):
        self._config.checkpoint_path = value
    
    @property
    def is_display(self):
        return self._config.is_display
    
    @is_display.setter
    def is_display(self, value):
        self._config.is_display = value
    
    @property
    def downscale_factor(self):
        return self._config.downscale_factor
    
    @downscale_factor.setter
    def downscale_factor(self, value):
        self._config.downscale_factor = value
    
    @property
    def image_extensions(self):
        return self._config.image_extensions
    
    @image_extensions.setter
    def image_extensions(self, value):
        self._config.image_extensions = value
    
    @property
    def folderpath_source(self):
        return self._config.folderpath_source
    
    @folderpath_source.setter
    def folderpath_source(self, path: str):
        self._config.folderpath_source = path
    
    @property
    def folderpath_save(self):
        return self._config.folderpath_save
    
    @folderpath_source.setter
    def folderpath_save(self, path: str):
        self._config.folderpath_save = path    
    
    @property
    def num_positive_points(self):
        return self._config.num_positive_points
    
    @num_positive_points.setter
    def num_positive_points(self, value):
        self._config.num_positive_points = value
        
    @property
    def num_negative_points(self):
        return self._config.num_negative_points
    
    @num_negative_points.setter
    def num_negative_points(self, value):
        self._config.num_negative_points = value
    
    def build(self):
        return MaskDetector(self._config)
    
class ImagePathUtility:
    @staticmethod
    def get_image_paths(input_dir: str, image_extensions: list) -> list:
        image_paths = [
            os.path.join(input_dir, f) for f in os.listdir(input_dir)
            if f.lower().endswith(image_extensions)
        ]
        if not image_paths:
            raise ValueError(
                f"No images found in {input_dir} with supported extensions: {image_extensions}"
            )
        print(f"Found {len(image_paths)} images in {input_dir}")
        return image_paths
    
    @staticmethod
    def save_mask_as_image(mask: np.ndarray, output_path: str) -> None:
        """
        Saves the logical mask as an image at the given path.

        :param mask: Logical mask (numpy array)
        :param output_path: Path to the output file
        """
        mask = mask.astype(np.uint8)*255
        cv2.imwrite(output_path, mask)
        print(f"Saved mask to {output_path}")

class ImageProcessor:
    @staticmethod
    def load_image(image_path: str) -> np.ndarray:
        image = cv2.imread(image_path)
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        return image_rgb
    
    @staticmethod
    def rescale(image: np.ndarray, rescale_factor: float) -> np.ndarray:
        return cv2.resize(
            image, 
            (
            int(image.shape[1] // (1/rescale_factor)), 
            int(image.shape[0] // (1/rescale_factor))
             )
        )
        
    @staticmethod
    def invert_mask(mask: np.ndarray) -> np.ndarray:
        mask = mask * 255
        return cv2.bitwise_not(mask)/255
    
    @staticmethod
    def get_biggest_object_from_mask(mask: np.ndarray) -> np.ndarray:
        # Get the biggest mask object
        mask = mask.astype(np.uint8)*255
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        mask_final = np.zeros_like(mask)
        if contours:
            biggest_contour = max(contours, key=cv2.contourArea)
            cv2.drawContours(mask_final, [biggest_contour], -1, 1, cv2.FILLED)
        return mask_final
            
class MaskVisualizer:
    
    @staticmethod
    def _visualize_result(image, mask, points_positive, points_negative=None, alpha=0.5, point_radius = 5, point_thickness = 2):
        vis_image = image.copy()
        mask_colored = np.zeros_like(image)
        mask_colored[mask == 255] = [0, 255, 0]
        vis_image = cv2.addWeighted(vis_image, 1, mask_colored, alpha, 0)
        
        if points_positive is not None:
            for point in points_positive:
                cv2.circle(vis_image, 
                        (int(point[0]), int(point[1])), 
                        point_radius,
                        (0, 255, 0),
                        point_thickness)
        
        if points_negative is not None:
            for point in points_negative:
                cv2.circle(vis_image, 
                        (int(point[0]), int(point[1])), 
                        point_radius,
                        (255, 0, 0),
                        point_thickness)
        
        return vis_image
    
    @staticmethod
    def display_image_with_data(image_rgb, mask = None, points_positive = None, points_negative = None, time_display = 1000) -> None:
        mask = mask.astype(np.uint8) * 255
        # Visualize results
        vis_image = MaskVisualizer._visualize_result(image_rgb, mask, points_positive, points_negative)
        # Display visualization 
        cv2.imshow('Mask Visualization', cv2.cvtColor(vis_image, cv2.COLOR_RGB2BGR))
        cv2.waitKey(time_display)
        cv2.destroyAllWindows()

def distribute_points_using_kmeans(mask, num_points):
    """
    Distribute a specified number of points optimally on a binary mask using k-means clustering.

    Args:
        mask (np.ndarray): Binary mask array where points should be distributed.
        num_points (int): Number of points to distribute.

    Returns:
        np.ndarray: Array of [x, y] coordinates for the distributed points.
    """
    # Get valid points from the mask
    y_indices, x_indices = np.where(mask == 1)
    if len(y_indices) == 0:
        return np.array([])

    # Combine coordinates into a single array for clustering
    coordinates = np.column_stack((x_indices, y_indices))

    # Use k-means clustering to find clusters
    kmeans = KMeans(n_clusters=min(num_points, len(coordinates)), random_state=0, n_init='auto')
    kmeans.fit(coordinates)
    cluster_centers = kmeans.cluster_centers_

    # Round cluster centers to nearest integer coordinates
    distributed_points = np.round(cluster_centers).astype(int)
    
    return distributed_points

class ImageProcessingState:
    def __init__(self, points_positive = None, points_negative = None, mask_current = None, mask_previous = None):
        self.points_positive = points_positive
        self.points_negative = points_negative
        self.mask_current = mask_current
        self.mask_previous = mask_previous

class MaskDetector:
    def __init__(self, config: MaskDetectorConfig):
        self._model_type = config.model_type
        self._checkpoint_path = config.checkpoint_path
        self.is_display = config.is_display
        self.DOWNSCALE_FACTOR = config.downscale_factor
        self.IMAGE_EXTENSIONS = config.image_extensions
        self.folderpath_source = config.folderpath_source   
        self.folderpath_save = config.folderpath_save
        self.num_positive_points = config.num_positive_points
        self.num_negative_points = config.num_negative_points
        
        self._sam_predictor = SamPredictorWrapper(
            model_type=self._model_type,
            checkpoint_path=self._checkpoint_path
        )

    def process_images(self):
        """Main processing pipeline"""
        paths_image = ImagePathUtility.get_image_paths(self.folderpath_source, self.IMAGE_EXTENSIONS)
        os.makedirs(self.folderpath_save, exist_ok=True)
        
        # Initialize with first image
        first_image = ImageProcessor.load_image(paths_image[0])
        first_image = ImageProcessor.rescale(first_image, 1/self.DOWNSCALE_FACTOR)
        
        points_positive = get_click_coordinates(
            cv2.cvtColor(first_image, cv2.COLOR_RGB2BGR)
        )
        
        state = ImageProcessingState(points_positive, None, None, None)
        
        # Process all images
        for path_image in tqdm(paths_image, desc="Processing images"):
            image_rgb = ImageProcessor.load_image(path_image)
            state = self.process_single_image(image_rgb, state)
            
            # Save mask and update center
            output_path = self._get_save_path(path_image)
            ImagePathUtility.save_mask_as_image(state.mask_current, output_path)
            
    def process_single_image(self, image_rgb: np.ndarray, state: ImageProcessingState) -> ImageProcessingState:
        """Process a single image"""
        image_rgb = ImageProcessor.rescale(image_rgb, 1/self.DOWNSCALE_FACTOR)        
                
        # Predict mask
        masks = self._sam_predictor.predict_mask(
            image_rgb, 
            state.points_positive, 
            state.points_negative,
            mask_input=state.mask_current
        )
        
        state.mask_current = ImageProcessor.get_biggest_object_from_mask(masks[1])
        
        if self.is_display:
            MaskVisualizer.display_image_with_data(image_rgb, state.mask_current, state.points_positive, state.points_negative)
            
        if state.mask_previous is not None:
            mask_mutual = cv2.bitwise_and(state.mask_current, state.mask_previous)   
        else:
            mask_mutual = state.mask_current 
            
        # get points for next frame
        state.points_positive = distribute_points_using_kmeans(mask_mutual, num_points=self.num_positive_points)
        state.points_negative = distribute_points_using_kmeans(ImageProcessor.invert_mask(state.mask_current), num_points=self.num_negative_points)

        state.mask_previous = state.mask_current
        
        return state

    def _get_save_path(self, path_image):
        # Get the directory and filename from the input path
        directory, filename = os.path.split(path_image)
        
        # Split the filename into name and extension
        name, ext = os.path.splitext(filename)
        
        # Create a new filename for the mask
        mask_filename = f"{name}_mask{ext}"
        
        # Combine the directory and new filename to create the output path
        output_path = os.path.join(self.folderpath_save, mask_filename)
        
        return output_path
    