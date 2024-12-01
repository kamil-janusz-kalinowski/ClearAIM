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
    
    def build(self):
        return MaskDetector(self._config)
    
class DirectoryManager:
    @staticmethod
    def setup_directories(input_dir):
        results_dir = input_dir.replace("Materials", "Results")
        os.makedirs(results_dir, exist_ok=True)
        return results_dir

    @staticmethod
    def get_image_paths(input_dir, image_extensions):
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

class ImageProcessor:
    @staticmethod
    def load_image(image_path):
        image = cv2.imread(image_path)
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        return image_rgb
    
    @staticmethod
    def rescale(image: np.ndarray, rescale_factor) -> np.ndarray:
        return cv2.resize(
            image, 
            (
            int(image.shape[1] // (1/rescale_factor)), 
            int(image.shape[0] // (1/rescale_factor))
             )
        )


class MaskVisualizer:
    
    @staticmethod
    def _visualize_result(image, mask, points_positive, points_negative=None, alpha=0.5, point_radius = 5, point_thickness = 2):
        if isinstance(image, np.ndarray) and image.dtype != np.uint8:
            image = (image * 255).astype(np.uint8)
        
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
        # Visualize results
        vis_image = MaskVisualizer._visualize_result(image_rgb, mask, points_positive, points_negative)
        # Display visualization 
        cv2.imshow('Mask Visualization', cv2.cvtColor(vis_image, cv2.COLOR_RGB2BGR))
        cv2.waitKey(time_display)
        cv2.destroyAllWindows()



class MaskDetector:
    def __init__(self, config: MaskDetectorConfig):
        self._model_type = config.model_type
        self._checkpoint_path = config.checkpoint_path
        self.is_display = config.is_display
        self.DOWNSCALE_FACTOR = config.downscale_factor
        self.IMAGE_EXTENSIONS = config.image_extensions
        self.folderpath_source = config.folderpath_source   
        self.folderpath_save = config.folderpath_save
        
        self._sam_predictor = SamPredictorWrapper(
            model_type=self._model_type,
            checkpoint_path=self._checkpoint_path
        )

    def process_images(self):
        """Main processing pipeline"""
        paths_image = DirectoryManager.get_image_paths(self.folderpath_source, self.IMAGE_EXTENSIONS)
        results_dir = DirectoryManager.setup_directories(self.folderpath_save)
        
        # Initialize with first image
        first_image = ImageProcessor.load_image(paths_image[0])
        first_image = ImageProcessor.rescale(first_image, 1/self.DOWNSCALE_FACTOR)
        
        points_positive = get_click_coordinates(
            cv2.cvtColor(first_image, cv2.COLOR_RGB2BGR)
        )
        
        points_negative = None
        mask = None
        # Process all images
        for path_image in tqdm(paths_image, desc="Processing images"):
            image_rgb = ImageProcessor.load_image(path_image)
            image_rgb = ImageProcessor.rescale(image_rgb, 1/self.DOWNSCALE_FACTOR)
            
            # Predict mask
            masks = self._sam_predictor.predict_mask(
                image_rgb, 
                points_positive, 
                points_negative,
                mask_input=mask
            )
            
            mask = masks[1]
            # Get the biggest mask object
            mask = mask.astype(np.uint8)*255
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            mask = np.zeros_like(mask)
            if contours:
                biggest_contour = max(contours, key=cv2.contourArea)
                cv2.drawContours(mask, [biggest_contour], -1, 255, cv2.FILLED)
            
            if self.is_display:
                MaskVisualizer.display_image_with_data(image_rgb, mask, points_positive, points_negative)
                
            # get points for next frame
            points_negative = self._get_negative_points(mask, num_points=30, min_distance=18)
            points_positive = self._get_positive_points(mask, num_points=2)

            # Save mask and update center
            output_path = self._get_save_path(path_image)
            save_mask_as_image(mask, output_path)
    
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
        
    
    def _get_positive_points(self, mask, num_points=2):
        """
        Generate positive points using k-means clustering within the mask.
        
        Args:
            mask: Binary mask array
            num_points: Number of points to generate
        
        Returns:
            np.array: Array of [x, y] coordinates for positive points
        """
        # Get valid points from the mask
        y_indices, x_indices = np.where(mask == 255)
        if len(y_indices) == 0:
            return np.array([])

        # Combine coordinates into a single array for clustering
        coordinates = np.column_stack((x_indices, y_indices))

        # Use k-means clustering to find clusters
        kmeans = KMeans(n_clusters=min(num_points, len(coordinates)), random_state=0, n_init='auto')
        kmeans.fit(coordinates)
        cluster_centers = kmeans.cluster_centers_

        # Round cluster centers to nearest integer coordinates
        selected_points = np.round(cluster_centers).astype(int)
        
        return selected_points
    
    def _get_negative_points(self, mask, num_points=5, min_distance=50):
        import cv2
        import numpy as np
        from scipy.spatial.distance import cdist

        # Find mask contours
        mask = mask.astype(np.uint8)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Create grid of candidate points
        h, w = mask.shape
        x = np.linspace(0, w-1, 20).astype(int)
        y = np.linspace(0, h-1, 20).astype(int)
        xx, yy = np.meshgrid(x, y)
        grid_points = np.column_stack((xx.ravel(), yy.ravel()))
        
        # Calculate distances from each grid point to contour points
        contour_points = np.vstack([c.squeeze() for c in contours])
        distances = cdist(grid_points, contour_points)
        min_distances = distances.min(axis=1)
        
        # Filter points that are:
        # 1. Far enough from mask
        distance_mask = min_distances > min_distance
        
        # 2. Outside the mask
        outside_mask = []
        for point in grid_points:
            result = cv2.pointPolygonTest(contours[0], (float(point[0]), float(point[1])), False)
            outside_mask.append(result < 0)  # Negative means outside
        
        # Combine both conditions
        valid_points = grid_points[np.logical_and(distance_mask, outside_mask)]
        
        # Randomly select points
        if len(valid_points) > num_points:
            indices = np.random.choice(len(valid_points), num_points, replace=False)
            selected_points = valid_points[indices]
        else:
            selected_points = valid_points
            
        return selected_points.tolist()
    
    