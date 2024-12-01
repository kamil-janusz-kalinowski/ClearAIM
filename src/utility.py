import cv2
import numpy as np
import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import KMeans



def save_mask_as_image(mask, output_path):
    """
    Saves the logical mask as an image at the given path.

    :param mask: Logical mask (numpy array)
    :param output_path: Path to the output file
    """
    mask = mask.astype(np.uint8)
    cv2.imwrite(output_path, mask)
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
    
    coordinates = np.array(coordinates).reshape(-1, 2)
    
    return coordinates

def get_negative_points(mask, num_points=5, min_distance=50):
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

def display_mask(image_rgb, masks, point, mask_index=1):
    plt.figure(figsize=(10, 10))
    plt.imshow(image_rgb)
    plt.imshow(masks[mask_index], alpha=0.5)
    plt.scatter(point[:, 0], point[:, 1], color='red', s=100, marker='x')
    plt.axis('off')
    plt.show()

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