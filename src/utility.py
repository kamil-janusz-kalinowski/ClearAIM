'''Module containing utility functions for the project.'''
# pylint: disable=maybe-no-member

import cv2
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
    """
    Displays an image and allows the user to click on it to select coordinates.
    Args:
        image (numpy.ndarray): The image on which the user will click to select points.
    Returns:
        numpy.ndarray: An array of shape (n, 2) containing the coordinates of the points 
                       selected by the user, where n is the number of points.
    """

    coordinates = []

    def mouse_callback(event, x, y, flags, param): # pylint: disable=unused-argument
        if event == cv2.EVENT_LBUTTONDOWN:
            coordinates.append((x, y))
            cv2.destroyAllWindows()

    cv2.imshow('Click to select point', image)
    cv2.setMouseCallback('Click to select point', mouse_callback)
    cv2.waitKey(0)

    coordinates = np.array(coordinates).reshape(-1, 2)
    return coordinates

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
