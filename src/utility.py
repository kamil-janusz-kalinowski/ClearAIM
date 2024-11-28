import cv2
import numpy as np
import matplotlib.pyplot as plt

def save_mask_as_image(mask, output_path):
    """
    Saves the logical mask as an image at the given path.

    :param mask: Logical mask (numpy array)
    :param output_path: Path to the output file
    """
    mask = mask.astype(np.uint8) * 255
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
    return coordinates[0] if coordinates else None

def get_negative_points(mask, num_points=5, min_distance=50):
    """
    Generate negative points away from the mask
    Args:
        mask: Binary mask array
        num_points: Number of negative points to generate
        min_distance: Minimum distance from mask boundary
    Returns:
        List of [x,y] coordinates for negative points
    """
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
    
    # Filter points that are far enough from mask
    far_points = grid_points[min_distances > min_distance]
    
    # Randomly select points
    if len(far_points) > num_points:
        indices = np.random.choice(len(far_points), num_points, replace=False)
        selected_points = far_points[indices]
    else:
        selected_points = far_points
        
    return selected_points.tolist()

def display_mask(image_rgb, masks, point, mask_index=1):
    plt.figure(figsize=(10, 10))
    plt.imshow(image_rgb)
    plt.imshow(masks[mask_index], alpha=0.5)
    plt.scatter(point[:, 0], point[:, 1], color='red', s=100, marker='x')
    plt.axis('off')
    plt.show()