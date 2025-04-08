import numpy as np

import supervisely as sly
from supervisely.geometry.pointcloud import Pointcloud


# function for projecting 3D point cloud points on 2D photo context image
def project_3d_to_uvz_array(P_w_array, K, R, T):
    """
    Project multiple 3D points in world coordinates to 2D image coordinates (u, v)
    and include the depth (z) for each point.

    Parameters:
    - P_w_array: A 2D numpy array of shape (N, 3) where N is the number of 3D points.
    - K: The intrinsic matrix (3x3).
    - R: The rotation matrix (3x3).
    - T: The translation vector (3x1).

    Returns:
    - A 2D numpy array of shape (N, 3) where each row contains [u, v, z] for a point.
    """

    # Convert the world points to camera coordinates
    P_c_array = np.dot(P_w_array, R.T) + T  # (N, 3)

    # Extract camera coordinates (X_c, Y_c, Z_c)
    X_c, Y_c, Z_c = P_c_array[:, 0], P_c_array[:, 1], P_c_array[:, 2]

    # Calculate the 2D projections (u, v) using the intrinsic matrix
    u = (K[0, 0] * X_c / Z_c) + K[0, 2]
    v = (K[1, 1] * Y_c / Z_c) + K[1, 2]

    # Stack the results: [u, v, z] for each point
    uvz_array = np.vstack([u, v, Z_c]).T  # (N, 3)

    return uvz_array


def extract_largest_cluster(pcd, indices, eps=1.5, min_points=100):
    """
    Extract the largest cluster from a point cloud subset.

    Parameters:
    - pcd: Open3D point cloud
    - indices: Indices of points to consider
    - eps: DBSCAN epsilon parameter
    - min_points: DBSCAN minimum points parameter

    Returns:
    - List of indices from the largest cluster
    """
    masked_pcd = pcd.select_by_index(indices)
    cluster_labels = np.array(masked_pcd.cluster_dbscan(eps=eps, min_points=min_points))

    if len(cluster_labels) == 0:
        return []

    clusters, counts = np.unique(cluster_labels, return_counts=True)

    biggest_cluster = clusters[np.argsort(counts)][-1:]
    biggest_cluster_indices = []
    for idx, label in enumerate(cluster_labels):
        if label in biggest_cluster:
            biggest_cluster_indices.append(int(indices[idx]))
    return biggest_cluster_indices


# def get_points_inside_mask(u, v, z, mask, img_width, img_height):
#     """
#     Get indices of points that project inside a given mask.

#     Parameters:
#     - u, v, z: Projected point coordinates
#     - mask: Binary mask
#     - img_width, img_height: Image dimensions

#     Returns:
#     - List of indices of points inside the mask
#     """
#     inside_masks = []

#     for idx in range(len(u)):
#         x, y = int(u[idx]), int(v[idx])
#         depth = z[idx]

#         # Check if point is within image bounds and has positive depth
#         if x <= 0 or x >= img_width or y <= 0 or y >= img_height or depth < 0:
#             continue

#         # Check if point is inside the mask
#         if mask[y, x] == 1:
#             inside_masks.append(idx)

#     return inside_masks


def get_points_inside_mask(u, v, z, mask, img_width, img_height):
    """
    Get indices of points that project inside a given mask.

    Parameters:
    - u, v, z: Projected point coordinates
    - mask: Binary mask
    - img_width, img_height: Image dimensions

    Returns:
    - List of indices of points inside the mask
    """
    # Convert to integers
    x = np.floor(u).astype(int)
    y = np.floor(v).astype(int)

    # Create mask for valid points
    valid_points = (x >= 0) & (x < img_width) & (y >= 0) & (y < img_height) & (z > 0)

    # Find indices of valid points
    valid_indices = np.where(valid_points)[0]

    # Filter valid points
    valid_x = x[valid_indices]
    valid_y = y[valid_indices]

    # Find which valid points are inside the mask
    inside_mask_indices = []
    for i, (x_i, y_i) in enumerate(zip(valid_x, valid_y)):
        if mask[y_i, x_i] == 1:
            inside_mask_indices.append(valid_indices[i])

    return inside_mask_indices


def get_obj_class(meta: sly.ProjectMeta, src_obj_class: sly.ObjClass):
    """
    Check if the object class exists in the project metadata.
    If not, add it.

    Returns:
    - obj_class: The object class from the metadata.
    - meta: Updated project metadata.
    - need_update: Boolean indicating if metadata needs to be updated.
    """
    obj_class_name = src_obj_class.name
    obj_class = meta.get_obj_class(obj_class_name)
    need_update = False

    if obj_class is not None and obj_class.geometry_type != Pointcloud:
        # Rename the existing class to avoid conflicts
        obj_class_name = f"{obj_class_name}_3d"
        obj_class = None

    if obj_class is None:
        # Create a new object class
        obj_class = sly.ObjClass(obj_class_name, Pointcloud, color=src_obj_class.color)
        meta = meta.add_obj_class(obj_class)
        need_update = True

    return obj_class, meta, need_update
