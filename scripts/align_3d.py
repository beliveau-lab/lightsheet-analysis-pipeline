"""Module for 3D object alignment using principal component analysis."""

import numpy as np
from scipy.spatial.transform import Rotation as R
from sklearn.decomposition import PCA

def get_axes_lengths(eigenvecs, zyx_coords):
    """
    Calculate the lengths along principal axes. Only used for testing.
    Params:
        eigenvecs: Principal component vectors.
        zyx: 3D coordinate array in ZYX format.
        verbose: Whether to print debug information.
    
    Returns:
        List of axis lengths.
    """
    projections = np.dot(zyx_coords, eigenvecs.T)
    lengths = np.ptp(projections, axis=0) # max - min along each axis
    return lengths

def get_principal_axes(zyx_coords):
    """
    Get principal axes from 3D coordinates using PCA.
    Params:
        zyx_coords: 3D coordinate array in ZYX format.
    
    Returns:
        List of principal axes (major, intermediate, minor)
    """
    pca = PCA(n_components=3)
    pca.fit(zyx_coords)
    eigenvecs = pca.components_
    # standardizing to right hand rule
    # principal axis form orthonormal basis -> 
    # if (major x intermediate) . minor < 0, then minor is in the opposite direction
    if np.dot(np.cross(eigenvecs[0], eigenvecs[1]), eigenvecs[2]) < 0: 
        eigenvecs[2] = -eigenvecs[2]
    # return major, intermediate, minor
    return eigenvecs
 
def get_zyx_coords(label_slice):
    """
    Extract ZYX coordinates from 3D binary array.
    Params:
        label_slice: 3D binary array.
    
    Returns:
        Array of ZYX coordinates as float64.
    """
    indices = np.nonzero(label_slice)
    if len(indices[0]) == 0:
        return np.empty((0, 3))
    return np.column_stack(indices).astype(np.float64)

def align_object(label_slice, df_props):
    """
    Align 3D object using coordinate transformation.
    Params:
        label_slice: 3D binary array containing the object.
        verbose: Whether to print debug information.
    
    Returns:
        Aligned 3D binary array and dataframe of object properties
    """
    zyx_coords = get_zyx_coords(label_slice)
    if len(zyx_coords) < 3:
        return label_slice, df_props
    
    eigenvecs = get_principal_axes(zyx_coords)
    lengths = get_axes_lengths(eigenvecs, zyx_coords)
    df_props['major_magnitude'] = lengths[0]
    df_props['intermediate_magnitude'] = lengths[1]
    df_props['minor_magnitude'] = lengths[2]

    target_axes = np.eye(3)
    rotation_obj, _ = R.align_vectors(target_axes, eigenvecs)

    centered_coords = zyx_coords - np.mean(zyx_coords, axis=0)
    rotated_coords = rotation_obj.apply(centered_coords)

    padding = 100
    min_coords = np.min(rotated_coords, axis=0)
    shifted_coords = rotated_coords - min_coords + padding
    int_coords = np.round(shifted_coords).astype(int)
    
    output_shape = tuple(np.max(int_coords, axis=0) + padding)
    aligned_obj = np.zeros(output_shape, dtype=bool)
    z_coords = int_coords[:, 0]
    y_coords = int_coords[:, 1] 
    x_coords = int_coords[:, 2]
    aligned_obj[z_coords, y_coords, x_coords] = True
    return aligned_obj, df_props

# def align_object(label_slice, props, verbose=False):
#     zyx_coords = get_zyx_coords(label_slice)
#     original_volume = np.sum(label_slice > 0)
#     if len(zyx_coords) < 3:
#         return label_slice, props
    
#     eigenvecs = get_principal_axes(zyx_coords)
#     lengths = get_axes_lengths(eigenvecs, zyx_coords)
#     props['axis_major_length'] = lengths[0]
#     props['axis_minor_length'] = lengths[2]

#     target_axes = np.eye(3)
#     rotation_obj, _ = R.align_vectors(target_axes, eigenvecs)

#     if verbose:
#         print(f"Original volume: {original_volume} voxels")
#         print(f"Rotation quaternion (x,y,z,w): {rotation_obj.as_quat()}")

#     centered_coords = zyx_coords - np.mean(zyx_coords, axis=0)
#     rotated_coords = rotation_obj.apply(centered_coords)
    
#     padding = 200  # voxels on every side
#     min_coords = np.min(rotated_coords, axis=0)
#     shifted_coords = rotated_coords - min_coords + padding
#     int_coords = np.round(shifted_coords).astype(int)
    
#     output_shape = tuple(np.max(int_coords, axis=0) + padding)
#     aligned_volume = np.zeros(output_shape, dtype=bool)
#     aligned_volume[int_coords[:, 0], int_coords[:, 1], int_coords[:, 2]] = True
    
#     if verbose:
#         final_volume = np.sum(aligned_volume)
#         print(f"Final volume: {final_volume} voxels")
#         print(f"Volume preservation ratio: {final_volume/original_volume:.3f}")

#     return aligned_volume, props

