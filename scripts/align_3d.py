"""3D object alignment using principal component analysis."""

import numpy as np
from scipy.spatial.transform import Rotation as R
from sklearn.decomposition import PCA
from scipy.ndimage import center_of_mass

def get_principal_axes(coords):
    """Get principal axes from 3D coordinates using PCA."""
    pca = PCA(n_components=3)
    pca.fit(coords)
    eigenvecs = pca.components_
    
    # Ensure right-handed coordinate system
    if np.dot(np.cross(eigenvecs[0], eigenvecs[1]), eigenvecs[2]) < 0: 
        eigenvecs[2] = -eigenvecs[2]
    
    return eigenvecs

def get_zyx_coords(label_slice):
    """
    Extract ZYX coordinates from 3D binary array.
    
    Args:
        label_slice: 3D binary array
    
    Returns:
        Array of ZYX coordinates as float64
    """
    indices = np.nonzero(label_slice)
    if len(indices[0]) == 0:
        return np.empty((0, 3))
    return np.column_stack(indices).astype(np.float64)

def align_object(label_slice, df_props):
    """
    Align 3D object to principal axes.
    
    Args:
        label_slice: 3D binary array containing the object
        df_props: Dictionary of object properties
    
    Returns:
        Aligned: 3D binary array aligned to principal axes
        df_props: Dictionary of object properties updated with
            principal axis lengths
    """
    # Get ZYX coordinates and align to principal axes
    coords = get_zyx_coords(label_slice)
    if len(coords) < 10:  # Require at least 10 voxels for meaningful alignment
        return label_slice, df_props
    
    # Get principal axes and calculate lengths
    eigenvecs = get_principal_axes(coords)
    projections = coords @ eigenvecs.T
    lengths = np.ptp(projections, axis=0)
    
    # Update properties
    df_props.update({
        'major_magnitude': lengths[0],
        'intermediate_magnitude': lengths[1], 
        'minor_magnitude': lengths[2]
    })
    
    # align eigenvecs to standard axes using identity matrix 
    rotation = R.align_vectors(eigenvecs, np.eye(3))[0]

    # Transform coordinates
    centroid = np.round(center_of_mass(label_slice)).astype(int)
    # centroid = coords.mean(axis=2, keepdims=True) # centoid is middle of x axis
    centered = coords - centroid
    rotated = rotation.apply(centered)
    
    # Shift to positive coordinates with padding
    padding = 20
    shifted = rotated - np.min(rotated, axis=0) + padding
    int_coords = np.round(shifted).astype(int)
    
    # Create aligned object (maintains ZYX coordinate system)
    shape = tuple(np.max(int_coords, axis=0) + padding + 1)
    aligned = np.zeros(shape, dtype=bool)
    z_coords = int_coords[:, 0]
    y_coords = int_coords[:, 1] 
    x_coords = int_coords[:, 2]
    aligned[z_coords, y_coords, x_coords] = True
    
    return aligned, df_props

# project cell onto spline
# local coordinate system -> tan is AP axis, L/R is binormal, DV is normal
# 9 new columns -> cosine 