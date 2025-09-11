import numpy as np
from scipy.spatial.transform import Rotation as R
from sklearn.decomposition import PCA
from scipy.ndimage import affine_transform 
from scipy.stats import skew

def enforce_righthand_rule(M):
    if np.linalg.det(M) < 0:
        M[:, 2] *= -1.0
    return M

def get_xyz_coords(label_slice) -> np.ndarray:
    """
    Convert volumetrix ZYX data to XYZ coordinates for PCA based alignment

    Params
    -------
    label_slice: (3,3,3) ndarray 
        Volumetric binary mask of object

    Returns
    -------
    coords: (N, 3)
        Coordinate representation of label_slice in XYZ order
    """
    indices = np.nonzero(label_slice)
    if len(indices[0]) == 0:
        return np.empty((0, 3))
    coords = np.column_stack(indices).astype(np.float32)
    return coords[:, [2,1,0]]

def update_properties(V: np.ndarray, X: np.ndarray, df_props: dict) -> dict:
    """
    Compute principal-axis magnitudes and their cosines with the world Z,Y,X axes.

    Params
    -------
    V: (3, 3) ndarray
        Columns are the principal axes [major, intermediate, minor] expressed in XYZ
        coordinates. This must be the sign-disambiguated basis 
    X: (N, 3) ndarray
        Centered point coordinates in XYZ (coords - centroid)
    df_props: dict
        dictionary from process_object that contains various metadata about object

    Returns
    -------
    df_props: dict
        Properties dictionary with updates: length of principal axis, cos(major, Z),
        cos(intermediate, Y), cos(minor, X)
    """
    # Project centered points onto each principal axis to get extent per axis.
    projections = X @ V
    lengths = np.ptp(projections, axis=0)  #[major, intermediate, minor]

    # Cosines with Z,Y,X respectively are just the components of each axis vector
    # Sign is preserved so that cosines reflect orientation
    df_props.update({
        'major_magnitude': float(lengths[0]),
        'intermediate_magnitude': float(lengths[1]),
        'minor_magnitude': float(lengths[2]),
        'cos_major': float(V[2, 0]),         # major vs Z
        'cos_intermediate': float(V[1, 1]),  # intermediate vs Y
        'cos_minor': float(V[0, 2])          # minor vs X
    })
    return df_props


def perform_rotation(label_slice: np.ndarray, R_zyx: np.ndarray) -> np.ndarray:
    """
    After standardizing and removing ambiguity from principal axes alignment,
    perform actual rotation of object. 

    Params
    -------
    label_slice: (3,3,3) ndarray 
        Volumetric binary mask of object - prior to rotation
    R_zyx: np.ndarray
        Rotation matrix meeting requirements for unique rotatoin
    Returns
    -------
    rotated: np.ndarray
        Aligned object in volumetric form. Converted to binary
        data for consistency. 
    """
    shape_in = np.array(label_slice.shape, dtype=np.float64)
    half_in = (shape_in - 1.0) / 2.0

    # Bounding-box of rotated cuboid: new half-sizes = |R| @ half_in
    half_out = np.abs(R_zyx) @ half_in
    shape_out = np.ceil(2.0 * half_out + 1.0).astype(int)

    center_out = (shape_out - 1.0) / 2.0
    M = R_zyx.T                     # inverse
    offset = half_in - M @ center_out
    rotated = affine_transform(
        label_slice.astype(np.uint8),
        M,
        offset=offset,
        output_shape=tuple(shape_out),
        order=0,
        mode='constant',
        cval=0
    )
    return rotated

def orient_axes_skew(X):
    # 1) Principal axes
    pca = PCA(n_components=3).fit(X)
    E = pca.components_.T  # columns v1,v2,v3

    # 2) Skewness-based sign fix (PC1, PC2)
    proj_major = X @ E[:, 0]
    proj_intermediate = X @ E[:, 1]
    skew_major, skew_intermediate = (
        skew(proj_major, bias=False, nan_policy="omit"), 
        skew(proj_intermediate, bias=False, nan_policy="omit")
        )
    if skew_major < 0:
        E[:, 0] *= -1.0
    if skew_intermediate < 0:
        E[:, 1] *= -1.0
    return enforce_righthand_rule(E)

    
def align_object_skew(label_slice: np.ndarray, df_props: dict) -> np.ndarray:
    """
    Align a ZYX binary mask by PCA with skewness-based sign disambiguation.
    Returns a rotated binary mask in ZYX order.
    """
    if label_slice.ndim != 3:
        raise ValueError(f"label_slice must be 3D (Z,Y,X), got {label_slice.shape}")

    # Foreground coords -> XYZ for PCA
    coords_xyz = get_xyz_coords(label_slice)
    if coords_xyz.shape[0] < 3:
        return label_slice.astype(bool), df_props

    centroid_xyz = coords_xyz.mean(axis=0)
    X = coords_xyz - centroid_xyz
    E = orient_axes_skew(X)

    # Create rotation matrix
    R_xyz = np.column_stack([E[:, 0], E[:, 1], E[:, 2]]).T

    # Map back to ZYX form
    P = np.array([[0,0,1],[0,1,0],[1,0,0]])  # XYZ <-> ZYX
    R_zyx = P @ R_xyz @ P.T
    
    # 5) Update properties and perform finalized rotation
    df_props = update_properties(E, X, df_props)
    aligned = perform_rotation(label_slice, R_zyx)
    return aligned, df_props

def get_standardized_axes(coords, reference_axes):
    """
    """    
    pca = PCA(n_components=3)
    pca.fit(coords)
    E = pca.components_.T
    # Disambiguate axes direction by ensuring all principal axes are in the positive octant
    # Sign disambiguation: compare EACH COLUMN to its reference
    for i in range(3):
        if np.dot(E[:, i], reference_axes[i]) < 0:
            E[:, i] *= -1.0

    E = enforce_righthand_rule(E)
    return E
    
def align_object_reference(label_slice, df_props):
    coords = get_xyz_coords(label_slice)
    if coords.shape[0] < 10:
        return label_slice, df_props

    # Center coordinates (XYZ)
    centroid = coords.mean(axis=0)
    X = coords - centroid

    # Reference axes (rows): major→Z, intermediate→Y, minor→X
    reference_axes = np.array([[0, 0, 1],
                               [0, 1, 0],
                               [1, 0, 0]], dtype=float)

    # PCA basis with sign disambiguation against reference
    E = get_standardized_axes(X, reference_axes)

    R_xyz = reference_axes @ E.T

    #XYZ->ZYX for the volume
    P = np.array([[0, 0, 1],
                  [0, 1, 0],
                  [1, 0, 0]], dtype=float)
    R_zyx = P @ R_xyz @ P.T

    aligned = perform_rotation(label_slice, R_zyx)
    df_props = update_properties(E, X, df_props)
    return aligned, df_props