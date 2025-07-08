import numpy as np
from skimage.measure import marching_cubes
import matplotlib.pyplot as plt
import align_3d as align
import skimage.morphology as skmorpho
import skimage.filters as skfilters
import vtk
from vtk.util import numpy_support as vtknp
import align_3d as align
from aicsshparam.shtools import get_mesh_from_image


def visualize(mask, max_points=10000):
    coords = align.get_zyx_coords(mask)
    if len(coords) > max_points:
        idx = np.random.choice(len(coords), max_points, replace=False)
        coords = coords[idx]
    fig = plt.figure(figsize=(15, 6))
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(coords[:, 2], coords[:, 1], coords[:, 0], alpha=0.7, s=1)
    ax.set_xlabel('Z')
    ax.set_ylabel('Y')
    ax.set_zlabel('X')
    ax.set_title('Rotated Object')
    plt.show()


def visualize_comparison(original, aligned, max_points=10000):
    """
    Create a 3D visualization comparing original and aligned objects.
    
    Args:
        original: Original boolean 3D array
        aligned: Aligned boolean 3D array  
        max_points: Maximum points to plot (for performance)
    """
    # Get coordinates for both objects
    orig_coords = align.get_zyx_coords(original)
    aligned_coords = align.get_zyx_coords(aligned)
    # Subsample if too many points
    if len(orig_coords) > max_points:
        idx = np.random.choice(len(orig_coords), max_points, replace=False)
        orig_coords = orig_coords[idx]
    if len(aligned_coords) > max_points:
        idx = np.random.choice(len(aligned_coords), max_points, replace=False)
        aligned_coords = aligned_coords[idx]
    # Create side-by-side 3D plots
    fig = plt.figure(figsize=(15, 6))
    # Original object
    ax1 = fig.add_subplot(121, projection='3d')
    ax1.scatter(orig_coords[:, 2], 
                orig_coords[:, 1], 
                orig_coords[:, 0], 
                c='red', 
                alpha=0.6, 
                s=1)
    ax1.set_title('Original Object')
    ax1.set_xlabel('Z')
    ax1.set_ylabel('Y')
    ax1.set_zlabel('X')
    # Aligned object
    ax2 = fig.add_subplot(122, projection='3d')
    ax2.scatter(aligned_coords[:, 2], 
                aligned_coords[:, 1], 
                aligned_coords[:, 0], 
                c='blue', 
                alpha=0.6, s=1)
    ax2.set_title('Aligned Object')
    ax2.set_xlabel('Z')
    ax2.set_ylabel('Y')
    ax2.set_zlabel('X')
    
    plt.tight_layout()
    plt.show()


def plot_mesh(mask, max_points=10000):
    """Simple three-view visualization of 3D object with principal axes"""
    # Get coordinates and subsample if needed
    coords = align.get_zyx_coords(mask)
    if len(coords) > max_points:
        idx = np.random.choice(len(coords), max_points, replace=False)
        coords = coords[idx]
    # Get principal axes
    major, intermediate, minor, lengths = align.get_principal_axes(coords, return_lengths=True)
    centroid = coords.mean(axis=0)
    # Setup three views
    fig = plt.figure(figsize=(18, 6))
    views = [
        {'elev': 0, 'azim': 90, 'title': 'Side View'},
        {'elev': 45, 'azim': 45, 'title': 'Isometric View'},
        {'elev': 90, 'azim': 0, 'title': 'Top View'}
    ]
    for idx, view in enumerate(views):
        ax = fig.add_subplot(1, 3, idx + 1, projection='3d')
        ax.scatter(coords[:, 2], 
                   coords[:, 1], 
                   coords[:, 0], 
                   alpha=0.7, 
                   s=1)
        # plot principal axes
        for vec, color, label, length in zip([major, intermediate, minor], 
                                           ['r', 'g', 'b'], 
                                           ['Major', 'Intermediate', 'Minor'],
                                           lengths):
            pt1 = centroid - length/2 * vec
            pt2 = centroid + length/2 * vec
            # Convert ZYX to XYZ for plotting
            ax.plot([pt1[2], pt2[2]], [pt1[1], pt2[1]], [pt1[0], pt2[0]], 
                   color=color, linewidth=3, label=label, alpha=0.5)
        
        ax.view_init(elev=view['elev'], azim=view['azim'])
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_title(view['title'])
        if idx == 0:
            ax.legend()
    
    plt.tight_layout()
    plt.show()

def plot_aligned_object(aligned_volume):
    """
    Plots a 3D mesh of an object after it has been aligned, with its
    principal axes aligned with the coordinate system axes.

    Args:
        aligned_volume (np.ndarray): The 3D numpy array of the aligned object.
    """
    print("Plotting aligned object...")
    xyz_coords = align.get_zyx_coords(aligned_volume)
    major, minor, intermediate = align.get_principal_axes(xyz_coords)
    # For an aligned object, principal axes correspond to the standard basis vectors.
    print(f"\tPlotting with standard axes:")
    print(f"\tMajor Axis: {major}")
    print(f"\tIntermediate Axis: {intermediate}")
    print(f"\tMinor Axis: {minor}")

    plot_mesh(aligned_volume)

