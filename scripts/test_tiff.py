import numpy as np
import tifffile
import matplotlib.pyplot as plt
import os
from align_3d import (
    align_object,
    get_zyx_coords
)
def visualize_comparison(original, aligned, main_dir, file, max_points=5000):
    """
    Create a 3D visualization comparing original and aligned objects.
    
    Args:
        original: Original boolean 3D array
        aligned: Aligned boolean 3D array  
        max_points: Maximum points to plot (for performance)
    """
    print(f"\nCreating visualization for {file}...")
    try:
        # Get coordinates for both objects
        orig_coords = get_zyx_coords(original)
        aligned_coords = get_zyx_coords(aligned)
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
        plt.savefig(os.path.join(f'{main_dir}/visuals/', f"{file}_visualization.png"))
        print("   ✓ Visualization created")
    except Exception as e:
        print(f"   ✗ Error creating visualization: {e}")

def load_and_test_pipeline(main_dir):
    try:
        original_files = os.listdir(os.path.join(main_dir, "original"))
        objects = []
        label_slices = []
        for file in original_files:
            information={}
            information['original_file'] = file
            object = tifffile.imread(os.path.join(main_dir, "original", file))
            objects.append(object)
            # Convert to boolean if needed
            if object.dtype != bool:
                label_slice = object > 0
                label_slices.append(label_slice)
            else:
                label_slice = object
                label_slices.append(label_slice)
            aligned_obj = align_object(label_slice)
            # Check volume preservation
            aligned_volume = np.sum(aligned_obj > 0)
            information['original_shape'] = object.shape
            information['original_volume'] = np.sum(object > 0)
            information['aligned_shape'] = aligned_obj.shape
            information['aligned_volume'] = aligned_volume
            volume_diff = abs(information['original_volume'] - aligned_volume)
            volume_percent = (volume_diff / information['original_volume']) * 100
            information['percent_difference'] = volume_percent
            if volume_percent > 15:
                print(f"   ⚠ Warning: Volume difference is {volume_percent:.2f}% (>15%)")
            for key, value in information.items():
                print(f"   ✓ {key}: {value}")
            visualize_comparison(label_slice, aligned_obj, main_dir, file)
    except Exception as e:
        print(f"   ✗ Error in object alignment: {e}")
        return None, None

if __name__ == "__main__":
    main_dir = '/net/beliveau/vol1/home/msforman/msf_project/lightsheet-analysis-pipeline/scripts/'
    load_and_test_pipeline(main_dir)