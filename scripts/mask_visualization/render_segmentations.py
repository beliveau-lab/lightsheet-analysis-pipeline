import paraview.simple as pv
import os, sys
import csv
import argparse
from datetime import date
import time
import math
import hashlib
import gc
import colorsys

from paraview import servermanager

# ffmpeg -framerate 5 -i frame_%04d.png  output2.gif

def hash_to_color(label_value):
    """Generate a deterministic color for a label using improved hash-based mapping, cryptographic hash functions,
    and large prime numbers to distribute consecutive values widely.
    
    Parameters
    ----------
    label_value : int
        The label value to generate a color for

    Returns
    -------
    r : float
        The red component of the color
    g : float
        The green component of the color
    b : float
        The blue component of the color
    """
    # 1) multiple transformations to break sequential coloring of nearby labels via large primes
    transformed1 = (label_value * 1299709) % 982451653  
    transformed2 = (label_value * 1299721 + 982451629) % 982451653
    transformed3 = (label_value * 1299743 + 982451631) % 982451653
    
    # 2) create distinct hash inputs for the following hash algorithms to avoid clustering
    hash_input1 = f"{transformed1}_{label_value}_salt1"
    hash_input2 = f"{transformed2}_{label_value}_salt2" 
    hash_input3 = f"{transformed3}_{label_value}_salt3"
    
    # 3) generate hashes from 3 different algoirthms for 'randomization'
    hash1 = hashlib.md5(hash_input1.encode()).hexdigest()
    hash2 = hashlib.sha256(hash_input2.encode()).hexdigest()
    hash3 = hashlib.blake2b(hash_input3.encode(), digest_size=16).hexdigest()
    
    # 4) combine hash values via XOR for bit mixing
    hue_combined = int(hash1[:8], 16) ^ int(hash2[:8], 16) ^ int(hash3[:8], 16)

    # 5) force hue to jump by at least 120 degrees (1/3 of color wheel) for MAXIMUM contrast
    hue_base = (hue_combined % 3) * 120  # Snap to 0, 120, or 240 degrees (Red, Green, Blue)
    hue_offset = (hue_combined // 3) % 120  # Add 0-119 degree offset within third
    hue = (hue_base + hue_offset) / 360.0
    
    # 6) force high saturation and avoid light colors
    sat_combined = int(hash1[8:16], 16) ^ int(hash2[8:16], 16)
    saturation = 0.7 + (sat_combined % 30) / 100.0  # Range 0.7-1.0 (high saturation)
    
    val_combined = int(hash1[16:24], 16) ^ int(hash2[16:24], 16)
    value = 0.4 + (val_combined % 40) / 100.0  # Range 0.4-0.8 (avoid very light colors)  
    
    # 7) finally, convert to RGB
    r, g, b = colorsys.hsv_to_rgb(hue, saturation, value)
    return r, g, b


def generate_colormap(min_label, max_label, total_labels):
    """For very large label ranges, create a much denser color map to minimize interpolation.
    This is what fixed the 'blocky' appeareance of nearby labels being colored similarly. 
    
    Parameters
    ----------
    min_label : int
        The minimum label value in the range
    max_label : int
        The maximum label value in the range
    color_dict : dict
        A dictionary of label values and their corresponding RGB colors
    total_labels : int
        The total number of labels in the range

    Returns
    -------
    rgb_points : list
        A list of RGB points for the color transfer function
    """
    rgb_points = []
    # denser sampling, 24 labels per color 
    num_samples = min(25000, total_labels)  
    step = max(1, total_labels // num_samples)
    
    # Sample every 'step' labels to cover the space densely
    # for embryo 317, this is sampling every 23rd label 
    for i in range(min_label, max_label + 1, step):
        r, g, b = hash_to_color(i)
        rgb_points.extend([i, r, g, b])
    
    # Always ensure there is a color for the min and max labels
    if min_label not in [i for i in range(min_label, max_label + 1, step)]:
        r, g, b = hash_to_color(min_label)
        rgb_points = [min_label, r, g, b] + rgb_points
        
    if max_label not in [i for i in range(min_label, max_label + 1, step)]:
        r, g, b = hash_to_color(max_label)
        rgb_points.extend([max_label, r, g, b])
    return rgb_points



def build_colormap(min_label, max_label):
    """Create optimized color mapping for ParaView.
    
    Parameters
    ----------
    min_label : int
        The minimum label value in the range
    max_label : int
        The maximum label value in the range

    Returns
    -------
    rgb_points : list
        A list of RGB points for the color transfer function
    """
    total_labels = max_label - min_label + 1
    rgb_points = generate_colormap(min_label, max_label, total_labels)
    return rgb_points

def setup_mpi():
    """
    Initialize MPI and return rank info to track threads

    Returns
    -------
    rank : int
        The rank of the current process
    size : int
        The total number of processes
    """
    controller = servermanager.vtkProcessModule.GetProcessModule().GetGlobalController()
    rank = controller.GetLocalProcessId()
    size = controller.GetNumberOfProcesses()
    
    if rank == 0:
        print(f"Script started successfully - Rank {rank} of {size} processes")
        sys.stdout.flush()

    return rank, size

def load_data(collection_file, rank):
    """Load data and convert cell data to point data."""
    if rank == 0:
        print("--- Loading and processing data ---")
        sys.stdout.flush()
    
    src = pv.PVDReader(FileName=collection_file)
    src.UpdatePipeline()
    
    data_info = src.GetDataInformation()
    if rank == 0:
        print(f"Data bounds: {data_info.GetBounds()}")
        print(f"Number of cells: {data_info.GetNumberOfCells()}")
        print(f"Number of points: {data_info.GetNumberOfPoints()}")
        sys.stdout.flush()
    
    # Convert cell data -> point data to address coloring by cell data hardware limitations
    c2p = pv.CellDatatoPointData(Input=src)
    c2p.UpdatePipeline()
    
    # Decimation for large data
    dec = pv.Decimate(Input=c2p)
    dec.TargetReduction = 0.5
    dec.PreserveTopology = 1
    dec.UpdatePipeline()

    if rank == 0:
        orig_points = c2p.GetDataInformation().GetNumberOfPoints()
        new_points = dec.GetDataInformation().GetNumberOfPoints()
        print(f"Decimation: {orig_points} → {new_points} points ({(1-new_points/orig_points)*100:.1f}% reduction)")
    
    return dec, dec.GetDataInformation()

def find_segmentation_array(c2p, rank):
    """Find and validate that the SegmentationLabel array exists.
    
    Parameters
    ----------
    c2p : paraview.simple.Show
        The paraview display object
    rank : int
        The rank of the current process

    Returns
    -------
    segmentation_array_info : paraview.simple.ArrayInformation
        The array information for the SegmentationLabel array (contains labels)
    """
    data_info = c2p.GetDataInformation()
    point_data_info = data_info.GetPointDataInformation()

    segmentation_array_info = None
    array_names = []
    
    for i in range(point_data_info.GetNumberOfArrays()):
        array_info = point_data_info.GetArrayInformation(i)
        array_name = array_info.GetName()
        array_names.append(array_name)
        if array_name == 'SegmentationLabel':
            segmentation_array_info = array_info

    if segmentation_array_info is None:
        raise ValueError("'SegmentationLabel' array not found in point data after conversion.")
    elif rank == 0:
        print(f"Available point arrays: {array_names}")
        sys.stdout.flush()
    
    return segmentation_array_info

def setup_display(c2p, rank):
    """
    Configure display properties for rendering.
    
    Parameters
    ----------
    c2p : paraview.simple.Show
        The paraview display object
    rank : int
        The rank of the current process

    Returns
    -------
    display : paraview.simple.Show
        The paraview display object
    """
    display = pv.Show(c2p)
    display.Representation = 'Surface'
    
    # Ensure all surfaces are visible using properties available in your ParaView version
    try:
        # Set backface properties to show both sides of surfaces
        display.BackfaceRepresentation = 'Surface'  # Show backfaces as surfaces
        display.BackfaceOpacity = 1.0               # Full opacity for backfaces
        
        # Set main surface properties
        display.Opacity = 1.0                       # Full opacity
        display.Visibility = 1                      # Ensure visibility
        
        # Disable shader replacements that might hide surfaces
        display.UseShaderReplacements = 0
        
        # Ensure proper lighting
        display.DisableLighting = 0                 # Keep lighting enabled
        
        if rank == 0:
            print("Display properties successfully configured for full surface visibility")
            sys.stdout.flush()
            
    except Exception as e:
        if rank == 0:
            print(f"Error setting display properties: {e}")
            sys.stdout.flush()
    return display

def setup_view(rank):
    """Configure the view settings for rendering.
    
    Returns
    -------
    view : paraview.simple.View
        The paraview view object
    """
    view = pv.GetActiveViewOrCreate('RenderView')
    view.LODThreshold = 0.0
    view.RemoteRenderThreshold = 0
    view.ViewSize = [3000, 3000]
    view.Background = [0.0, 0.0, 0.0] # black background
    
    try:
        view.LODResolution = 1.0
    except Exception as e:
        if rank == 0:
            print(f"Error setting view properties: {e}")
            sys.stdout.flush()
    
    return view

def apply_color_mapping(display, min_label, max_label, rank)-> None:
    """Apply optimized color mapping to the display.
    
    Parameters
    ----------
    display : paraview.simple.Show
        The paraview display object
    color_map_file : str
        The path to the color map CSV file
    min_label : int
        The minimum label value in the range
    max_label : int
        The maximum label value in the range
    rank : int
        The rank of the current process
    """
    if rank == 0:
        print("Applying optimized color mapping...")
        sys.stdout.flush()
        
    pv.ColorBy(display, ('POINTS', 'SegmentationLabel'))
    lut = pv.GetColorTransferFunction('SegmentationLabel')

    rgb_points = build_colormap(min_label, max_label)
    
    if rank == 0:
        print(f"Using {len(rgb_points)//4} color points for transfer function")
        sys.stdout.flush()
    
    lut.RescaleTransferFunction(min_label, max_label)
    lut.RGBPoints = rgb_points
    
    # Try to set discrete mode properties
    try:
        lut.UseLogScale = 0
        lut.Discretize = 1
        lut.NumberOfTableValues = min(65536, len(rgb_points) // 4)
    except Exception as e:
        if rank == 0:
            print(f"Error setting color mapping properties: {e}")
            sys.stdout.flush()
    return

def setup_camera(view, data_info, rank)-> None:
    """Position camera to show the full dataset.
    
    Parameters
    ----------
    view : paraview.simple.View
        The paraview view object
    data_info : paraview.simple.DataInformation
        The data information for the dataset
    rank : int
        The rank of the current process
    """
    if rank == 0:
        print("Setting up camera to show full dataset...")
        sys.stdout.flush()
    
    view.ResetCamera()
    
    data_bounds = data_info.GetBounds()
    center_x = (data_bounds[0] + data_bounds[1]) / 2
    center_y = (data_bounds[2] + data_bounds[3]) / 2
    center_z = (data_bounds[4] + data_bounds[5]) / 2
    
    max_extent = max(data_bounds[1] - data_bounds[0], 
                     data_bounds[3] - data_bounds[2], 
                     data_bounds[5] - data_bounds[4])
    camera_distance = max_extent * 3
    
    view.Set(
        CameraPosition=[center_x, center_y, center_z + camera_distance],
        CameraFocalPoint=[center_x, center_y, center_z],
        CameraParallelScale=max_extent * 0.6,
    )
    
    if rank == 0:
        print(f"Dataset bounds: {data_bounds}")
        print(f"Camera distance: {camera_distance}")
        sys.stdout.flush()
    return

def save_screenshot(output_file, rank)-> None:
    """Render and save the main screenshot.
    
    Parameters
    ----------
    output_file : str
        The path to the output file
    rank : int
        The rank of the current process
    """
    pv.Render()
    if rank == 0:
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        pv.SaveScreenshot(output_file)
        print(f'Screenshot saved to: {output_file}')
        sys.stdout.flush()
    return 

def generate_animation(view, animation_dir, num_frames, rank)-> None:
    """
    Generate rotating animation frames. FFMPEG can be used to combine the frames into a video 
    or gif
    
    Parameters
    ----------
    view : paraview.simple.View
        The paraview view object
    animation_dir : str
        The path to the animation directory
    num_frames : int
        The number of frames to render
    rank : int
        The rank of the current process
    """
    if rank == 0:
        print("--- Starting Animation ---")
        os.makedirs(animation_dir, exist_ok=True)
        sys.stdout.flush()
        
    focal_point = view.CameraFocalPoint
    initial_position = view.CameraPosition
    radius_vector = [initial_position[i] - focal_point[i] for i in range(3)]
    radius = math.sqrt(sum([c*c for c in radius_vector]))

    angle_step = 2 * math.pi / num_frames
    focal_x, focal_y, focal_z = focal_point
    initial_y = initial_position[1]
    
    for i in range(num_frames):
        angle = i * angle_step
        new_x = focal_x + radius * math.sin(angle)
        new_z = focal_z + radius * math.cos(angle)
        
        view.CameraPosition = [new_x, initial_y, new_z]
        # garbage collection every 5 frames to prevent memory buildup
        if i % 5 == 0:
            gc.collect()
        pv.Render()
        
        # only pid 0 should save the frames
        if rank == 0:
            frame_filename = os.path.join(animation_dir, f'frame_{i:04d}.png')
            pv.SaveScreenshot(frame_filename)
            print(f"Saved frame {i+1}/{num_frames}", end='\r')
            sys.stdout.flush()

    if rank == 0:
        print(f"\nSaved {num_frames} frames to {animation_dir}")
        sys.stdout.flush()
    return

def main():
    parser = argparse.ArgumentParser(description="Visualize compressed surface data with optimized color mapping.")
    parser.add_argument('--embryonic_number', type=int, required=True, help="Embryonic number (e.g., 317, 290).")
    parser.add_argument('--output_dir', type=str, required=True, help="Directory to save the output PNG file.")
    parser.add_argument('--surface_dir', type=str, required=True, help="Directory to load the surface data from.")
    parser.add_argument('--num_frames', type=int, default=25, help="Number of frames to render.")
    parser.add_argument('--output_tag', type=str, default='_1', help="Tag to add to the output file name.")
    args = parser.parse_args()

    rank, _ = setup_mpi()
    embryonic_number = args.embryonic_number
    surface_dir = args.surface_dir
    output_tag = args.output_tag
    tag_str = f'vtp_{embryonic_number}_{date.today()}{output_tag}'
    
    collection_file = os.path.join(surface_dir, 'combined_surface.pvd')
    output_file = os.path.join(args.output_dir, f'{tag_str}.png')
    animation_dir = os.path.join(args.output_dir, f'animation_frames_{tag_str}')

    # --- VALIDATE INPUT FILES ---
    if not os.path.exists(collection_file):
        if rank == 0:
            print(f'Collection file not found: {collection_file}')
        sys.exit(1)

    # --- BEGIN VISUALIZATION PROCESS ---

    if rank == 0:
        print(f"---------- Beginning Optimized Visualization of E9.5_{args.embryonic_number} ----------")
        sys.stdout.flush()

    # --- DATA LOADING ---
    c2p, data_info = load_data(collection_file, rank)
        
    segmentation_array_info = find_segmentation_array(c2p, rank)
    
    data_range = segmentation_array_info.GetComponentRange(0)
    min_label = int(data_range[0])
    max_label = int(data_range[1])
    
    if rank == 0:
        print(f"SegmentationLabel range: {min_label} to {max_label}")
        print(f"Total unique labels: {max_label - min_label + 1}")
        sys.stdout.flush()
    
    # --- VISUALIZATION SETUP ---
    display = setup_display(c2p, rank)
    view = setup_view(rank)
    apply_color_mapping(display, min_label, max_label, rank)
    setup_camera(view, data_info, rank)
    
    # --- GENERATE MAIN SCREENSHOT ---
    save_screenshot(output_file, rank)

    # --- GENERATE ANIMATION ---
    gc.collect()
    generate_animation(view, animation_dir, args.num_frames, rank)

    return output_file

if __name__ == "__main__":
    try:
        start_time = time.time()
        result = main()
        end_time = time.time()
        time_taken = end_time - start_time
        num_minutes = time_taken / 60
        
        controller = servermanager.vtkProcessModule.GetProcessModule().GetGlobalController()
        rank = controller.GetLocalProcessId()
        
        if rank == 0:
            print(f"Time taken: {num_minutes} minutes")
            if result:
                print("---------- Optimized Visualization Ended Successfully ----------")
            else:
                print("---------- Optimized Visualization Failed ----------")
            sys.stdout.flush()
            
        if not result:
            sys.exit(1)
            
    except Exception as e:
        controller = servermanager.vtkProcessModule.GetProcessModule().GetGlobalController()
        rank = controller.GetLocalProcessId()
        
        if rank == 0:
            print(f"Error during optimized VTM visualization: {e}")
            import traceback
            traceback.print_exc()
            sys.stdout.flush()
        sys.exit(1)