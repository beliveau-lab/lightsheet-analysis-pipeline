import paraview.simple as pv
import os, sys
import csv
import argparse
from datetime import date
import time
import math
import hashlib
import gc
import colorcet as cc
import random
import colorsys
from paraview import servermanager
import logging

# ffmpeg -framerate 5 -i frame_%04d.png  output2.gif
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def all_hex_to_rgb(hex_colors):
    def hex_to_rgb(hex):
        hex = hex.lstrip('#')
        return tuple(int(hex[i:i+2], 16) for i in (0, 2, 4))
    return [hex_to_rgb(hex) for hex in hex_colors]

def remove_light_colors(rgb_colors):
    for color in rgb_colors:
        _, l, _ = colorsys.rgb_to_hls(color[0], color[1], color[2])
        if l > 0.6:
            rgb_colors.remove(color)
    return rgb_colors

def hash_to_color(label_value, colors):
    if label_value == 0:
        return 0.0, 0.0, 0.0
    # last_digit = label_value % 10
    # index = last_digit % len(colors)
    # hash_object = hashlib.sha256(str(label_value).encode('utf-8'))
    # seed = int(hash_object.hexdigest(), 16)
    # rng = random.Random(seed)
    # index = rng.randint(0, len(colors) - 1)
    index = random.randint(0, len(colors) - 1)
    r, g, b = colors[index]
    return r, g, b

def generate_colormap(min_label, max_label, total_labels, colors):
    rgb_points = []
    num_samples = 100000
    step = total_labels // num_samples
    for i in range(min_label, max_label+1, step):
        r, g, b = hash_to_color(i, colors)
        rgb_points.extend([i, r, g, b])
    return rgb_points

def build_colormap(min_label, max_label, colors):
    total_labels = max_label - min_label + 1
    rgb_points = generate_colormap(min_label, max_label, total_labels, colors)
    return rgb_points

def setup_mpi():
    controller = servermanager.vtkProcessModule.GetProcessModule().GetGlobalController()
    rank = controller.GetLocalProcessId()
    size = controller.GetNumberOfProcesses()
    
    if rank == 0:
        logger.info(f"Script started successfully - Rank {rank} of {size} processes")
        sys.stdout.flush()

    return rank, size

def load_data(collection_file, rank, decimated_path):
    if rank == 0:
        logger.info('='*40)
        logger.info("Loading and processing data")
        logger.info('='*40)
        sys.stdout.flush()
    
    src = pv.PVDReader(FileName=collection_file)
    src.UpdatePipeline()
    
    c2p = pv.CellDatatoPointData(Input=src)
    c2p.UpdatePipeline()
    
    dec = pv.Decimate(Input=c2p)
    dec.TargetReduction = 0.8
    dec.PreserveTopology = 1
    dec.UpdatePipeline()

    if rank == 0:
        orig_points = c2p.GetDataInformation().GetNumberOfPoints()
        new_points = dec.GetDataInformation().GetNumberOfPoints()
        logger.info(f"Decimation: {orig_points} -> {new_points} points ({(1-new_points/orig_points)*100:.1f}% reduction)")
    return dec, dec.GetDataInformation()

def find_labels(c2p, rank):
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
        logger.info(f"Available point arrays: {array_names}")
        sys.stdout.flush()
    
    return segmentation_array_info

def setup_display(c2p, rank):
    display = pv.Show(c2p)
    display.Representation = 'Surface'
    
    # Ensure all surfaces are visible using properties available in your ParaView version
    # try:
    #     display.BackfaceRepresentation = 'Surface'  # Show backfaces as surfaces
    #     display.BackfaceOpacity = 1.0               # Full opacity for backfaces
        
    #     display.Opacity = 1.0                       # Full opacity
    #     display.Visibility = 1                      # Ensure visibility
        
    #     display.UseShaderReplacements = 0
        
    #     display.DisableLighting = 0                 # Keep lighting enabled
        
    #     if rank == 0:
    #         logger.info("Display properties successfully configured for full surface visibility")
    #         sys.stdout.flush()
            
    # except Exception as e:
    #     if rank == 0:
    #         logger.info(f"Error setting display properties: {e}")
    #         sys.stdout.flush()
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
            logger.info(f"Error setting view properties: {e}")
            sys.stdout.flush()
    
    return view

def apply_color_mapping(display, min_label, max_label, rank, colors)-> None:
    if rank == 0:
        logger.info("Applying colormap...")
        sys.stdout.flush()
        
    pv.ColorBy(display, ('POINTS', 'SegmentationLabel'))
    lut = pv.GetColorTransferFunction('SegmentationLabel')

    rgb_points = build_colormap(min_label, max_label, colors)
    
    if rank == 0:
        logger.info(f"Using {len(rgb_points)//4} color points for transfer function")
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
            logger.info(f"Error setting color mapping properties: {e}")
            sys.stdout.flush()
    return

def save_state_file(output_dir, tag_str, rank):
    if rank == 0:
        os.makedirs(output_dir, exist_ok=True)
        state_file = os.path.join(output_dir, f'{tag_str}_state.pvsm')
        try:
            pv.SaveState(state_file)
            if os.path.exists(state_file):
                logger.info(f'ParaView state saved to: {state_file}')
            else:
                logger.info(f'Failed to save ParaView state to: {state_file}')
        except Exception as e:
            logger.info(f'Error saving ParaView state: {e}')
        sys.stdout.flush()

def save_colormap(output_dir, tag_str, min_label, max_label, rank, colors):
    if rank == 0:
        lut = pv.GetColorTransferFunction('SegmentationLabel')
        try:
            colormap_file = os.path.join(output_dir, f'{tag_str}_colormap.xml')
            pv.ExportLookupTable(FileName=colormap_file, LookupTable=lut)
            logger.info(f'Color map exported to: {colormap_file}')
        except:
            import xml.etree.ElementTree as ET
            import xml.dom.minidom
            os.makedirs(output_dir, exist_ok=True)
            rgb_points = build_colormap(min_label, max_label, colors)
            
            # Create XML structure following ParaView's documented format
            root = ET.Element('ColorMaps')
            colormap = ET.SubElement(root, 'ColorMap')
            colormap.set('name', f'Segmentation_{tag_str}')
            colormap.set('space', 'RGB')
            
            # Add color points - normalize RGB values to 0-1 range
            for i in range(0, len(rgb_points), 4):
                scalar_value = rgb_points[i]
                r = rgb_points[i+1] / 255.0  # Normalize to 0-1
                g = rgb_points[i+2] / 255.0  # Normalize to 0-1
                b = rgb_points[i+3] / 255.0  # Normalize to 0-1
                
                point = ET.SubElement(colormap, 'Point')
                point.set('x', f'{scalar_value:.3f}')
                point.set('o', '1')  # Full opacity
                point.set('r', f'{r:.6f}')
                point.set('g', f'{g:.6f}')
                point.set('b', f'{b:.6f}')
            
            # Pretty print the XML
            rough_string = ET.tostring(root, 'unicode')
            reparsed = xml.dom.minidom.parseString(rough_string)
            pretty_xml = reparsed.toprettyxml(indent='  ')
            
            # Remove empty lines and fix XML declaration
            lines = [line for line in pretty_xml.split('\n') if line.strip()]
            pretty_xml = '\n'.join(lines)
            
            xml_file = os.path.join(output_dir, f'{tag_str}_colormap.xml')
            with open(xml_file, 'w') as f:
                f.write(pretty_xml)
            logger.info(f'ParaView XML colormap saved to: {xml_file}')
        sys.stdout.flush()

def setup_camera(view, data_info, rank)-> None:
    if rank == 0:
        logger.info("Setting up camera to show full dataset...")
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
        logger.info(f"Dataset bounds: {data_bounds}")
        logger.info(f"Camera distance: {camera_distance}")
        sys.stdout.flush()
    return

def save_screenshot(output_file, rank)-> None:
    pv.Render()
    if rank == 0:
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        pv.SaveScreenshot(output_file)
        logger.info(f'Screenshot saved to: {output_file}')
        sys.stdout.flush()
    return 

def generate_animation(view, animation_dir, num_frames, rank)-> None:
    if rank == 0:
        logger.info("--- Starting Animation ---")
        os.makedirs(animation_dir, exist_ok=True)
        sys.stdout.flush()
        
    focal_point = view.CameraFocalPoint
    initial_position = view.CameraPosition
    radius_vector = [initial_position[i] - focal_point[i] for i in range(3)]
    radius = math.sqrt(sum([c*c for c in radius_vector]))

    angle_step = 2 * math.pi / num_frames
    focal_x, focal_y, focal_z = focal_point
    initial_y = initial_position[1]
    
    halfway = num_frames // 2
    for i in range(num_frames):
        angle = i * angle_step
        new_x = focal_x + radius * math.sin(angle)
        new_z = focal_z + radius * math.cos(angle)
        
        view.CameraPosition = [new_x, initial_y, new_z]
        if i == halfway:
            # ZOOM in, spin around, zoom out
            view.CameraPosition = [focal_x, initial_y, focal_z]
            view.CameraParallelScale = radius * 0.6
            view.CameraPosition = [focal_x, initial_y, focal_z]
        if i % 5 == 0:
            gc.collect()
        pv.Render()
        
        if rank == 0:
            frame_filename = os.path.join(animation_dir, f'frame_{i:04d}.png')
            pv.SaveScreenshot(frame_filename)
            logger.info(f"Saved frame {i+1}/{num_frames}")
            sys.stdout.flush()

    if rank == 0:
        logger.info(f"\nSaved {num_frames} frames to {animation_dir}")
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
    output_dir = args.output_dir
    tag_str = f'vtp_{embryonic_number}_{date.today()}{output_tag}'
    
    collection_file = os.path.join(surface_dir, 'combined_surface.pvd')
    output_file = os.path.join(args.output_dir, f'{tag_str}.png')
    animation_dir = os.path.join(args.output_dir, f'animation_frames_{tag_str}')

    GLASBEY_DARK_COLORS = remove_light_colors(all_hex_to_rgb(cc.glasbey_dark))
    GENERATE_ANIMATION = True

    # --- VALIDATE INPUT FILES ---
    if not os.path.exists(collection_file):
        if rank == 0:
            logger.info(f'Collection file not found: {collection_file}')
        sys.exit(1)

    # --- BEGIN VISUALIZATION PROCESS ---
    if rank == 0:
        logger.info(f"---------- Beginning Visualization of: E9.5_{args.embryonic_number} ----------")
        sys.stdout.flush()

    # --- DATA LOADING ---
    c2p, data_info = load_data(collection_file, rank, output_dir)
        
    segmentation_array_info = find_labels(c2p, rank)
    
    data_range = segmentation_array_info.GetComponentRange(0)
    min_label = int(data_range[0])
    max_label = int(data_range[1])
    
    if rank == 0:
        logger.info(f"SegmentationLabel range: {min_label} to {max_label}")
        logger.info(f"Total unique labels: {max_label - min_label + 1}")
        sys.stdout.flush()
    
    # --- VISUALIZATION SETUP ---
    display = setup_display(c2p, rank)
    view = setup_view(rank)
    apply_color_mapping(display, min_label, max_label, rank, GLASBEY_DARK_COLORS)
    setup_camera(view, data_info, rank)
    
    # --- SAVE VISUALIZATION STATE ---
    save_state_file(output_dir, tag_str, rank)
    save_colormap(output_dir, tag_str, min_label, max_label, rank, GLASBEY_DARK_COLORS)
    
    # --- SAVE DATA ---
    active_source = pv.GetActiveSource()
    pv.SaveData(f'{output_dir}/decimated_data.vtm',
                proxy=active_source,
                PointDataArrays=['Normals', 'SegmentationLabel'],
                Assembly='Hierarchy')
    
    # --- GENERATE MAIN SCREENSHOT ---
    save_screenshot(output_file, rank)

    # --- GENERATE ANIMATION ---
    gc.collect()
    if GENERATE_ANIMATION:
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
            logger.info(f"Time taken: {num_minutes} minutes")
            if result:
                print("="*40)
                logger.info("Optimized Visualization Ended Successfully")
                print("="*40)
            else:
                print("="*40)
                logger.info("Optimized Visualization Failed")
                print("="*40)
            sys.stdout.flush()
            
        if not result:
            sys.exit(1)
            
    except Exception as e:
        controller = servermanager.vtkProcessModule.GetProcessModule().GetGlobalController()
        rank = controller.GetLocalProcessId()
        
        if rank == 0:
            logger.info(f"Error during optimized VTM visualization: {e}")
            import traceback
            traceback.print_exc()
            sys.stdout.flush()
        sys.exit(1)