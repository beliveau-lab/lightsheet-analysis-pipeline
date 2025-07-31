import paraview.simple as pv
import os, sys
import random
import colorsys

def main():
    embryonic_number = 290
    if embryonic_number == 317:
        surface_dir = '/net/beliveau/vol2/instrument/E9.5_317/Zoom_317/surfaces_compressed_2'
    elif embryonic_number == 290:
        surface_dir = '/net/beliveau/vol2/instrument/E9.5_290/Zoom_290_subset_test/surfaces_compressed_2'
    else:
        print(f"No directory specified for embryonic number {embryonic_number}")
        sys.exit(1)
    collection_file = os.path.join(surface_dir, 'combined_surface.pvd')
    output_file = f'/net/beliveau/vol1/home/msforman/msf_project/lightsheet-analysis-pipeline/figures/vtp_{embryonic_number}_parallel_531.png'

    if not os.path.exists(collection_file):
        print(f'collection file not found: {collection_file}')
        sys.exit(1)

    print("Beginning visualization")

    # Load the data
    src = pv.PVDReader(FileName=collection_file)
    src.UpdatePipeline()
    
    # Get information about available arrays
    data_info = src.GetDataInformation()
    cell_data_info = data_info.GetCellDataInformation()  # Changed from GetPointDataInformation
    print(f"Data bounds: {data_info.GetBounds()}")
    print(f"Number of cells: {data_info.GetNumberOfCells()}")
    print(f"Number of points: {data_info.GetNumberOfPoints()}")
    
    # Check if SegmentationLabel exists
    array_names = [cell_data_info.GetArrayInformation(i).GetName() 
                   for i in range(cell_data_info.GetNumberOfArrays())]
    print(f"Available cell arrays: {array_names}")
    
    # Create display
    display = pv.Show(src)
    display.Representation = 'Surface'
    
    print("Coloring")
    
    if 'SegmentationLabel' in array_names:
        # Get the range of segmentation labels
        array_info = cell_data_info.GetArrayInformation(
            cell_data_info.GetArrayIndex('SegmentationLabel'))
        data_range = array_info.GetComponentRange(0)
        min_label = int(data_range[0])
        max_label = int(data_range[1])
        print(f"SegmentationLabel range: {min_label} to {max_label}")
        
        # Color by SegmentationLabel - using CELLS instead of POINTS
        pv.ColorBy(display, ('CELLS', 'SegmentationLabel'))
        
        # Get the color transfer function
        lut = pv.GetColorTransferFunction('SegmentationLabel')
        
        # Set up for categorical coloring
        lut.InterpretValuesAsCategories = 1
        
        # Generate 30 distinct colors using HSV color space
        n_colors = 30
        colors = []
        for i in range(n_colors):
            hue = i / n_colors
            # High saturation and value for vivid colors
            r, g, b = colorsys.hsv_to_rgb(hue, 0.9, 0.9)
            colors.append((r, g, b))
        
        # Shuffle for randomness
        random.shuffle(colors)
        
        # Clear existing color map and set new one
        lut.RGBPoints = []
        
        # If using categorical mode with specific label mapping
        if max_label - min_label < 1000:  # Reasonable number of labels
            for i, label in enumerate(range(min_label, max_label + 1)):
                color = colors[i % n_colors]
                # Add color point for this label
                lut.RGBPoints.extend([label, color[0], color[1], color[2]])
        else:
            # For very large label ranges, use a simplified approach
            lut.NumberOfTableValues = n_colors
            for i in range(n_colors):
                # Map evenly across the range
                val = min_label + (max_label - min_label) * i / (n_colors - 1)
                lut.RGBPoints.extend([val, colors[i][0], colors[i][1], colors[i][2]])
        
        # Apply the lookup table
        display.LookupTable = lut
        display.UseLookupTableScalarRange = 1
        
        # Hide scalar bar for cleaner image
        display.SetScalarBarVisibility(pv.GetActiveView(), False)
        
    else:
        print("SegmentationLabel not found in cell data. Using default coloring.")
        display.DiffuseColor = [0.8, 0.3, 0.2]
        pv.ColorBy(display, None)
    
    # Set up the view
    view = pv.GetActiveViewOrCreate('RenderView')
    view.ViewSize = [3000, 3000]
    view.Background = [1.0, 1.0, 1.0]  # White background
    
    # Set camera position
    view.CameraPosition = [1195.5, 1191.0, 10698.492594775726]
    view.CameraFocalPoint = [1195.5, 1191.0, 691.75]
    view.CameraParallelScale = 2640.732222949385
    
    # Render
    pv.Render()
    
    # Save screenshot
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    pv.SaveScreenshot(output_file, ImageResolution=[3000, 3000], 
                      TransparentBackground=0, ImageQuality=100)
    print('Finished visualization, wrote to:', output_file)
    return output_file

if __name__ == "__main__":
    try:
        result = main()
        if result:
            print("VTM visualization completed successfully!")
        else:
            print("VTM visualization failed!")
            sys.exit(1)
    except Exception as e:
        print(f"Error during VTM visualization: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


# import paraview.simple as pv
# import os, sys
# import math

# def main():
#     embryonic_number = 290
#     if embryonic_number == 317:
#         surface_dir = '/net/beliveau/vol2/instrument/E9.5_317/Zoom_317/surfaces_compressed_2'
#     elif embryonic_number == 290:
#         surface_dir = '/net/beliveau/vol2/instrument/E9.5_290/Zoom_290_subset_test/surfaces_compressed_2'
#     else:
#         print(f"No directory specified for embryonic number {embryonic_number}")
#         sys.exit(1)
#     collection_file = os.path.join(surface_dir, 'combined_surface.pvd')
#     output_file = f'/net/beliveau/vol1/home/msforman/msf_project/lightsheet-analysis-pipeline/figures/vtp_{embryonic_number}_parallel_531.png'

#     if not os.path.exists(collection_file):
#         print(f'collection file not found: {collection_file}')
#         sys.exit(1)

#     print("Beginning visualization")

#     src = pv.PVDReader(FileName=collection_file)   # one reader only
#     src.UpdatePipeline()
#     display = pv.Show(src)
#     display.Representation = 'Surface'

#     print("Coloring")
#     # try:
    
#     display.ColorArrayName = ['CELLS', 'SegmentationLabel']
#     # pv.ColorBy(display, ('CELLS', 'SegmentationLabel'))
#     # lut = pv.GetColorTransferFunction('SegmentationLabel')
#     # lut.InterpretValuesAsCategories = 1 
#     # n_colours = 30                                            # any smallish integer works
#     # lut.Discretize = 1
#     # lut.NumberOfTableValues = n_colours
#     # lut.Build()
#     # display.LookupTable = lut
#     pv.ColorBy(display, ('CELLS', 'SegmentationLabel'))
#     lut = pv.GetColorTransferFunction('SegmentationLabel')
#     lut.InterpretValuesAsCategories = 1
#     lut.ApplyPreset('Spectrum', True)  # or 'Rainbow', 'jet', etc.
#     lut.NumberOfTableValues = 30
#     display.LookupTable = lut

#     # except:
#     #     try:
#     #         display.ColorArrayName = ['POINTS', 'SegmentationLabel']
#     #     except:
#     # display.DiffuseColor = [0.8, 0.3, 0.2]
#     # pv.ColorBy(display, None)
#     # print("rendering")
#     # bounds = src.GetDataInformation().GetBounds()
#     # print("bounds:", bounds)    

#     view = pv.GetActiveViewOrCreate('RenderView')
#     view.ViewSize = [3000, 3000]
#     # view.Background = [0.1, 0.1, 0.2]
#     # src.UpdatePipeline()          # 1. make sure the reader really loads the data
#     # view.ResetCamera()            # 2. fit the whole dataset that is now in memory
#     # view.Zoom(15)                  # 3. adjust view (or move camera, change view-angle, etc.)
#     #The following camera settings are from a trace of the ParaView GUI
#     # which correctly visualized the data. Using these explicit settings
#     # instead of ResetCamera() ensures the object is framed correctly.
#     view.Set(
#         CameraPosition=[1195.5, 1191.0, 10698.492594775726],
#         CameraFocalPoint=[1195.5, 1191.0, 691.75],
#         CameraParallelScale=2640.732222949385,
#     )
    
#     pv.Render()                   # 5. draw and/or save screenshot

#     os.makedirs(os.path.dirname(output_file), exist_ok=True)
#     pv.SaveScreenshot(output_file)
#     print('Finished visualization, wrote to:', output_file)
#     return output_file

# if __name__ == "__main__":
#     try:
#         result = main()
#         if result:
#             print("VTM visualization completed successfully!")
#         else:
#             print("VTM visualization failed!")
#             sys.exit(1)
#     except Exception as e:
#         print(f"Error during VTM visualization: {e}")
#         sys.exit(1) 


