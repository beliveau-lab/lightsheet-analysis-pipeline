#!/usr/bin/env python3
"""
Distributed ParaView visualization using dask.distributed for large zarr files.
Processes chunks in parallel and combines VTK surfaces.
"""

import numpy as np
import dask.array as da
import dask
from dask import delayed
# from dask.distributed import Client
import zarr
import vtk
# from vtk.util import numpy_support
# import paraview.simple as pv
import logging
import os
# import glob
from utils.dask_utils import setup_dask_sge_cluster, shutdown_dask


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

import xml.etree.ElementTree as ET

def write_pvd_collection(vtp_paths, output_path, timestep=0):
    """
    Generate a ParaView collection (.pvd) that references many .vtp pieces.

    Parameters
    ----------
    vtp_paths : list[str]
        Paths to the per-chunk .vtp files.
    output_path : str
        Destination .pvd file.
    timestep : float or int
        Time value stored for each piece (ParaView needs one).
    """
    root = ET.Element("VTKFile", type="Collection", version="0.1")
    coll = ET.SubElement(root, "Collection")

    for path in sorted(vtp_paths):
        ET.SubElement(coll, "DataSet",
                      timestep=str(timestep),
                      group="",
                      part="0",
                      file=os.path.relpath(path, os.path.dirname(output_path)))

    tree = ET.ElementTree(root)
    tree.write(output_path, encoding="utf-8", xml_declaration=True)
    print(f"Wrote collection file with {len(vtp_paths)} pieces → {output_path}")
    return

# --- Simplified N5/Zarr loading ---
def load_n5_zarr_array(path, n5_subpath=None, chunks=None):
    logger.info(f"Attempting to load from: {path}" + (f" with N5 subpath: {n5_subpath}" if n5_subpath else ""))
    if path.endswith('.n5'):
        store = zarr.N5Store(path)
        if not n5_subpath:
             raise ValueError(f"N5 path specified ({path}), but n5_subpath is required.")
        arr_handle = zarr.open_array(store=store, path=n5_subpath, mode='r')
        logger.info(f"Loaded N5 array: Shape={arr_handle.shape}, Chunks={arr_handle.chunks}")
        return da.from_zarr(arr_handle, chunks=chunks)
    elif path.endswith('.zarr'):
        arr_handle = zarr.open(path, mode='r')
        logger.info(f"Loaded Zarr array: Shape={arr_handle.shape}")
        return da.from_zarr(path, chunks=chunks)
    else:
        raise ValueError(f"Unsupported array format (expected .n5 or .zarr): {path}")

def process_chunk_to_contours(chunk_data, chunk_offset, out_dir, block_id):
    """Process single chunk to extract contour surfaces for objects."""
    import vtk
    from vtk.util import numpy_support
    
    unique_labels = np.unique(chunk_data)
    major_labels = unique_labels[unique_labels > 0]
    
    if len(major_labels) == 0:
        return None
    
    # Ensure the output directory exists on the worker
    os.makedirs(out_dir, exist_ok=True)
    surfaces = []  # (chunk, vtkPolyData) pairs
    for label in major_labels: 
        binary_data = (chunk_data == label).astype(np.uint8)
        
        # create VTK image data
        img_data = vtk.vtkImageData()
        img_data.SetDimensions(*binary_data.shape[::-1])  # VTK uses XYZ order
        img_data.SetOrigin(*chunk_offset[::-1])
        img_data.SetSpacing(1.0, 1.0, 1.0)
        
        # Add data
        vtk_array = numpy_support.numpy_to_vtk(binary_data.ravel(), deep=False) # switched from true
        img_data.GetPointData().SetScalars(vtk_array)
        
        # Generate contour surface
        contour = vtk.vtkContourFilter()
        contour.SetInputData(img_data)
        contour.SetValue(0, 0.5)
        contour.Update()
        
        surface = contour.GetOutput()
        if surface.GetNumberOfPoints() > 0:
            label_array = vtk.vtkIntArray() # add label as scalar
            label_array.SetName("SegmentationLabel")
            # label_array.SetNumberOfTuples(surface.GetNumberOfPoints())
            # label_array.Fill(label)
            # surface.GetPointData().SetScalars(label_array)
            label_array.SetNumberOfTuples(surface.GetNumberOfCells())   # 1 value per polygon (or object)
            label_array.FillComponent(0, label)                         # fill with current label
            surface.GetCellData().SetScalars(label_array)        
            surfaces.append((label, surface))

    if len(surfaces) == 0:
        return None

    # mb = vtk.vtkMultiBlockDataSet()
    # for idx, (lbl, surf) in enumerate(surfaces):
    #     mb.SetBlock(idx, surf)
    #     meta = mb.GetMetaData(idx)
    #     meta.Set(vtk.vtkCompositeDataSet.NAME(), f"label_{lbl}")

    # file_name = f"block_{'_'.join(map(str, block_id))}.vtm"
    append = vtk.vtkAppendPolyData()
    for _, surf in surfaces:
        append.AddInputData(surf)
    append.Update()

    file_name = f"block_{'_'.join(map(str, block_id))}.vtp"
    file_path = os.path.join(out_dir, file_name)

    writer = vtk.vtkXMLPolyDataWriter()
    writer.SetFileName(file_path)
    writer.SetInputData(append.GetOutput())
    writer.SetDataModeToAppended() # binary “appended-raw” section
    writer.SetCompressorTypeToZLib()
    writer.SetCompressionLevel(9)
    writer.Write()

    return [{"path": file_path, "labels": [int(l) for l, _ in surfaces]}]

def distributed_surface_extraction(zarr_path, client, out_dir, downsample=0):
    """Extract surfaces from zarr in parallel chunks. Submitted to dask workers"""
    logger.info(f"Loading zarr: {zarr_path}")
    arr = load_n5_zarr_array(zarr_path)
    
    if downsample > 1:
        arr = arr[::downsample, ::downsample, ::downsample]
    
    chunk_shape = tuple(c[0] for c in arr.chunks)  # Extract chunk shape for reference
    logger.info(f"Loaded Mask (ZYX assumed): Shape={arr.shape}, Dtype={arr.dtype}, Chunk Shape: {chunk_shape}")

    delayed_tasks = []
    for block_id in np.ndindex(*arr.numblocks):
        # Calculate chunk slice and offset
        chunk_slices = tuple(
            slice(sum(arr.chunks[i][:block_id[i]]), 
                  sum(arr.chunks[i][:block_id[i]+1]))
            for i in range(arr.ndim)
        )
        chunk_offset = tuple(sum(arr.chunks[i][:block_id[i]]) for i in range(arr.ndim))
        chunk = arr[chunk_slices]
        
        # Create delayed task – each worker writes its own .vtp files and returns lightweight metadata only.
        task = delayed(process_chunk_to_contours)(chunk, chunk_offset, out_dir, block_id)
        delayed_tasks.append(task)
    
    logger.info(f"Submitting {len(delayed_tasks)} chunk processing tasks...")
    
    chunk_results = client.compute(delayed_tasks, sync=True)
    metadata_list = []
    for chunk_meta in chunk_results:
        if chunk_meta:
            metadata_list.extend(chunk_meta)

    logger.info(f"Generated {len(metadata_list)} surface files")
    return metadata_list

def _setup_dask(params):
    """Setup Dask SGE cluster for distributed processing."""
    try:
        logger.info("Setting up distributed Dask cluster...")
        cluster, client = setup_dask_sge_cluster(
            n_workers=params.get('num_workers', 1),
            cores=params.get('cpu_cores', 4),
            processes=params.get('cpu_processes', 1),
            memory=params.get('cpu_memory', '60G'),
            project=params.get('project', 'beliveaulab'),
            queue=params.get('queue', 'beliveau-long.q'),
            runtime=params.get('runtime', '7200'),
            resource_spec=params.get('cpu_resource_spec', 'mfree=60G'),
            log_directory=params.get('log_dir', None),
            conda_env=params.get('conda_env', 'otls-pipeline'),
            dashboard_port=params.get('dashboard_port', None)
        )
        logger.info(f"Dask dashboard link: {client.dashboard_link}")
        return cluster, client
    except Exception as e:
        logger.error(f"Failed to setup distributed cluster: {e}")
        return False

def _shutdown_dask(cluster, client):
    """Shutdown Dask cluster."""
    if cluster and client:
        try:
            shutdown_dask(cluster, client)
            logger.info("Dask cluster shut down successfully")
        except Exception as e:
            logger.error(f"Error shutting down cluster: {e}")
            
def main():
    params = {
    # -- dask parameters --
    "num_workers": 1,
    "cpu_memory": "256G", 
    "cpu_cores": 16,
    "cpu_processes": 32, # 2 proc per core
    "cpu_resource_spec": "mfree=16G",  # RAM/worker = RAM/core * cores/worker (16G/core * 1 core/2 proc = 8G/proc)
    # -- tracking parameters --
    'use_dask': True,
    "log_dir" : '/net/beliveau/vol1/home/msforman/msf_project/lightsheet-analysis-pipeline/logs/visualization/',
    "save_dir": '/net/beliveau/vol1/home/msforman/msf_project/lightsheet-analysis-pipeline/figures/',
    "dashboard_port": ":41263",
    # -- task specific parameters --
    "zarr_path": '/net/beliveau/vol2/instrument/E9.5_317/Zoom_317/dataset_fused_masks.zarr',
    # "zarr_path": '/net/beliveau/vol2/instrument/E9.5_290/Zoom_290_subset_test/dataset_fused_masks_cpsamr5.zarr',
    "downsample": 0,
    "surface_out_dir": '/net/beliveau/vol2/instrument/E9.5_317/Zoom_317/surfaces_compressed_2',
    # "surface_out_dir": '/net/beliveau/vol2/instrument/E9.5_290/Zoom_290_subset_test/surfaces_compressed_2',
    }
        
    cluster, client = _setup_dask(params)
    try:
        flag = 1
        if flag:
            logger.info(f"Dask dashboard: {client.dashboard_link}")
            # Distributed surface extraction – each worker writes compressed .vtp
            # files; only metadata is returned.
            metadata = distributed_surface_extraction(
                params.get('zarr_path'),
                client,
                params.get('surface_out_dir'),
                downsample=params.get('downsample', 0)
            )
            vtp_files = [m['path'] for m in metadata]
            # collect all vtp files into one pvd
            collection_path = os.path.join(params["surface_out_dir"], "combined_surface.pvd")
            write_pvd_collection(vtp_files, collection_path)
            logger.info(
                f"Extraction complete. {len(metadata)} surface files written to {params.get('surface_out_dir')}"
            )

    finally:
        _shutdown_dask(cluster, client)

if __name__ == "__main__":
    main()