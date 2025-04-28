import sys
import argparse
import dask.array as da
import numpy as np
from dask import delayed
from dask.distributed import Client, progress, get_worker
import zarr
from skimage import measure
import pandas as pd
# from scipy.ndimage import center_of_mass # Removed unused imports (already removed)
import time
import logging
from contextlib import contextmanager
# from dask_jobqueue import SGECluster # Removed direct import
# from tifffile import imread # Assuming Zarr/N5 input now
# import matplotlib.pyplot as plt # Removed plotting imports for production
# from mpl_toolkits.mplot3d import Axes3D
import os
# import dask.distributed as dd # Removed unused import
from datetime import datetime
import dask
import gc # Garbage collector interface
from tifffile import imread
import cloudpickle
import platform
import distributed
import msgpack
import skimage
# from dask.delayed import Delayed # Removed unused import

# --- Import Dask utility functions ---
from utils.dask_utils import setup_dask_sge_cluster, shutdown_dask

# --- Environment Logging (Add this near the start of your script) ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
# Optional: Set higher level for noisy logs if needed, but keep INFO for our messages
# logging.getLogger('distributed').setLevel(logging.WARNING)
logger = logging.getLogger(__name__) # Use __name__ for logger

# --- Log Environment Info Once ---
def log_environment():
    logger.info("--- Environment Versions ---")
    logger.info(f"Platform: {platform.platform()}")
    logger.info(f"Python: {sys.version}")
    logger.info(f"Dask: {dask.__version__}")
    logger.info(f"Distributed: {distributed.__version__}")
    logger.info(f"Cloudpickle: {cloudpickle.__version__}")
    logger.info(f"Msgpack: {msgpack.__version__}")
    logger.info(f"Zarr: {zarr.__version__}")
    logger.info(f"NumPy: {np.__version__}")
    logger.info(f"Scikit-image: {skimage.__version__}")
    logger.info("--- Dask Config (relevant parts) ---")
    logger.info(f"distributed.comm.compression: {dask.config.get('distributed.comm.compression')}")
    # Add any other specific config settings you might have changed
    logger.info("--- End Environment Info ---")

# Call environment logging at the start
log_environment()


@contextmanager
def timer(description, chunk_id=None, extra_info=None):
    """Context manager for timing code blocks with detailed logging."""
    start = time.time()
    yield
    elapsed = time.time() - start

    log_msg = f"{description} took {elapsed:.2f} seconds"
    if chunk_id is not None:
        log_msg = f"Chunk {chunk_id}: {log_msg}"
    if extra_info:
        log_msg = f"{log_msg} | {extra_info}"

    logger.info(log_msg)


# Function to estimate serialized size safely
def get_pickle_size(obj):
    try:
        return len(cloudpickle.dumps(obj))
    except Exception as e:
        # Be more specific about the object causing failure if possible
        try:
            obj_repr = repr(obj)
            if len(obj_repr) > 200: # Limit length of representation
                 obj_repr = obj_repr[:200] + "..."
        except Exception:
            obj_repr = "[repr failed]"
        logger.error(f"Failed to pickle object of type {type(obj)} ({obj_repr}): {e}", exc_info=True) # Log traceback
        return -1 # Indicate failure


# --- Simplified N5/Zarr loading ---
def load_n5_zarr_array(path, n5_subpath=None):
    """Loads N5 or Zarr, handling potential subpath for N5."""
    logger.info(f"Attempting to load from: {path}" + (f" with N5 subpath: {n5_subpath}" if n5_subpath else ""))
    if path.endswith('.n5'):
        store = zarr.N5Store(path)
        if not n5_subpath:
             raise ValueError(f"N5 path specified ({path}), but n5_subpath is required.")
        arr_handle = zarr.open_array(store=store, path=n5_subpath, mode='r')
        logger.info(f"Loaded N5 array: Shape={arr_handle.shape}, Chunks={arr_handle.chunks}")
        return da.from_zarr(arr_handle)
    elif path.endswith('.zarr'):
        arr_handle = zarr.open(path, mode='r')
        logger.info(f"Loaded Zarr array: Shape={arr_handle.shape}, Chunks={arr_handle.chunks}")
        return da.from_zarr(path)
    else:
        raise ValueError(f"Unsupported array format (expected .n5 or .zarr): {path}")

# --- TIF Loading (kept separate for clarity, uses Dask array creation) ---
def load_tif_as_dask_array(tif_path, processing_chunk_size=None, expected_dims=None):
    """Loads TIF into memory, creates a Dask array with specified chunks."""
    logger.info(f"Loading TIF image from: {tif_path}")
    img_np = imread(tif_path)
    logger.info(f"Loaded TIF image with shape: {img_np.shape}")

    # Dimension handling (example for CZYX)
    if expected_dims == 'CZYX':
        if img_np.ndim == 4 and img_np.shape[-1] < min(img_np.shape[0], img_np.shape[1], img_np.shape[2]):
            logger.info("Assuming TIF is ZYXC, transposing to CZYX...")
            img_np = np.transpose(img_np, (3, 0, 1, 2))
        elif img_np.ndim == 3:
            logger.info("Detected 3D TIF, adding channel dimension (assuming CZYX).")
            img_np = img_np[np.newaxis, ...]
        elif img_np.ndim != 4:
            raise ValueError(f"Expected 4D CZYX TIF, but got dimensions: {img_np.ndim}")
    elif expected_dims == 'ZYX':
        if img_np.ndim != 3:
             raise ValueError(f"Expected 3D ZYX TIF (for mask), but got dimensions: {img_np.ndim}")
    else:
        # No specific expectation, just log
        logger.info(f"Processing TIF with shape {img_np.shape} (no specific dim expectation)")

    logger.info(f"NumPy array shape after potential transpose/reshape: {img_np.shape}")

    # --- Create Dask Array with Explicit Chunks ---
    if processing_chunk_size:
        # Ensure chunk size matches array dimensions
        if len(processing_chunk_size) != img_np.ndim:
             raise ValueError(f"processing_chunk_size dimensions ({len(processing_chunk_size)}) "
                              f"must match image dimensions ({img_np.ndim})")

        final_chunks = []
        for i, dim_size in enumerate(img_np.shape):
            final_chunks.append(min(processing_chunk_size[i], dim_size))
        final_chunks = tuple(final_chunks)
        logger.info(f"Using explicit chunks for da.from_array: {final_chunks}")
    else:
        logger.warning("No processing_chunk_size hint provided, using automatic chunking for da.from_array.")
        final_chunks = 'auto'

    dask_array = da.from_array(img_np, chunks=final_chunks)
    logger.info(f"Created Dask array. Shape: {dask_array.shape}, Chunks: {dask_array.chunksize}")

    # Pickle size check (optional but useful for debugging large headers)
    pickle_size_test = get_pickle_size(dask_array)
    logger.info(f"   Pickle size test of da.from_array object: {pickle_size_test} bytes")
    if pickle_size_test < 0 or pickle_size_test > 5 * 1024 * 1024: # Check if > 5MB
        logger.warning(f"Pickle size of Dask array created with from_array is large ({pickle_size_test} bytes). This might indicate large metadata.")

    del img_np
    gc.collect()
    return dask_array


def compute_chunk_properties(image_chunk, mask_chunk, chunk_offset, core_relative_end_coords, full_image_spatial_shape, num_channels, voxel_size=(1.0, 1.0, 1.0)):
    """
    Compute region properties using skimage's regionprops, filtering results *after* computation.
    Optimized to avoid per-label iteration before regionprops.

    Parameters:
    -----------
    image_chunk : ndarray
        Image data including overlap area. Should be ZYXC.
    mask_chunk : ndarray
        Mask data including overlap area. Assumed uint32, ZYX.
    chunk_offset : tuple
        Absolute (z, y, x) coordinates of the chunk's starting position in the full image.
    core_relative_end_coords : tuple
        (z, y, x) coordinates defining the *exclusive* end of the core region, relative to the start of `mask_chunk`.
    full_image_spatial_shape : tuple
        (z, y, x) dimensions of the full image.
    num_channels : int
        Number of channels in the image.
    voxel_size : tuple, optional
        Physical size of a voxel.
    """
    chunk_id = f"{chunk_offset}"
    worker = None
    try:
        worker = get_worker()
        worker_id = worker.id
    except ValueError:
        worker_id = "local"

    log_prefix = f"Worker {worker_id} | Chunk {chunk_id}"
    start_time = time.time()
    # Ensure image_chunk is ZYXC and mask_chunk is ZYX
    if image_chunk.ndim != 4 or image_chunk.shape[-1] != num_channels:
        logger.error(f"{log_prefix}: Unexpected image chunk shape {image_chunk.shape}. Expected ZYXC with {num_channels} channels.")
        return []
    if mask_chunk.ndim != 3:
        logger.error(f"{log_prefix}: Unexpected mask chunk shape {mask_chunk.shape}. Expected ZYX.")
        return []
    if image_chunk.shape[:3] != mask_chunk.shape:
        logger.error(f"{log_prefix}: Image spatial shape {image_chunk.shape[:3]} doesn't match mask shape {mask_chunk.shape}")
        return []


    logger.info(f"{log_prefix}: Processing image chunk {image_chunk.shape}, mask chunk {mask_chunk.shape}. Mask dtype: {mask_chunk.dtype}. Core relative end: {core_relative_end_coords}")
    # print(f"{log_prefix}: Processing image chunk {image_chunk.shape}, mask chunk {mask_chunk.shape}. Mask dtype: {mask_chunk.dtype}. Core relative end: {core_relative_end_coords}")


    properties = []
    processed_count = 0
    min_voxel_size = 0 # Example minimum size filter

    try:
        # --- Step 1: Run regionprops ---
        # regionprops expects label_image (spatial) and intensity_image (spatial + channels)
        label_image_for_props = mask_chunk.astype(np.int32, copy=False) # Ensures int type, avoids copy if possible
        # print(f"{log_prefix}: Running regionprops on mask {label_image_for_props.shape} with intensity image {image_chunk.shape}...")
        rp_start_time = time.time()
        # Pass the 4D image_chunk (ZYXC) directly as intensity_image
        props_list = measure.regionprops(label_image_for_props, intensity_image=image_chunk)
        rp_end_time = time.time()
        # print(f"{log_prefix}: regionprops completed in {rp_end_time - rp_start_time:.2f} seconds, found {len(props_list)} potential objects.")
        del label_image_for_props
        gc.collect()

       # --- Step 2: Iterate through results and filter ---
        with timer(f"{log_prefix} Filtering regionprops", extra_info=f"{len(props_list)} objects"):
            core_z_max_rel, core_y_max_rel, core_x_max_rel = core_relative_end_coords

            for prop in props_list:
                label = prop.label
                if label == 0: continue # Skip background if present

                # Bbox from regionprops is (min_row, min_col, ..., max_row, max_col, ...)
                # For 3D ZYX: (min_z, min_y, min_x, max_z, max_y, max_x) - *inclusive* max
                bbox_rel = prop.bbox
                z_min_rel, y_min_rel, x_min_rel = bbox_rel[0], bbox_rel[1], bbox_rel[2]
                # regionprops bbox max is inclusive, our core boundaries are exclusive
                z_max_rel_incl, y_max_rel_incl, x_max_rel_incl = bbox_rel[3]-1, bbox_rel[4]-1, bbox_rel[5]-1 # zero-based inclusive max index

                # --- Check 1: Does the object touch an internal chunk border? ---
                # Check if min coord is 0 AND the chunk is not at the global image start
                touches_z_start = (z_min_rel == 0) and (chunk_offset[0] > 0)
                touches_y_start = (y_min_rel == 0) and (chunk_offset[1] > 0)
                touches_x_start = (x_min_rel == 0) and (chunk_offset[2] > 0)
                # Check if *inclusive* max index touches the chunk boundary AND chunk is not at global end
                touches_z_end = (z_max_rel_incl == mask_chunk.shape[0] - 1) and (chunk_offset[0] + mask_chunk.shape[0] < full_image_spatial_shape[0])
                touches_y_end = (y_max_rel_incl == mask_chunk.shape[1] - 1) and (chunk_offset[1] + mask_chunk.shape[1] < full_image_spatial_shape[1])
                touches_x_end = (x_max_rel_incl == mask_chunk.shape[2] - 1) and (chunk_offset[2] + mask_chunk.shape[2] < full_image_spatial_shape[2])
                touches_internal_border = touches_z_start or touches_y_start or touches_x_start or touches_z_end or touches_y_end or touches_x_end

                # --- Check 2: Is the object's centroid contained within the relative core region? ---
                # Alternative: Check if bbox is within core, but centroid is often preferred
                centroid_rel = np.asarray(prop.centroid) # Z, Y, X relative to chunk start
                is_centroid_in_core = (centroid_rel[0] >= 0 and centroid_rel[0] < core_z_max_rel and
                                       centroid_rel[1] >= 0 and centroid_rel[1] < core_y_max_rel and
                                       centroid_rel[2] >= 0 and centroid_rel[2] < core_x_max_rel)

                # --- Check 3: Is the object large enough? ---
                volume = int(prop.area) # prop.area should give voxel count for 3D
                is_large_enough = volume >= min_voxel_size

                # --- Decision: Process if centroid is in core and object doesn't touch border (OR is large enough?) ---
                # Original logic used bbox containment, let's stick to centroid for simplicity now.
                # If an object touches the border, it will be fully captured by the chunk where its centroid lies.
                # If the centroid is in the core, we process it fully here.
                if is_centroid_in_core and not touches_internal_border and is_large_enough:
                    processed_count += 1
                    try:
                        # --- Non-intensity properties ---
                        centroid_abs = centroid_rel + np.asarray(chunk_offset)
                        major_axis_length = float(prop.major_axis_length) if hasattr(prop, 'major_axis_length') else 0.0
                        # Minor axis calculation can fail for ~spherical objects in 3D
                        minor_axis_length = float(prop.minor_axis_length) if hasattr(prop, 'minor_axis_length') else 0.0
                        if minor_axis_length <= 1e-6: minor_axis_length = 0.0 # Handle near-zero cases
                        # Intermediate axis not directly available in regionprops basic props

                        elongation = major_axis_length / minor_axis_length if minor_axis_length > 0 else 0.0
                        # Flatness would require moments_central or inertia_tensor, not calculated by default

                        # --- Intensity properties (PER CHANNEL) ---
                        # These should return arrays of shape (C,) for intensity_image ZYXC
                        # Handle potential absence if intensity image wasn't used correctly
                        mean_intensities = prop.mean_intensity if hasattr(prop, 'mean_intensity') else np.zeros(num_channels)
                        max_intensities = prop.max_intensity if hasattr(prop, 'max_intensity') else np.zeros(num_channels)
                        min_intensities = prop.min_intensity if hasattr(prop, 'min_intensity') else np.zeros(num_channels)

                        # Calculate std dev per channel
                        std_intensities = np.zeros(num_channels, dtype=np.float32)
                        if prop.coords.shape[0] > 0: # Need at least one voxel
                            coords_spatial = prop.coords # Shape (N_voxels, 3) Z,Y,X indices relative to chunk
                            # Extract all intensities for the object: shape (N_voxels, C) using ZYXC indexing
                            # Correct indexing: image_chunk[Z_indices, Y_indices, X_indices, :]
                            all_object_intensities = image_chunk[coords_spatial[:, 0], coords_spatial[:, 1], coords_spatial[:, 2], :]
                            # Calculate std dev across the voxel axis (axis=0)
                            if all_object_intensities.shape[0] > 1: # Need >1 voxel for std dev
                                std_intensities = np.std(all_object_intensities, axis=0)
                            del all_object_intensities

                        # --- Assemble properties dict with channel suffixes ---
                        props_dict = {
                            'label': label, 'volume': volume,
                            'centroid_z': float(centroid_abs[0]), 'centroid_y': float(centroid_abs[1]), 'centroid_x': float(centroid_abs[2]),
                            'major_axis_length': major_axis_length,
                            # 'intermediate_axis_length': 0.0, # Placeholder
                            'minor_axis_length': minor_axis_length,
                            'elongation': elongation,
                            # 'flatness': 0.0, # Placeholder
                            # Bbox relative to global coords (inclusive end)
                            'bbox_z_min': int(z_min_rel + chunk_offset[0]),
                            'bbox_y_min': int(y_min_rel + chunk_offset[1]),
                            'bbox_x_min': int(x_min_rel + chunk_offset[2]),
                            'bbox_z_max': int(z_max_rel_incl + chunk_offset[0]),
                            'bbox_y_max': int(y_max_rel_incl + chunk_offset[1]),
                            'bbox_x_max': int(x_max_rel_incl + chunk_offset[2]),
                        }
                        # Add per-channel intensity stats
                        for ch in range(num_channels):
                            props_dict[f'mean_intensity_c{ch}'] = float(mean_intensities[ch])
                            props_dict[f'max_intensity_c{ch}'] = float(max_intensities[ch])
                            props_dict[f'min_intensity_c{ch}'] = float(min_intensities[ch])
                            props_dict[f'std_intensity_c{ch}'] = float(std_intensities[ch])

                        properties.append(props_dict)

                    except Exception as e_prop_extract:
                        logger.error(f"{log_prefix}: Label {label}: Failed to extract/process properties. Error: {e_prop_extract}", exc_info=True)
                        # print(f"{log_prefix}: Label {label}: Failed to extract/process properties. Error: {e_prop_extract}", exc_info=True)
                        continue # Skip this object
    except Exception as e:
        logger.error(f"{log_prefix}: Error during regionprops or filtering for chunk: {e}", exc_info=True)
        # print(f"{log_prefix}: Error processing chunk: {e}", exc_info=True)
        return [] # Return empty list on chunk error

    # Log summary for this chunk
    total_time = time.time() - start_time
    avg_time = total_time / processed_count if processed_count > 0 else 0
    logger.info(f"{log_prefix}: Successfully processed chunk in {total_time:.2f}s. Found {processed_count} complete objects (avg {avg_time:.3f}s/obj)")
    # print(f"{log_prefix}: Successfully processed chunk in {total_time:.2f}s. Found {processed_count} complete objects (avg {avg_time:.3f}s/obj)")

    # Explicitly release memory before returning
    del image_chunk, mask_chunk, props_list
    gc.collect()

    return properties

# --- Modified: Takes client as argument ---
def compute_distributed_properties(image_source, mask_source, client: distributed.Client, chunk_size=(128, 128, 128), overlap=(32, 32, 32)):
    """
    Compute region properties using Dask with distributed processing.

    Parameters:
    -----------
    image_source : dask.array.Array
        Dask array for image data (CZYX).
    mask_source : dask.array.Array
        Dask array for mask data (ZYX).
    client : distributed.Client
        Active Dask client for computation.
    chunk_size : tuple, optional
        Base size of each *core* processing chunk (without overlap).
    overlap : tuple, optional
        Size of overlap region in each dimension.
    """
    logger.info("Starting distributed properties computation...")
    # --- Data Loading / Validation ---
    if not isinstance(image_source, da.Array):
        raise TypeError("image_source must be a Dask Array")
    if not isinstance(mask_source, da.Array):
        raise TypeError("mask_source must be a Dask Array")

    image = image_source
    mask = mask_source

    if image.ndim != 4:
         raise ValueError(f"Input image dask array must be 4D (C, Z, Y, X), got {image.ndim}")
    if mask.ndim != 3:
         raise ValueError(f"Input mask dask array must be 3D (Z, Y, X), got {mask.ndim}")

    # --- Transpose image to ZYXC for regionprops intensity_image ---
    logger.info(f"Transposing image from CZYX {image.shape} to ZYXC for regionprops compatibility...")
    image = da.transpose(image, (1, 2, 3, 0))
    logger.info(f"Image shape is now ZYXC: {image.shape}")


    full_image_spatial_shape = image.shape[:-1] # Z, Y, X
    num_channels = image.shape[-1] # C
    logger.info(f"Processing {num_channels} channels.")
    logger.info(f"Image spatial shape (ZYX): {full_image_spatial_shape}")
    logger.info(f"Mask shape (ZYX): {mask.shape}")

    # --- Pre-loop Dask Array Checks ---
    logger.info("--- Pre-loop Dask Array Checks ---")
    logger.info(f"Image check: type={type(image)}, shape={image.shape}, chunks={image.chunksize}, dtype={image.dtype}")
    # logger.info(f"Image graph complexity estimate: {len(image.dask.items())} tasks") # Can be large
    logger.info(f"Mask check: type={type(mask)}, shape={mask.shape}, chunks={mask.chunksize}, dtype={mask.dtype}")
    # logger.info(f"Mask graph complexity estimate: {len(mask.dask.items())} tasks")

    if full_image_spatial_shape != mask.shape:
        logger.error("Image spatial dimensions (ZYX) and Mask shape (ZYX) do not match!")
        logger.error(f"Image spatial shape: {full_image_spatial_shape}")
        logger.error(f"Mask shape: {mask.shape}")
        raise ValueError("Image and Mask spatial dimensions mismatch.")


    logger.info(f"Core chunk size: {chunk_size}")
    logger.info(f"Overlap size: {overlap}")
    # logger.info(f"Data chunking (blocksize): Image={image.chunksize}, Mask={mask.chunksize}") # Log underlying chunking

    # --- Prepare Delayed Computations ---
    logger.info("Creating computation graph using dask.delayed...")
    delayed_results = []
    chunk_coords = [] # Keep track for logging

    # Iterate through the *core* chunk indices based on spatial dimensions (ZYX)
    for z in range(0, full_image_spatial_shape[0], chunk_size[0]):
        for y in range(0, full_image_spatial_shape[1], chunk_size[1]):
            for x in range(0, full_image_spatial_shape[2], chunk_size[2]):

                # Calculate core region start/end (exclusive end)
                z_core_start, y_core_start, x_core_start = z, y, x
                z_core_end = min(z + chunk_size[0], full_image_spatial_shape[0])
                y_core_end = min(y + chunk_size[1], full_image_spatial_shape[1])
                x_core_end = min(x + chunk_size[2], full_image_spatial_shape[2])

                # Calculate read region start/end including overlap (exclusive end)
                z_read_start = max(0, z_core_start - overlap[0])
                y_read_start = max(0, y_core_start - overlap[1])
                x_read_start = max(0, x_core_start - overlap[2])

                z_read_end = min(full_image_spatial_shape[0], z_core_end + overlap[0])
                y_read_end = min(full_image_spatial_shape[1], y_core_end + overlap[1])
                x_read_end = min(full_image_spatial_shape[2], x_core_end + overlap[2])

                # Define SPATIAL read slice (ZYX)
                read_slice_spatial = (slice(z_read_start, z_read_end),
                                      slice(y_read_start, y_read_end),
                                      slice(x_read_start, x_read_end))

                # Define FULL read slice for image (ZYXC - select ALL channels)
                read_slice_image = read_slice_spatial + (slice(None), )

                # Calculate core end coordinates relative to the start of the *read* chunk
                core_rel_z_end = z_core_end - z_read_start
                core_rel_y_end = y_core_end - y_read_start
                core_rel_x_end = x_core_end - x_read_start
                core_relative_end_coords_param = (core_rel_z_end, core_rel_y_end, core_rel_x_end)

                # Absolute offset of the read chunk
                read_chunk_offset = (z_read_start, y_read_start, x_read_start) # Spatial offset

                # --- Create Delayed Task ---
                # Pass the slices to index the Dask arrays *within* the delayed function
                chunk_result = delayed(compute_chunk_properties)(
                    image[read_slice_image],       # Index image (ZYXC) array
                    mask[read_slice_spatial],      # Index mask (ZYX) array
                    read_chunk_offset,
                    core_relative_end_coords_param,
                    full_image_spatial_shape,
                    num_channels
                )
                delayed_results.append(chunk_result)
                chunk_coords.append(read_chunk_offset) # Log chunk coords

    total_chunks = len(delayed_results)
    logger.info(f"Created {total_chunks} delayed tasks for computation.")

    # logger.info("Estimate size of the list containing all delayed objects...")
    # total_list_pickle_size = get_pickle_size(delayed_results)
    # logger.info(f"Pickle size of the list of delayed objects: {total_list_pickle_size} bytes") # Can be large

    # --- Execute Computations using the provided client ---
    logger.info("Submitting computations to Dask cluster...")
    future = client.compute(delayed_results)
    logger.info("Waiting for results... Use Dask dashboard to monitor progress.")
    progress(future) # Display text-based progress bar

    logger.info("Gathering results...")
    results = client.gather(future) # Blocks until completion, gather results to client
    logger.info("Computations finished. Results gathered.")

    # --- Process Results ---
    # Flatten results (list of lists of dictionaries)
    flat_results = [item for sublist in results if sublist for item in sublist] # Handles empty/failed chunks
    logger.info(f"Gathered {len(flat_results)} property dictionaries from all chunks.")

    if not flat_results:
        logger.warning("No objects found or processed across all chunks!")
        return pd.DataFrame()

    logger.info("Creating Pandas DataFrame...")
    df = pd.DataFrame(flat_results)

    # --- Duplicate Check (Centroid-based logic should inherently avoid this, but check anyway) ---
    n_unique_labels = df['label'].nunique()
    if n_unique_labels != len(df):
        logger.warning(f"Found {len(df)} results but only {n_unique_labels} unique labels. "
                       f"Duplicate labels detected. Applying drop_duplicates(subset=['label']).")
        initial_rows = len(df)
        df = df.drop_duplicates(subset=['label'], keep='first')
        logger.info(f"Removed {initial_rows - len(df)} duplicate label entries.")
    else:
        logger.info("Verified that all processed objects have unique labels.")

    # Log summary statistics
    if not df.empty:
        logger.info(f"Final DataFrame shape: {df.shape}")
        try:
            logger.info(f"Volume range: {df['volume'].min()} to {df['volume'].max()} voxels")
            logger.info(f"Mean object volume: {df['volume'].mean():.2f} voxels")
        except KeyError:
            logger.warning("Could not log volume stats (column missing).")
    else:
        logger.warning("DataFrame is empty after processing and potential duplicate removal.")

    return df


def parse_list_string(list_str, dtype=int):
    """Parses comma-separated string into a list of specified type."""
    print(list_str)
    print(type(list_str))
    if not list_str:
        return []
    return [dtype(item.strip()) for item in list_str.split(',')]

def parse_tuple_string(tuple_str, dtype=int):
    """Parses comma-separated string into a tuple of specified type."""
    return tuple(parse_list_string(tuple_str, dtype))

def main():
    parser = argparse.ArgumentParser(description="Distributed Feature Extraction for Microscopy Images")

    # Input/Output Arguments
    parser.add_argument("--input_n5", required=True, help="Path to the input N5 store (intensity image data)")
    parser.add_argument("--n5_path_pattern", default="ch{}/s0", help="Pattern for N5 dataset paths within the store (use {} for channel number)")
    parser.add_argument("--input_mask", required=True, help="Path to the input mask Zarr store or TIF file (label image data)")
    parser.add_argument("--output_csv", required=True, help="Path to the output CSV file for features")
    parser.add_argument("--channels", required=True, help="Comma-separated list of channel indices to process (e.g., '0,1,2')")

    # Processing Arguments
    parser.add_argument("--chunk_size", default="128,128,128", help="Core chunk size (Z,Y,X) for processing (comma-separated)")
    parser.add_argument("--overlap", default="32,32,32", help="Overlap size (Z,Y,X) for processing (comma-separated)")

    # Dask SGE Cluster Arguments
    parser.add_argument("--num_workers", type=int, default=16, help="Number of Dask workers (SGE jobs)")
    parser.add_argument("--cores_per_worker", type=int, default=1, help="Number of cores per worker")
    parser.add_argument("--mem_per_worker", default="60G", help="Memory per worker (e.g., '60G')")
    parser.add_argument("--processes", type=int, default=1, help="Number of Python processes per worker (usually 1)")
    parser.add_argument("--project", required=True, help="SGE project code")
    parser.add_argument("--queue", required=True, help="SGE queue name")
    parser.add_argument("--runtime", default="140000", help="Job runtime (SGE format or seconds)") # Keep as string for flexibility
    parser.add_argument("--resource_spec", default="mfree=60G", help="SGE resource specification (e.g., 'mfree=60G')")
    parser.add_argument("--log_dir", default=None, help="Directory for Dask worker logs (defaults to ./dask_worker_logs_TIMESTAMP)")
    parser.add_argument("--conda_env", default="dask-cellpose", help="Conda environment to activate on workers")

    # --- Argument Parsing ---
    # Check if running under Snakemake
    if 'snakemake' in globals():
        logger.info("Running under Snakemake, using snakemake object for parameters.")
        # Create a namespace object to mimic argparse result
        args = argparse.Namespace(
            input_n5=snakemake.input.n5,
            input_mask=snakemake.input.zarr, # Assuming mask is zarr input
            output_csv=snakemake.output.csv,
            n5_path_pattern=snakemake.params.get("n5_path_pattern", "ch{}/s0"),
            channels=",".join(map(str, snakemake.params.channels)), # Get channels list from params
            chunk_size=snakemake.params.chunk_size,
            overlap=snakemake.params.overlap,
            num_workers=snakemake.resources.num_workers,
            cores_per_worker=snakemake.resources.cores_per_worker,
            mem_per_worker=snakemake.resources.mem_per_worker,
            processes=snakemake.resources.processes,
            project=snakemake.resources.project,
            queue=snakemake.resources.queue,
            runtime=str(snakemake.resources.runtime), # Ensure string
            resource_spec=snakemake.resources.resource_spec,
            log_dir=snakemake.params.log_dir,
            conda_env=snakemake.conda_env_name if hasattr(snakemake, 'conda_env_name') else "dask-cellpose" # Get conda env name if available
        )
    else:
        logger.info("Not running under Snakemake, parsing command-line arguments.")
        args = parser.parse_args()

    # --- Parameter Processing ---
    try:
        core_chunk_shape = parse_tuple_string(args.chunk_size)
        overlap_shape = parse_tuple_string(args.overlap)
        channels_to_process = parse_list_string(args.channels)
        if len(core_chunk_shape) != 3 or len(overlap_shape) != 3:
            raise ValueError("Chunk size and overlap must have 3 dimensions (Z,Y,X).")
        if not channels_to_process:
             raise ValueError("Must specify at least one channel to process.")
    except Exception as e:
        logger.error(f"Error parsing parameters: {e}", exc_info=True)
        sys.exit(1)

    # Ensure output directory exists
    try:
        os.makedirs(os.path.dirname(args.output_csv), exist_ok=True)
    except OSError as e:
        logger.error(f"Could not create output directory {os.path.dirname(args.output_csv)}: {e}")
        sys.exit(1)

    # --- Dask Cluster Setup ---
    cluster = None
    client = None
    try:
        logger.info("Setting up Dask cluster...")
        cluster, client = setup_dask_sge_cluster(
            n_workers=args.num_workers,
            cores=args.cores_per_worker,
            processes=args.processes,
            memory=args.mem_per_worker,
            project=args.project,
            queue=args.queue,
            runtime=args.runtime,
            resource_spec=args.resource_spec,
            log_directory=args.log_dir,
            conda_env=args.conda_env
            # Add other relevant kwargs if needed, e.g., job_name='feat_extract'
        )
    except Exception as e:
        logger.error(f"Failed to set up Dask cluster: {e}", exc_info=True)
        sys.exit(1) # Exit if cluster setup fails

    # --- Main Processing Block ---
    try:
        start_time = time.time()
        logger.info("--- Starting Distributed Property Computation ---")
        logger.info(f"Input N5: {args.input_n5}")
        logger.info(f"Input Mask: {args.input_mask}")
        logger.info(f"Output CSV: {args.output_csv}")
        logger.info(f"Channels: {channels_to_process}")
        logger.info(f"N5 Pattern: {args.n5_path_pattern}")
        logger.info(f"Core chunk: {core_chunk_shape}, Overlap: {overlap_shape}")

        # --- Load N5 Channels Dynamically ---
        image_channels = []
        store = zarr.N5Store(args.input_n5) # Open store once
        for ch_idx in channels_to_process:
            n5_channel_path = args.n5_path_pattern.format(ch_idx)
            logger.info(f"Loading channel {ch_idx} from N5 path: {n5_channel_path}")
            try:
                 # Use the Zarr handle directly with da.from_zarr for efficiency
                 arr_handle = zarr.open_array(store=store, path=n5_channel_path, mode='r')
                 image_channels.append(da.from_zarr(arr_handle))
            except Exception as e:
                 logger.error(f"Failed to load channel {ch_idx} at path {n5_channel_path}: {e}")
                 raise # Re-raise error to stop processing if a channel fails

        if not image_channels:
            raise ValueError("No image channels were successfully loaded.")

        # Stack channels along the first axis (CZYX)
        image = da.stack(image_channels, axis=0)
        # Optional: Rechunk after stacking if needed, e.g., to align chunks across channels
        # image = image.rechunk({0: 'auto', 1: 128, 2: 128, 3: 128}) # Example rechunk
        logger.info(f"Loaded and stacked N5 image: Shape={image.shape}, Chunks={image.chunksize}, Dtype={image.dtype}")

        # --- Load Mask ---
        # Determine required chunk size for TIF loading if applicable
        mask_read_chunk_hint = tuple(c + 2*o for c, o in zip(core_chunk_shape, overlap_shape)) # Spatial ZYX hint

        if args.input_mask.lower().endswith(('.tif', '.tiff')):
             logger.info("Loading mask from TIF...")
             # Expecting 3D ZYX for mask
             mask = load_tif_as_dask_array(args.input_mask, processing_chunk_size=mask_read_chunk_hint, expected_dims='ZYX')
        else: # Assume Zarr/N5
             # For mask, assume it's a single dataset at the root or a specified path
             # Modify as needed if mask is also multichannel or nested in N5
             mask = load_n5_zarr_array(args.input_mask) # N5 mask would need n5_subpath
             if mask.ndim != 3:
                 raise ValueError(f"Loaded mask from {args.input_mask} has {mask.ndim} dimensions, expected 3 (ZYX).")

        logger.info(f"Loaded Mask: Shape={mask.shape}, Chunks={mask.chunksize}, Dtype={mask.dtype}")


        # --- Compute Properties ---
        properties_df = compute_distributed_properties(
            image, # Pass CZYX Dask array
            mask,  # Pass ZYX Dask array
            client=client, # Pass the active client
            chunk_size=core_chunk_shape,
            overlap=overlap_shape,
        )

        end_time = time.time()
        logger.info(f"--- Total computation time: {end_time - start_time:.2f} seconds ---")

        # --- Save Results ---
        if properties_df is not None and not properties_df.empty:
            logger.info(f"Saving results ({len(properties_df)} objects) to {args.output_csv}")
            properties_df.to_csv(args.output_csv, index=False)
            logger.info("Results saved.")
        elif properties_df is not None and properties_df.empty:
             logger.warning("Computation finished, but no objects were found or processed.")
        else:
            logger.error("Computation failed. No results generated.")

    except Exception as e:
        logger.error(f"An error occurred during the main processing: {e}", exc_info=True)
        # Ensure shutdown happens even if main logic fails

    finally:
        # --- Shutdown Dask Client and Cluster ---
        if cluster and client:
            shutdown_dask(cluster, client)
        else:
            logger.warning("Cluster/client not fully initialized, skipping shutdown.")


if __name__ == "__main__":
    main()