import numpy as np
import dask.array as da
import warnings
import sys
import argparse
import zarr
import pandas as pd
import time
import logging
import os
# import dask.distributed as dd # Removed unused import
import dask
import cloudpickle
import platform
import distributed
import msgpack
import skimage
from aicsshparam import shparam

# Suppress aicsshparam warnings globally for all workers
warnings.filterwarnings("ignore", message="Mesh centroid seems to fall outside the object.*", module="aicsshparam")
warnings.filterwarnings("ignore", message=".*spherical harmonics.*", module="aicsshparam")

from skimage.measure import regionprops, regionprops_table
import psutil
from dask import delayed, compute
from utils.dask_utils import setup_dask_sge_cluster, shutdown_dask
import sparse
import dask.dataframe as dd


sys.path.append(os.path.dirname(__file__))
import align_3d as align

# --- Script-specific constants ---
DEFAULT_LMAX = 16 
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- Helper Functions ---
def to_sparse(block):
    # block is a numpy ndarray here; convert to sparse.COO
    return sparse.COO.from_numpy(block)

def _array_chunk_location(block_id, chunks):
    """Pixel coordinate of top left corner of the array chunk."""
    array_location = []
    for idx, chunk in zip(block_id, chunks):
        array_location.append(sum(chunk[:idx]))
    return tuple(array_location)

def _find_bounding_boxes(x, array_location):
    """An alternative to scipy.ndimage.find_objects, supporting sparse.COO."""
    # 1) Extract unique non‑zero labels
    if isinstance(x, sparse.COO):
        # sparse.COO.data holds the non-zeros, .coords is shape (ndim, nnz)
        data   = x.data
        coords = x.coords
        # only consider nonzero labels
        mask   = data != 0
        unique_vals = np.unique(data[mask])
    else:
        unique_vals = np.unique(x)
        unique_vals = unique_vals[unique_vals != 0]

    # 2) For each label, find its min/max in each dim
    result = {}
    for val in unique_vals:
        if isinstance(x, sparse.COO):
            # locations of this label in sparse coords
            sel = np.nonzero(data == val)[0]
            # for each dim i, coords[i][sel] are the positions
            positions = [coords[i, sel] for i in range(x.ndim)]
        else:
            # numpy fallback
            positions = np.where(x == val)

        # build nd‑slice shifted by array_location
        slices = tuple(
            slice(
                int(positions[i].min()) + array_location[i],
                int(positions[i].max()) + 1 + array_location[i]
            )
            for i in range(x.ndim)
        )
        result[int(val)] = slices

    # 3) turn into a DataFrame with one column per dimension
    cols = list(range(x.ndim))
    return pd.DataFrame.from_dict(result, orient='index', columns=cols)


def _combine_slices(slices):
    """
    Return the union of all slice objects in `slices`.

    Ignores any elements that are not real slice objects.
    Raises ValueError if, after filtering, no slices remain.
    """
    slices = [eval(sl) for sl in slices]
    # keep only true slice instances
    good = [sl for sl in slices if isinstance(sl, slice)]
    
    if not good:
        raise ValueError(f"_combine_slices got no slice objects, got: {slices!r}")
    if len(good) == 1:
        return good[0]
    # extract starts & stops
    starts = [sl.start for sl in good]
    stops  = [sl.stop  for sl in good]
    return slice(min(starts), max(stops))


def _merge_bounding_boxes(x: pd.DataFrame, ndim: int) -> pd.Series:
    """
    Merge bounding‐box slices for one label across multiple chunks.
    
    Parameters
    ----------
    x : DataFrame
        Rows all belong to the same label.  Must be indexed by that label,
        and have columns 0..ndim-1 containing slice objects.
    ndim : int
        Number of dimensions.
    
    Returns
    -------
    Series
        One row per dimension (0..ndim-1), containing the unioned slice.
        Series.name is set to the integer label.
    """
    # The label is the index value (all rows share the same label).
    label = x.index[0]
    
    data = {}
    for i in range(ndim):
        # collect all slices in column i
        sls = list(x[i])
        data[i] = _combine_slices(sls)
    
    return pd.Series(data=data, index=list(range(ndim)), name=label)


def find_objects(label_image):
    """
    For each chunk of `label_image`, call our chunk‑level _find_bounding_boxes
    (delayed → pandas.DataFrame), then concatenate them all into one Dask DataFrame
    and group by label to merge bounding boxes across chunks.
    """
    # 1) build one delayed pandas.DataFrame per chunk
    delayed_dfs = []
    for block_id, slc in zip(
        np.ndindex(*label_image.numblocks),
        da.core.slices_from_chunks(label_image.chunks)
    ):
        chunk = label_image[slc]
        loc   = _array_chunk_location(block_id, label_image.chunks)
        delayed_dfs.append(
            delayed(_find_bounding_boxes)(chunk, loc)
        )

    # 2) turn that list of delayed DataFrames into a single Dask DataFrame
    #    each “partition” is one chunk’s DataFrame
    meta = dd.utils.make_meta({i: object for i in range(label_image.ndim)})
    ddf  = dd.from_delayed(delayed_dfs, meta=meta)
    # 3) group by the integer label (the index), and merge slices
    #    _merge_bounding_boxes takes a chunk of rows (all from one label)
    #    and returns a single Series of slices per dimension
    result = (
        ddf
        .reset_index()              # bring the label (index) into a column
        .rename(columns={"index": "label"})
        .groupby("label")
        .apply(
            lambda pdf: _merge_bounding_boxes(pdf, label_image.ndim),
            meta=meta
        )
    )

    return result

# --- Environment Logging  ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__) 


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
    logger.info("--- End Environment Info ---")


log_environment()

# --- Simplified N5/Zarr loading ---
def load_n5_zarr_array(path, n5_subpath=None, chunks=None):
    """Loads N5 or Zarr, handling potential subpath for N5.
    
    Args:
        path: str
            The path to the N5 or Zarr container
        n5_subpath: str
            The subpath to the underlying N5 array
        chunks: tuple
            The chunks to load the array into (leave as None for no rechunking)
    Returns:
        da.Array
            The dask array
    """
    
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

def process_object(obj, mask_da, image_da, optimal_lmax=4):
    """
    Enhanced process_object that includes spherical harmonics computation
    Args:
        obj: pandas.Series
            A row from the DataFrame containing the object bounding box coordinates
        mask_da: dask.array.Array
            The segmentation mask array (ZYX)
        image_da: dask.array.Array
            The input image array (ZYXC)
        optimal_lmax: int
            The optimized lmax value for spherical harmonics computation
    Returns:
        final_dict: dict
            A dictionary containing the object properties and spherical harmonics coefficients
    """
    import sys, os
    import warnings
    sys.path.append(os.path.dirname(__file__))   # path where feature_extraction.py lives
    from aicsshparam import shparam
    import align_3d as align
    
    # Suppress aicsshparam warnings that occur for every object
    warnings.filterwarnings("ignore", message="Mesh centroid seems to fall outside the object.*", module="aicsshparam")

    obj_id = int(obj.name)
    slice_z = obj[0]
    slice_y = obj[1] 
    slice_x = obj[2]

    # Compute centroid of object (in global coordinates)
    centroid_z = slice_z.start + (slice_z.stop - slice_z.start) // 2
    centroid_y = slice_y.start + (slice_y.stop - slice_y.start) // 2
    centroid_x = slice_x.start + (slice_x.stop - slice_x.start) // 2

    try:
        # Get only the pixels belonging to the object of interest
        label_slice = np.where(mask_da == obj_id, mask_da, 0)
        # Basic regionprops
        props = regionprops_table(label_slice, image_da, properties=["label", 
                                                                     "area", 
                                                                     "mean_intensity"])
        area = props['area'][0]
        if area < 4000:  # Increase threshold and fix indexing
            logger.info(f"Object {obj_id} too small (area={area}). Skipping...")
            return None
        else:
            logger.info(f"Object {obj_id} is large enough (area={area}). Processing...")

        # Add global centroid coordinates
        props['centroid_z'] = centroid_z
        props['centroid_y'] = centroid_y
        props['centroid_x'] = centroid_x
        # Align object
        aligned_slice, props = align.align_object(label_slice, props) # align object also adds features to df_props
        
        # Check if aligned object has any foreground voxels
        if np.sum(aligned_slice) == 0:
            logger.info(f"Object {obj_id} has no foreground voxels after alignment. Skipping spherical harmonics...")
            return None
            
        # Spherical harmonics computation
        (coeffs, _), _ = shparam.get_shcoeffs(
            aligned_slice, 
            lmax=optimal_lmax,
            alignment_2d=False)
        
        coeffs.update({'label': obj_id})
        final_dict = props | coeffs
        logger.info(f"Processed object {obj_id}")
        return final_dict
    except Exception as e:
        logger.error(f"Error processing object {obj_id}: {e}", exc_info=True)
        return None


def parallel_processing(objects_df, mask_da, image_da, optimal_lmax, batch_size=10000):  # Reduced from 10000
    """Process single objects in parallel to control memory usage.
    
    Args:
        objects_df: pandas.DataFrame
            A DataFrame containing the object bounding box coordinates
        mask_da: dask.array.Array
            The segmentation mask array (ZYX)
        image_da: dask.array.Array
            The input image array (ZYXC)
        optimal_lmax: int
            The optimized lmax value for spherical harmonics computation
        batch_size: int
            The number of objects to process in each batch
    Returns:
        results: list
            A list of DataFrames containing the object properties.
    """

    results = []
    batch = []
    batch_idx = 0
    total_num_batches = len(objects_df) // batch_size
    # Process in batches to control memory
    logger.info(f"Number of objects before batching: {len(objects_df)}")
    for i in range(1, len(objects_df)):
        # Get the object bounding box coordinates
        obj = objects_df.iloc[i]
        slice_z = obj[0]
        slice_y = obj[1]
        slice_x = obj[2]

        # Append delayed tasks to the batch
        batch.append(
            delayed(process_object)(
                obj,
                mask_da[slice_z, slice_y, slice_x],
                image_da[slice_z, slice_y, slice_x],
                optimal_lmax
            )
        )
        ## TODO: Prepare batches while waiting for the previous batch to finish?
        ## For loop is inefficient

        # Process in batches
        if len(batch) >= batch_size:
            batch_idx += 1
            logger.info(f"Processing batch {batch_idx} of {total_num_batches}")
            results.extend(compute(*batch, sync=True))
            batch = []
    
    # Process remaining tasks
    if batch:
        results.extend(compute(*batch, sync=True))
    return [r for r in results if r is not None]

def parse_list_string(list_str, dtype=int):
    """Parses comma-separated string into a list of specified type.
    
    Args:
        list_str: str
            A comma-separated string of values
        dtype: type
            The type to convert the values to
    Returns:
        list:
            A list of values of the specified type
    """
    if not list_str:
        return []
    return [dtype(item.strip()) for item in list_str.split(',')]

# Removed load_optimal_lmax; lmax is now a constant defined in main()

def main():

    parser = argparse.ArgumentParser(description="Distributed Feature Extraction for Microscopy Images")

    # Input/Output Arguments
    parser.add_argument("--input_n5", required=True, help="Path to the input N5 store (intensity image data)")
    parser.add_argument("--n5_path_pattern", default="ch{}/s0", help="Pattern for N5 dataset paths within the store (use {} for channel number)")
    parser.add_argument("--input_mask", required=True, help="Path to the input mask Zarr store or TIF file (label image data)")
    parser.add_argument("--output_csv", required=True, help="Path to the output CSV file for features")
    parser.add_argument("--channels", required=True, help="Comma-separated list of channel indices to process (e.g., '0,1,2')")

    # Processing Arguments
    parser.add_argument("--batch_size", default=10000, help="Batch size for processing")

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
    parser.add_argument("--conda_env", default="otls-pipeline", help="Conda environment to activate on workers")

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
            batch_size=snakemake.params.batch_size,
            num_workers=snakemake.resources.num_workers,
            cores_per_worker=snakemake.resources.cores_per_worker,
            mem_per_worker=snakemake.resources.mem_per_worker,
            processes=snakemake.resources.processes,
            project=snakemake.resources.project,
            queue=snakemake.resources.queue,
            runtime=str(snakemake.resources.runtime), # Ensure string
            resource_spec=snakemake.resources.resource_spec,
            log_dir=snakemake.params.log_dir,
            conda_env=snakemake.conda_env_name if hasattr(snakemake, 'conda_env_name') else "otls-pipeline" # Get conda env name if available
        )
    else:
        logger.info("Not running under Snakemake, parsing command-line arguments.")
        args = parser.parse_args()

    channels_to_process = parse_list_string(args.channels)

    # --- Lmax Parameter ---
    logger.info(f"Using lmax: {DEFAULT_LMAX}")

    # --- Dask Cluster Setup ---
    cluster = None
    client = None
    try:
        logger.info("Setting up Dask cluster...")
        # Make log_dir optional in setup function call if it handles None
        cluster, client = setup_dask_sge_cluster(
            n_workers=args.num_workers,
            cores=args.cores_per_worker,
            processes=args.processes,
            memory=args.mem_per_worker,
            project=args.project,
            queue=args.queue,
            runtime=args.runtime,
            resource_spec=args.resource_spec,
            log_directory=args.log_dir, # Pass None if not specified
            conda_env=args.conda_env
            # job_name='feat_extract' # Consider adding a job name
        )
        logger.info(f"Dask dashboard link: {client.dashboard_link}") # Log dashboard link!
        
    except Exception as e:
        logger.error(f"Failed to set up Dask cluster: {e}", exc_info=True)
        if cluster: # Attempt shutdown even if client setup failed after cluster object created
             try:
                 shutdown_dask(cluster, client)
             except Exception as e_shutdown:
                 logger.error(f"Error during cluster shutdown after setup failure: {e_shutdown}")
        sys.exit(1) # Exit if cluster setup fails

    # --- Main Processing Block ---
    try:
        logger.info("--- Starting Distributed Property Computation ---")
        logger.info(f"Input N5: {args.input_n5}")
        logger.info(f"Input Mask: {args.input_mask}")
        logger.info(f"Output CSV: {args.output_csv}")
        logger.info(f"Channels: {channels_to_process}")
        logger.info(f"N5 Pattern: {args.n5_path_pattern}")

        image_array_list = []
        store = zarr.N5Store(args.input_n5) # Open store once

        for ch_idx in channels_to_process:
            n5_channel_path = 'ch{}/s0'.format(ch_idx)
            logger.info(f"Loading channel {ch_idx} from N5 path: {args.input_n5}/{n5_channel_path}")
            try:
                    arr_handle = zarr.open_array(store=store, path=n5_channel_path, mode='r')
                    dask_arr = da.from_zarr(arr_handle)
                    logger.info(f"  Channel {ch_idx} loaded: Shape={dask_arr.shape}, Chunks={dask_arr.chunksize}, Dtype={dask_arr.dtype}")
                    image_array_list.append(dask_arr)
            except Exception as e:
                    logger.error(f"Failed to load channel {ch_idx} at path {n5_channel_path}: {e}", exc_info=True)
                    raise # Re-raise error to stop processing if a channel fails
                    
        if not image_array_list:
            raise ValueError("No image channels were successfully loaded.")

        # --- Load Mask and Image Arrays ---

        image_array = da.stack(image_array_list, axis=3)
        mask_array = load_n5_zarr_array(args.input_mask)

        logger.info(f"Loaded Mask (ZYX assumed): Shape={mask_array.shape}, Dtype={mask_array.dtype}")

        chunk_shape = tuple(c[0] for c in mask_array.chunks)  
        #    (this grabs the first size of each chunk-axis; e.g. (100,100))
        meta_block = sparse.COO.from_numpy(
            np.zeros(chunk_shape, dtype=mask_array.dtype)
        )

        mask_sparse = mask_array.map_blocks(to_sparse, 
                                        dtype=mask_array.dtype,
                                        meta=meta_block,
                                        chunks=mask_array.chunks
                                        )


        # --- First Pass: Get Object Bounding Boxes ---
        df_bboxes = find_objects(mask_sparse).compute()
        df_bboxes = pd.DataFrame(df_bboxes)
        df_bboxes.to_csv(args.output_csv.replace(".csv", "_bboxes.csv"), index=False)
        # df_bboxes = df_bboxes.sample(n=100)

        # --- Second Pass: Extract Features ---
        data_frames = parallel_processing(
            df_bboxes, 
            mask_array, 
            image_array, 
            optimal_lmax=DEFAULT_LMAX, 
            batch_size=args.batch_size
        )
        df_properties = pd.DataFrame(data_frames)
        df_properties = df_properties.applymap(lambda x: x.item() if hasattr(x, "item") else x)
        df_properties.to_csv(args.output_csv, index=False)
        logger.info(f"Finished processing all objects and csv saved.")
    except Exception as e:
        logger.error(f"Error in main processing block: {e}", exc_info=True)
        sys.exit(1)

if __name__ == "__main__":
    main()