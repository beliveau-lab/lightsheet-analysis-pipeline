import numpy as np
import matplotlib.pyplot as plt
from skimage import filters, exposure
#from skimage.filters import gaussian
#from scipy.ndimage import uniform_filter
from tifffile import imread
import zarr
from time import time
from utils.dask_utils import setup_dask_sge_cluster, shutdown_dask
from dask import delayed, compute
import dask.array as da
import logging
from numcodecs import Blosc
import sys
import argparse
from dask import annotate
import gc
import dask
import cupy as cp
from cucim.skimage import filters as cucim_filters
from cucim.skimage import exposure as cucim_exposure
from cupyx.scipy.ndimage import uniform_filter as cupy_uniform_filter
from cucim.skimage.filters import gaussian as cucim_gaussian

# --- Environment Logging  ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__) 


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


# --- GPU-ACCELERATED STRIPE FIX FUNCTION (Copied from the main script) ---
def stripe_fix_gpu(
    img,
    clip_lowhi=(95, 99),
    min_mask_ratio=0.005,
    tissue_blur_kernel=None,
    vertical_blur_sigma=None,
    stripe_smooth_sigma=20
):
    # Ensure img is a cupy array.
    img = cp.asarray(img)
    img = cp.squeeze(img)

    if tissue_blur_kernel is None:
        tissue_blur_kernel = img.shape[0] / 100
    if vertical_blur_sigma is None:
        vertical_blur_sigma = (img.shape[0] / 100, 0)

    # Step 1: Create tissue mask using GPU operations
    low_clip = cp.percentile(img, clip_lowhi[0]) / 5
    high_clip = cp.percentile(img, clip_lowhi[1])
    img_clip = cp.clip(img, low_clip, high_clip) - low_clip
    
    blurred_img = cupy_uniform_filter(img_clip, tissue_blur_kernel)
    blurred_img[blurred_img < 0] = 0

    threshold_value = cucim_filters.threshold_otsu(blurred_img)
    mask = blurred_img > threshold_value

    # Step 2: Stripe profile estimation
    masked_blurimg = cucim_gaussian(img_clip, sigma=vertical_blur_sigma, preserve_range=True) * mask
    masked_blurimg_lineprof = cp.sum(masked_blurimg, axis=0)
    min_mask_pixels = img.shape[0] * min_mask_ratio
    mask_lineprof = cp.sum(mask, axis=0)
    col_masked_lineprof = cp.where(
        mask_lineprof > min_mask_pixels,
        masked_blurimg_lineprof / mask_lineprof,
        0
    )
    norm_line_prof = cucim_exposure.rescale_intensity(col_masked_lineprof, out_range=(0, 1))
    stripe = norm_line_prof * mask
    stripe[stripe == 0] = 1
    stripe_profile = cucim_gaussian(stripe, sigma=stripe_smooth_sigma, preserve_range=True)

    img_nobg = np.clip(img, low_clip, img.max()) - low_clip  ## modified
    corrected = img_nobg / stripe_profile

    del img, blurred_img, threshold_value, mask, masked_blurimg, masked_blurimg_lineprof, mask_lineprof, col_masked_lineprof, norm_line_prof, stripe, stripe_profile
    gc.collect()
    return cp.expand_dims(corrected, axis=0).get()

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

def parse_tuple_string(tuple_str, dtype=int):
    """Parses comma-separated string into a tuple of specified type.

    Parameters
    ----------
    tuple_str : str
        A comma-separated string of integers
    dtype : type
        The type to convert the string elements to

    Returns
    -------
    tuple : tuple
        A tuple of the specified type
    """
    if not tuple_str:
        return ()
    return tuple(dtype(item.strip()) for item in tuple_str.split(','))

def parse_dict_string(dict_str):
    """
    Parses a string like 'key1:val1,key2=val2' into a dictionary.
    Handles strings that might be wrapped in quotes from Snakemake.
    Values are attempted to be converted to integers, otherwise kept as strings.
    """
    if not dict_str:
        return {}
    
    # Strip potential outer quotes (single or double) from Snakemake
    dict_str = dict_str.strip("'\"")

    result_dict = {}
    import re
    
    for item in dict_str.split(','):
        item = item.strip()
        if not item:
            continue
            
        # Split by the first '=' or ':'
        parts = re.split(r'[:=]', item, maxsplit=1)
        
        if len(parts) == 2:
            key, value = parts
            key = key.strip()
            value = value.strip().strip("'\"") # Strip quotes from value
            # Attempt to convert value to integer, otherwise keep as string
            try:
                result_dict[key] = int(value)
            except ValueError:
                result_dict[key] = value
        else:
            logger.warning(f"Could not parse item '{item}' into key:value pair. Skipping.")
            
    return result_dict


def main():
    # --- Argument Parsing ---
    # Check if running under Snakemake
    if 'snakemake' in globals():
        logger.info("Running under Snakemake, using snakemake object for parameters.")
        # Create a namespace object to mimic argparse result
        args = argparse.Namespace(
            input_zarr=snakemake.input.zarr,
            channels=",".join(map(str, snakemake.params.channels)), # Get channels list from params

            # Worker configs
            num_workers=snakemake.resources.n_workers,
            cores_per_worker=snakemake.resources.cores,
            mem_per_worker=snakemake.resources.worker_memory,
            processes=snakemake.resources.processes,
            project=snakemake.resources.project,
            queue=snakemake.resources.queue,
            resource_spec=snakemake.resources.resource_spec,

            runtime=str(snakemake.resources.runtime), # Ensure string
            dashboard_port=snakemake.resources.dashboard_port,
            log_dir=snakemake.params.log_dir,
            conda_env=snakemake.conda_env_name if hasattr(snakemake, 'conda_env_name') else "otls-pipeline", # Get conda env name if available
            output_zarr=snakemake.output.zarr,
            dask_resources=snakemake.resources.get("dask_resources", None)
        )
    else:
        logger.info("Not running under Snakemake, parsing command-line arguments.")
        args = parser.parse_args()


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
            conda_env=args.conda_env,
            dashboard_port=args.dashboard_port,
            resources=True
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
    channels_to_process = parse_list_string(args.channels)
    compressor = Blosc(cname="zstd", clevel=3, shuffle=Blosc.BITSHUFFLE)

    for ch_idx in channels_to_process:
        n5_channel_path = 'ch{}/s0'.format(ch_idx)
        logger.info(f"Loading channel {ch_idx} from zarr path: {args.input_zarr}/{n5_channel_path}")
        dask_arr = da.from_zarr(args.input_zarr, component=f"ch{ch_idx}/s0")

        logger.info(f"  Channel {ch_idx} loaded: Shape={dask_arr.shape}, Chunks={dask_arr.chunksize}, Dtype={dask_arr.dtype}")
        with dask.config.set(resources={"GPU": 1}):
            logger.info(f"Annotating with resources: {parse_dict_string(args.dask_resources) if args.dask_resources else {}}")
            corrected_img = da.map_blocks(
                stripe_fix_gpu, 
                dask_arr, 
                dtype=np.uint16
            )

            da.to_zarr(
                corrected_img,
                url=args.output_zarr,
                component=f"ch{ch_idx}/s0",        # top-level array
                compressor=compressor,
                overwrite=True
            )

        logger.info(f"Channel {ch_idx} processed and saved to: {args.output_zarr}")
                    

if __name__ == "__main__":
    main()