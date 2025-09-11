import numpy as np
import matplotlib.pyplot as plt
from skimage import filters, exposure
#from skimage.filters import gaussian
#from scipy.ndimage import uniform_filter
from tifffile import imread
import zarr
from time import time
from utils.dask_utils import setup_dask_sge_cluster, shutdown_dask
from dask.base import compute
import dask.array as da
import logging
from numcodecs import Blosc
import sys
import argparse
from dask.base import annotate
import gc
import dask
import cupy as cp
from dask.distributed import print as dask_print


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

def Interpl_8x(metric_array_8x, shape):
    """
    Interpolate to the shape of specified resolution data.

    Args:
    - metric_array_8x (np.ndarray): The metric array of 8x downsampled data.

    Returns:
    - metric (np.ndarray): The interpolated metric array.
    """

    img_length = shape[0]
    n = len(metric_array_8x)
    x = np.linspace(1,n,n)
    xvals = np.linspace(1,n,img_length)
    metric = np.interp(xvals, x, metric_array_8x)

    return metric

def calculate_rescale_lim(img_8x, shape):
    """
    Calculate the p2 and p98, min, and mean for the 8x downsampled 3D image.

    Args:
    - img_8x (np.ndarray): The 8x downsampled volume.
    - shape (tuple) : shape of selected res volume.

    Returns:
    - p2 (np.ndarray): Array of 2% min for the highest resolution volume interpolated from 8x downsampled volume.
    - p98 (np.ndarray): Array of 98% max for the highest resolution volume interpolated from 8x downsampled volume.
    - global_max (float): The max intensity for the 3D volume.
    """
    # img_8x = exposure.adjust_gamma(img_8x, 0.7)
    
    p2, p98 = np.percentile(img_8x,
                            (2, 99.5), 
                            axis = (1,2)
                            )
    p2[-1] = p2[-2]
    p98[-1] = p98[-2]
    global_max = np.max(p98)*0.98

    p2 = Interpl_8x(p2, shape)
    p98 = Interpl_8x(p98, shape)

    return p2, p98, global_max

def contrast_fix(img, p2, p98, global_max, block_info=None):
    """
    Rescale the p2 and p98 in the 2D image to the out_range.

    Args:
    - img (np.ndarray): The 2D image for the layer of interest.
    - p2 (float):
    - p98 (float):
    - global_max (float):
    - i (int): Index for current layer.

    Returns:
    - img_rescale (np.ndarray): The rescaled 2D image for that layer.
    """

    block_index = block_info[0]['array-location']
    # img = exposure.adjust_gamma(img, 0.7)
    img_rescale = exposure.rescale_intensity(img, 
                                            in_range=(p2[block_index[0][0]], p98[block_index[0][0]]*1.75), 
                                            out_range = (0, global_max)
                                            )

    return np.asarray(img_rescale, dtype=np.uint16)

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
        dask_arr_8x = dask_arr[::8, ::8, ::8]  # Downsample by 8x
        logger.info(f"  Channel {ch_idx} loaded: Shape={dask_arr.shape}, Chunks={dask_arr.chunksize}, Dtype={dask_arr.dtype}")
        logger.info(f"Annotating with resources: {parse_dict_string(args.dask_resources) if args.dask_resources else {}}")
        p2, p98, global_max = calculate_rescale_lim(dask_arr_8x, dask_arr.shape)
        corrected_img = da.map_blocks(
            contrast_fix, 
            dask_arr, 
            p2, p98, global_max,
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