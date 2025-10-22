import numpy as np
import dask.array as da
import sys
import argparse
import zarr
import logging
import dask
import cloudpickle
import platform
import distributed
import msgpack
import skimage
from numcodecs import Blosc
from utils.dask_utils import setup_dask_sge_cluster, shutdown_dask



logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
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


def main():

    parser = argparse.ArgumentParser(description="Rechunk to Planes for Microscopy Images")

    # Input/Output Arguments
    parser.add_argument("--input_n5", required=True, help="Path to the input N5 store (intensity image data)")

    parser.add_argument("--output_zarr", required=True, help="Path to the output Zarr store")

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
    parser.add_argument("--conda_env", default="otls-pipeline-cp3", help="Conda environment to activate on workers")
    parser.add_argument("--channels", default="0,1,2", help="Comma-separated list of channels to process (e.g., '0,1,2')")
    parser.add_argument("--dashboard_port", default=":8788", help="Dask dashboard port")
    # --- Argument Parsing ---
    # Check if running under Snakemake
    if 'snakemake' in globals():
        logger.info("Running under Snakemake, using snakemake object for parameters.")
        # Create a namespace object to mimic argparse result
        args = argparse.Namespace(
            input_n5=snakemake.input.n5,
            output_zarr=snakemake.output.zarr,
            num_workers=snakemake.resources.n_workers,
            cores_per_worker=snakemake.resources.cores,
            mem_per_worker=snakemake.resources.worker_memory,
            processes=snakemake.resources.processes,
            project=snakemake.resources.project,
            queue=snakemake.resources.queue,
            runtime=str(snakemake.resources.runtime), # Ensure string
            resource_spec=snakemake.resources.resource_spec,
            log_dir=snakemake.params.log_dir,
            conda_env=snakemake.conda_env_name if hasattr(snakemake, 'conda_env_name') else "otls-pipeline-cp3", # Get conda env name if available
            dashboard_port=snakemake.resources.dashboard_port,
            channels=",".join(map(str, snakemake.params.channels)),
            worker_timeout=snakemake.resources.get("worker_timeout", 600)
        )
    else:
        logger.info("Not running under Snakemake, parsing command-line arguments.")
        args = parser.parse_args()
        # Backwards-compat: map input_zarr to input_n5 for internal use
        if not hasattr(args, "input_n5"):
            args.input_n5 = args.input_zarr


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
            worker_timeout=args.worker_timeout
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
        logger.info(f"Output Zarr: {args.output_zarr}")
        channels_to_process = parse_list_string(args.channels)
        compressor = Blosc(cname='zstd', clevel=3, shuffle=Blosc.SHUFFLE, blocksize=0)
        store = zarr.N5Store(args.input_n5)
        n_levels = 4  # Number of downsampled levels
        downsample = 2  # Downsampling factor for each level

        for ch_idx in channels_to_process:
            n5_channel_path = 'ch{}/s0'.format(ch_idx)
            logger.info(f"Loading channel {ch_idx} from N5 path: {args.input_n5}/{n5_channel_path}")
            try:   
                arr_handle = zarr.open_array(store=store, path=n5_channel_path, mode='r')
                dask_arr = da.from_zarr(arr_handle, chunks=arr_handle.chunks)
                logger.info(f"Channel {ch_idx} loaded: Shape={dask_arr.shape}, Chunks={dask_arr.chunksize}, Dtype={dask_arr.dtype}")
            except Exception as e:
                    logger.error(f"Failed to load channel {ch_idx} at path {n5_channel_path}: {e}", exc_info=True)
                    raise # Re-raise error to stop processing if a channel fails

            # --- Rechunking ---
            logger.info("Rechunking to planes...")
            rechunked_array = dask_arr.rechunk({0: 1, 1: -1, 2: -1})
            logger.info(f"Rechunked array: Shape={rechunked_array.shape}, Chunks={rechunked_array.chunksize}, Dtype={rechunked_array.dtype}")

            # --- Saving ---
            logger.info(f"Saving rechunked array to: {args.output_zarr}")
            da.to_zarr(
                rechunked_array,
                url=args.output_zarr,
                component=f"ch{ch_idx}/s0",
                compressor=compressor,
                overwrite=True
            )

            for level in range(1, n_levels):
                prev = da.from_zarr(args.output_zarr, component=f"ch{ch_idx}/s{level-1}")
                coarse = da.coarsen(np.mean, prev, {0: downsample, 1: downsample, 2: downsample}, trim_excess=True).astype(np.uint16)
                da.to_zarr(coarse, url=args.output_zarr, component=f"ch{ch_idx}/s{level}", compressor=compressor, overwrite=True)

            logger.info(f"Rechunked array saved to: {args.output_zarr} + ch{ch_idx}/s0")

        logger.info(f"Finished processing all channels.")
    except Exception as e:
        logger.error(f"Error in main processing block: {e}", exc_info=True)
        sys.exit(1)

if __name__ == "__main__":
    main()