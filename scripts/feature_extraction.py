import numpy as np
import dask.array as da
import warnings
import sys
import argparse
import zarr
import pandas as pd
import logging
import os
import dask
import cloudpickle
import platform
import distributed
import msgpack
import skimage
from aicsshparam import shparam
from dask.distributed import as_completed
from scipy import ndimage as ndi

# Suppress aicsshparam warnings globally for all workers
warnings.filterwarnings("ignore", message="Mesh centroid seems to fall outside the object.*", module="aicsshparam")
warnings.filterwarnings("ignore", message=".*spherical harmonics.*", module="aicsshparam")

from skimage.measure import regionprops, regionprops_table
import psutil
from dask import delayed, compute
from utils.dask_utils import setup_dask_sge_cluster, shutdown_dask


from cellpose.models import CellposeModel

sys.path.append(os.path.dirname(__file__))
import align_3d as align
from aicsshparam import shparam

# --- Script-specific constants ---
DEFAULT_LMAX = 16 
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)



# --- Helper Functions ---

def _array_chunk_location(block_id, chunks):
    """Pixel coordinate of top left corner of the array chunk."""
    array_location = []
    for idx, chunk in zip(block_id, chunks):
        array_location.append(sum(chunk[:idx]))
    return tuple(array_location)



# --- Numeric-only bbox computation per chunk (fast, safe) ---
def _chunk_bboxes_numeric(block, array_location):
    """
    Compute bounding boxes for all labels present in a 3D block.

    Returns a pandas.DataFrame with columns:
    ['label', 'z0', 'z1', 'y0', 'y1', 'x0', 'x1'] in GLOBAL coordinates.
    """
    block = np.asarray(block)
    if block.ndim != 3:
        raise ValueError(f"_chunk_bboxes_numeric expects 3D block, got shape={block.shape}")

    # Background mask
    nonzero_mask = block != 0
    if not np.any(nonzero_mask):
        return pd.DataFrame(columns=["label", "z0", "z1", "y0", "y1", "x0", "x1"])  # empty

    # Compact the labels in this block to 1..K using return_inverse to avoid per-label scans
    nz_vals = block[nonzero_mask]
    unique_labels, inverse = np.unique(nz_vals, return_inverse=True)
    compact = np.zeros(block.shape, dtype=np.int32)
    compact[nonzero_mask] = inverse + 1  # background stays 0, labels are 1..K

    # Find object slices once using compact labeling
    obj_slices = ndi.find_objects(compact)
    if obj_slices is None:
        return pd.DataFrame(columns=["label", "z0", "z1", "y0", "y1", "x0", "x1"])  # empty

    rows = []
    z_off, y_off, x_off = array_location
    for idx, slc in enumerate(obj_slices, start=1):
        if slc is None:
            continue
        sz, sy, sx = slc
        rows.append({
            "label": int(unique_labels[idx - 1]),
            "z0": int(sz.start + z_off),
            "z1": int(sz.stop + z_off),
            "y0": int(sy.start + y_off),
            "y1": int(sy.stop + y_off),
            "x0": int(sx.start + x_off),
            "x1": int(sx.stop + x_off),
        })

    return pd.DataFrame(rows, columns=["label", "z0", "z1", "y0", "y1", "x0", "x1"])


def compute_bboxes_numeric(mask_da, client):
    """
    Compute global bounding boxes for all labels in mask_da using per-chunk extraction,
    then merge across chunks with pandas (min/max per label).
    """
    logger.info("Computing per-chunk numeric bounding boxes (no slice objects)...")
    # Build delayed DataFrames, one per chunk
    delayed_blocks = mask_da.to_delayed()
    delayed_dfs = []
    for block_id in np.ndindex(*mask_da.numblocks):
        loc = _array_chunk_location(block_id, mask_da.chunks)
        block_delayed = delayed_blocks[block_id]
        df_delayed = delayed(_chunk_bboxes_numeric)(block_delayed, loc)
        delayed_dfs.append(df_delayed)

    logger.info(f"Submitting {len(delayed_dfs)} bbox-chunk tasks...")
    futures = client.compute(delayed_dfs)
    dfs = []
    for fut, res in as_completed(futures, with_results=True):
        if res is not None and not (isinstance(res, pd.DataFrame) and res.empty):
            dfs.append(res)
        fut.release()

    if not dfs:
        raise RuntimeError("No bounding boxes found in mask array.")

    logger.info("Merging per-chunk bounding boxes across chunks...")
    df_all = pd.concat(dfs, ignore_index=True)
    # Union bboxes across chunks with min/max
    agg = {
        "z0": "min",
        "z1": "max",
        "y0": "min",
        "y1": "max",
        "x0": "min",
        "x1": "max",
    }
    df_merged = df_all.groupby("label", as_index=False).agg(agg)
    return df_merged


def add_chunk_indices(df_bboxes: pd.DataFrame, chunk_sizes: tuple) -> pd.DataFrame:
    """
    Add chunk index columns (cz, cy, cx) using bbox centroid and given chunk sizes.
    """
    cz_size, cy_size, cx_size = (int(chunk_sizes[0][0]), int(chunk_sizes[1][0]), int(chunk_sizes[2][0]))
    cz = ((df_bboxes["z0"] + df_bboxes["z1"]) // 2) // cz_size
    cy = ((df_bboxes["y0"] + df_bboxes["y1"]) // 2) // cy_size
    cx = ((df_bboxes["x0"] + df_bboxes["x1"]) // 2) // cx_size
    df_bboxes = df_bboxes.copy()
    df_bboxes["cz"] = cz.astype(np.int64)
    df_bboxes["cy"] = cy.astype(np.int64)
    df_bboxes["cx"] = cx.astype(np.int64)
    return df_bboxes


def _process_group(group_df: pd.DataFrame,
                   mask_path: str,
                   image_zarr_root: str,
                   channels: list,
                   lmax: int,
                   area_min: int,
                   group_key: tuple) -> pd.DataFrame:
    """
    Process a group of labels that share the same storage chunk indices.

    - Read a single super-ROI from mask and image for all labels in the group
    - Compute morphology features once per label using regionprops_table
    - Compute per-channel mean intensities via regionprops_table per channel
    - Compute spherical harmonics per label on aligned binary object mask

    Returns a pandas DataFrame with one row per label in the group.
    """

    import sys, os
    import warnings
    sys.path.append(os.path.dirname(__file__))   # path where feature_extraction.py lives
    from aicsshparam import shparam
    import align_3d as align
    
    if group_df.empty:
        return pd.DataFrame()

    labels = group_df["label"].astype(np.int64).tolist()
    sz = int(group_df["z0"].min()); ez = int(group_df["z1"].max())
    sy = int(group_df["y0"].min()); ey = int(group_df["y1"].max())
    sx = int(group_df["x0"].min()); ex = int(group_df["x1"].max())

    # Read ROI directly from Zarr to avoid nested Dask computes
    if mask_path.endswith('.n5'):
        store = zarr.N5Store(mask_path)
        mask_arr = zarr.open_array(store=store, path=None, mode='r')
    else:
        mask_arr = zarr.open(mask_path, mode='r')

    mask_roi = np.asarray(mask_arr[sz:ez, sy:ey, sx:ex])

    # Quick exit if no voxels
    if np.count_nonzero(mask_roi) == 0:
        return pd.DataFrame()

    # Morphology for all labels; regionprops_table with label image only
    morph = regionprops_table(mask_roi, properties=["label", "area", "centroid"])
    morph_df = pd.DataFrame(morph)
    if morph_df.empty:
        return pd.DataFrame()

    # Keep only our labels
    morph_df["label"] = morph_df["label"].astype(np.int64)
    morph_df = morph_df[morph_df["label"].isin(labels)].copy()
    if morph_df.empty:
        return pd.DataFrame()

    # Area threshold (filter tiny objects early)
    morph_df = morph_df[morph_df["area"] >= area_min]
    if morph_df.empty:
        return pd.DataFrame()

    # Convert centroid to global coordinates
    if "centroid-0" in morph_df.columns:
        morph_df.rename(columns={"centroid-0": "centroid_z",
                                 "centroid-1": "centroid_y",
                                 "centroid-2": "centroid_x"}, inplace=True)
    morph_df["centroid_z"] = morph_df["centroid_z"] + sz
    morph_df["centroid_y"] = morph_df["centroid_y"] + sy
    morph_df["centroid_x"] = morph_df["centroid_x"] + sx

    # Per-channel intensity means via regionprops_table join
    if channels:
        # Load intensity channels ROI
        ch_arrays = []
        for ch in channels:
            ch_path = os.path.join(image_zarr_root, f"ch{ch}", "s0")
            ch_arr = load_n5_zarr_array(image_zarr_root, n5_subpath=f"ch{ch}/s0")
            ch_arrays.append(np.asarray(ch_arr[sz:ez, sy:ey, sx:ex]))

        for idx, ch in enumerate(channels):
            ch_df = pd.DataFrame(
                regionprops_table(mask_roi, intensity_image=ch_arrays[idx], properties=["label", "mean_intensity"])  # type: ignore[arg-type]
            )
            ch_df["label"] = ch_df["label"].astype(np.int64)
            ch_df.rename(columns={"mean_intensity": f"mean_intensity_ch{ch}"}, inplace=True)
            morph_df = morph_df.merge(ch_df, on="label", how="left")

    # Build bbox map for faster lookup
    bbox_map = group_df.set_index("label")[['z0', 'z1', 'y0', 'y1', 'x0', 'x1']].to_dict('index')

    # Compute SH coefficients per object
    results = []
    for _, pr in morph_df.iterrows():
        label_id = int(pr["label"]) 
        bbox = bbox_map.get(label_id)
        if bbox is None:
            continue
        lz0 = int(bbox['z0'] - sz); lz1 = int(bbox['z1'] - sz)
        ly0 = int(bbox['y0'] - sy); ly1 = int(bbox['y1'] - sy)
        lx0 = int(bbox['x0'] - sx); lx1 = int(bbox['x1'] - sx)

        sub = (mask_roi[lz0:lz1, ly0:ly1, lx0:lx1] == label_id)
        if not np.any(sub):
            continue
        sub = np.ascontiguousarray(sub)

        props_dict = pr.to_dict()
        aligned_slice, props_dict = align.align_object(sub, props_dict)
        if np.sum(aligned_slice) == 0:
            continue

        try:
            (coeffs, _), _ = shparam.get_shcoeffs(aligned_slice, lmax=lmax, alignment_2d=False)
        except Exception as e:
            logger.error(f"SH computation failed for label {label_id}: {e}")
            continue

        coeffs.update({'label': label_id})
        final_dict = props_dict | coeffs
        results.append(final_dict)

    if not results:
        return pd.DataFrame()

    out_df = pd.DataFrame(results)
    cz, cy, cx = group_key
    out_df["cz"] = int(cz)
    out_df["cy"] = int(cy)
    out_df["cx"] = int(cx)
    return out_df

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
    parser.add_argument("--input_zarr", required=True, help="Path to the input Zarr store (intensity image data)")

    parser.add_argument("--input_mask", required=True, help="Path to the input mask Zarr store or TIF file (label image data)")
    parser.add_argument("--output_csv", required=True, help="Path to the output CSV file for features")
    parser.add_argument("--channels", required=True, help="Comma-separated list of channel indices to process (e.g., '0,1,2')")
    parser.add_argument("--generate_embeddings", required=True, help="Whether to generate embeddings (True/False)")
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


    # --- Argument Parsing ---
    # Check if running under Snakemake
    if 'snakemake' in globals():
        logger.info("Running under Snakemake, using snakemake object for parameters.")
        # Create a namespace object to mimic argparse result
        args = argparse.Namespace(
            input_zarr=snakemake.input.img_zarr,
            input_mask=snakemake.input.mask_zarr, # Assuming mask is zarr input
            output_csv=snakemake.output.csv,
            n5_path_pattern=snakemake.params.get("n5_path_pattern", "ch{}/s0"),
            channels=",".join(map(str, snakemake.params.channels)), # Get channels list from params
            generate_embeddings=snakemake.params.get("generate_embeddings", False), # Default to False if not set
            num_workers=snakemake.resources.num_workers,
            cores_per_worker=snakemake.resources.cores_per_worker,
            mem_per_worker=snakemake.resources.mem_per_worker,
            processes=snakemake.resources.processes,
            project=snakemake.resources.project,
            queue=snakemake.resources.queue,
            runtime=str(snakemake.resources.runtime), # Ensure string
            resource_spec=snakemake.resources.resource_spec,
            log_dir=snakemake.params.log_dir,
            conda_env=snakemake.conda_env_name if hasattr(snakemake, 'conda_env_name') else "otls-pipeline-cp3", # Get conda env name if available
            dashboard_port=snakemake.resources.dashboard_port
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
            conda_env=args.conda_env,
            dashboard_port=args.dashboard_port
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
        logger.info(f"Input Zarr (images root): {args.input_zarr}")
        logger.info(f"Input Mask: {args.input_mask}")
        logger.info(f"Output: {args.output_csv}")
        logger.info(f"Channels: {channels_to_process}")

        # --- Load Mask as Dask only to access chunks and parallelize bbox pass ---
        mask_array = load_n5_zarr_array(args.input_mask)
        logger.info(f"Loaded Mask (ZYX assumed): Shape={mask_array.shape}, Chunks={mask_array.chunks}, Dtype={mask_array.dtype}")

        # --- First Pass: Numeric BBoxes per chunk, merged globally ---
        df_bboxes = compute_bboxes_numeric(mask_array, client)
        # Save bboxes for debugging/inspection
        bboxes_csv = args.output_csv.replace(".csv", "_bboxes.csv")
        df_bboxes.to_csv(bboxes_csv, index=False)
        logger.info(f"Wrote bounding boxes to {bboxes_csv} (rows={len(df_bboxes)})")

        # --- Add storage chunk indices and group ---
        df_bboxes = add_chunk_indices(df_bboxes, mask_array.chunks)

        # Drop extremely tiny ROIs early by bbox volume heuristic to cut work
        voxel_est = (df_bboxes["z1"] - df_bboxes["z0"]) * (df_bboxes["y1"] - df_bboxes["y0"]) * (df_bboxes["x1"] - df_bboxes["x0"])
        df_bboxes = df_bboxes.loc[voxel_est >= 1, :]

        # Group by chunk indices
        grouped = df_bboxes.groupby(["cz", "cy", "cx"], sort=False)
        group_items = list(grouped)
        logger.info(f"Discovered {len(group_items)} chunk-groups with labels")

        # --- Output setup ---

        os.makedirs(os.path.dirname(args.output_csv), exist_ok=True)

        # --- Second Pass: Process groups in parallel ---
        futures = []
        for (cz, cy, cx), gdf in group_items:
            fut = client.submit(
                _process_group,
                gdf,
                args.input_mask,
                args.input_zarr,
                channels_to_process,
                DEFAULT_LMAX,
                4000,
                (int(cz), int(cy), int(cx))
            )
            futures.append(fut)

        header_written = False
        write_count = 0
        for fut, res in as_completed(futures, with_results=True):
            try:
                if res is None or (isinstance(res, pd.DataFrame) and res.empty):
                    fut.release()
                    continue

                else:
                    if not header_written:
                        res.to_csv(args.output_csv, index=False, mode='w')
                        header_written = True
                    else:
                        res.to_csv(args.output_csv, index=False, mode='a', header=False)
                write_count += 1
            finally:
                fut.release()

        logger.info(f"Finished processing all groups and wrote {write_count} output parts.")
    except Exception as e:
        logger.error(f"Error in main processing block: {e}", exc_info=True)
        sys.exit(1)

if __name__ == "__main__":
    main()