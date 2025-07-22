# Standard library dependencies
import os, pathlib, tempfile, functools, time, datetime, ast, sys, argparse, gc, logging, math

# non-stdlib core dependencies
import dask.array as da
import numpy as np
import scipy
import cellpose.io
from cellpose import models
from skimage.filters import threshold_otsu


# existing distributed dependencies
import distributed
import dask_image.ndmeasure
import zarr
import logging
import torch
import math
import dask_jobqueue
import subprocess
from threading import Thread

# --- Import Dask utility functions ---
from utils.dask_utils import setup_dask_sge_cluster, shutdown_dask, change_worker_attributes

# --- Configure Logging ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
cellpose.io.logger_setup()

# def monitor_gpu(log_file):
#     """Run nvidia-smi and write output to a log file with line buffering."""
#     # Use stdbuf to force line buffering (Linux/macOS)
#     proc = subprocess.Popen(
#         ["stdbuf", "-oL",  # Line buffer stdout
#          "nvidia-smi",
#          "--query-gpu=timestamp,memory.used",
#          "--format=csv",
#          "-l", "10"],  # Log every 1 second
#         stdout=subprocess.PIPE,
#         universal_newlines=True  # Handle text output
#     )

#     with open(log_file, "w") as f:
#         for line in proc.stdout:
#             f.write(line)
#             f.flush()  # Ensure immediate write

# monitor_thread = Thread(target=monitor_gpu, args=("/net/beliveau/vol1/project/VB_Segmentation/subprojects/OTLS-Analyzer/logs/test_rp_gpu_memory.log",))
# monitor_thread.daemon = True  # Terminate thread when main program exits
# monitor_thread.start()


# Reduce verbosity of dask logs if desired
# logging.getLogger('distributed').setLevel(logging.WARNING)
# logging.getLogger('distributed.worker').setLevel(logging.WARNING)
# logging.getLogger('distributed.scheduler').setLevel(logging.WARNING)
# logging.getLogger('distributed.client').setLevel(logging.WARNING)

# Optional: PyTorch configuration
# torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", 2))) # Use env var if set

######################## Helper Functions (Parsing, etc.) ######################

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


# Define normalization function using global percentiles
def normalize_chunk(image, percentile_low, percentile_high, crop=None):
    """Normalize a chunk of image data using global percentiles.

    Parameters
    ----------
    image : numpy.ndarray
        The image data to normalize.
    percentile_low : float
        The low percentile of the image data.
    percentile_high : float
        The high percentile of the image data.
    crop : tuple of slice objects
        The bounding box of the data to read from the input_zarr array

    Returns
    -------
    numpy.ndarray
        The normalized image data."""
    
    image = image.astype(np.float32)
    denom = percentile_high - percentile_low
    if denom < 1e-8:
        logger.warning(f"Normalization range near zero ({denom:.2e}). Clipping image instead of normalizing.")
        image = np.clip(image, 0, 1)
    else:
        image = (image - percentile_low) / (denom + 1e-8)  # Add epsilon for stability
        image = np.clip(image, 0, 1)  # Clip to [0, 1]
    #image = (image * 255).astype(np.uint8)  # Convert to 8-bit for CellPose compatibility (Optional, depends on model)
    return image

######################## the function to run on each block ####################

#----------------------- The main function -----------------------------------#
def process_block(
    block_index,
    crop,
    input_zarr_path, # Pass path instead of object
    n5_subpath, # Pass subpath if relevant
    model_path,
    model_kwargs,
    eval_kwargs,
    blocksize,
    overlap,
    output_zarr_path, # Pass path instead of object
    preprocessing_steps=[],
    test_mode=False,
    global_p99=None,
    global_p1=None
):
    """
    Preprocess and segment one block, of many, with eventual merger
    of all blocks in mind. The block is processed as follows:

    (1) Read block from disk, preprocess, and segment.
    (2) Remove overlaps.
    (3) Get bounding boxes for every segment.
    (4) Remap segment IDs to globally unique values.
    (5) Write segments to disk.
    (6) Get segmented block faces.

    A user may want to test this function on one block before running
    the distributed function. When test_mode=True, steps (5) and (6)
    are omitted and replaced with:

    (5) return remapped segments as a numpy array, boxes, and box_ids

    Parameters
    ----------
    block_index : tuple
        The (i, j, k, ...) index of the block in the overall block grid

    crop : tuple of slice objects
        The bounding box of the data to read from the input_zarr array

    input_zarr : zarr.core.Array
        The image data we want to segment

    preprocessing_steps : list of tuples (default: the empty list)
        Optionally apply an arbitrary pipeline of preprocessing steps
        to the image block before running cellpose.

        Must be in the following format:
        [(f, {'arg1':val1, ...}), ...]
        That is, each tuple must contain only two elements, a function
        and a dictionary. The function must have the following signature:
        def F(image, ..., crop=None)
        That is, the first argument must be a numpy array, which will later
        be populated by the image data. The function must also take a keyword
        argument called crop, even if it is not used in the function itself.
        All other arguments to the function are passed using the dictionary.
        Here is an example:

        def F(image, sigma, crop=None):
            return gaussian_filter(image, sigma)
        def G(image, radius, crop=None):
            return median_filter(image, radius)
        preprocessing_steps = [(F, {'sigma':2.0}), (G, {'radius':4})]

    model_kwargs : dict
        Arguments passed to cellpose.models.Cellpose
        This is how you select and parameterize a model.

    eval_kwargs : dict
        Arguments passed to the eval function of the Cellpose model
        This is how you parameterize model evaluation.

    blocksize : iterable (list, tuple, np.ndarray)
        The number of voxels (the shape) of blocks without overlaps

    overlap : int
        The number of voxels added to the blocksize to provide context
        at the edges

    output_zarr : zarr.core.Array
        A location where segments can be stored temporarily before
        merger is complete

    worker_logs_directory : string (default: None)
        A directory path where log files for each worker can be created
        The directory must exist

    test_mode : bool (default: False)
        The primary use case of this function is to be called by
        distributed_eval (defined later in this same module). However
        you may want to call this function manually to test what
        happens to an individual block; this is a good idea before
        ramping up to process big data and also useful for debugging.

        When test_mode is False (default) this function stores
        the segments and returns objects needed for merging between
        blocks.

        When test_mode is True this function does not store the
        segments, and instead returns them to the caller as a numpy
        array. The boxes and box IDs are also returned. When test_mode
        is True, you can supply dummy values for many of the inputs,
        such as:

        block_index = (0, 0, 0)
        output_zarr=None

    Returns
    -------
    If test_mode == False (the default), three things are returned:
        faces : a list of numpy arrays - the faces of the block segments
        boxes : a list of crops (tuples of slices), bounding boxes of segments
        box_ids : 1D numpy array, parallel to boxes, the segment IDs of the
                  boxes

    If test_mode == True, three things are returned:
        segments : np.ndarray containing the segments with globally unique IDs
        boxes : a list of crops (tuples of slices), bounding boxes of segments
        box_ids : 1D numpy array, parallel to boxes, the segment IDs of the
                  boxes
    """
    # --- Lazy load Zarr arrays within the worker ---
    try:
        if input_zarr_path.endswith(".n5"):
            store = zarr.N5Store(input_zarr_path)
            input_arr = zarr.open_array(store=store, path=n5_subpath, mode='r')
        else: # Assume .zarr
            input_arr = zarr.open(input_zarr_path, mode='r')

        if output_zarr_path:
            output_arr = zarr.open(output_zarr_path, mode='r+') # Open read-write
        else:
            output_arr = None
    except Exception as e:
        logger.error(f"Block {block_index}: Error opening Zarr arrays. Input: {input_zarr_path}, Output: {output_zarr_path}. Error: {e}", exc_info=True)
        # Return dummy values matching the expected output structure on error
        empty_face = np.array([[[0]]], dtype=np.uint32) # Minimal face structure
        ndim = len(blocksize) # Infer ndim
        return [empty_face] * (2 * ndim), [], np.array([0], dtype=np.uint32)



    logger.info(f'RUNNING BLOCK: {block_index}\tREGION: {crop}\tInput: {input_zarr_path}')

    segmentation = read_preprocess_and_segment(
        input_arr, crop, preprocessing_steps, model_path, model_kwargs, eval_kwargs, global_p99, global_p1
    )

    logger.debug(f"Block {block_index}: Segmentation shape raw: {segmentation.shape}, type: {type(segmentation)}")

    segmentation_trimmed, crop_trimmed = remove_overlaps(
        segmentation, crop, overlap, blocksize,
    )
    logger.debug(f"Block {block_index}: Segmentation shape trimmed: {segmentation_trimmed.shape}")

    boxes = bounding_boxes_in_global_coordinates(segmentation_trimmed, crop_trimmed) # Use trimmed crop for global coords

    # Ensure blocksize matches input_arr dimensions if possible
    actual_ndim = input_arr.ndim
    if len(blocksize) != actual_ndim:
        logger.warning(f"Block {block_index}: blocksize length {len(blocksize)} doesn't match input array ndim {actual_ndim}. Using input ndim.")
        # Try to adapt blocksize or raise error - for now, use actual ndim for nblocks calc
        nblocks = get_nblocks(input_arr.shape, blocksize[:actual_ndim])
    else:
        nblocks = get_nblocks(input_arr.shape, blocksize)


    segmentation_remapped, remap = global_segment_ids(segmentation_trimmed, block_index, nblocks)
    if remap[0] == 0: remap = remap[1:]
    logger.debug(f"Block {block_index}: Segmentation shape final: {segmentation_remapped.shape}, Remap len: {len(remap)}")
    if output_arr:
        logger.debug(f"Block {block_index}: Output Zarr shape: {output_arr.shape}, chunks: {output_arr.chunks}")


    if test_mode:
        return segmentation_remapped, boxes, remap

    if output_arr is None:
         logger.error(f"Block {block_index}: output_zarr_path not provided in non-test mode.")
         # Return dummy values
         empty_face = np.array([[[0]]], dtype=np.uint32)
         ndim = len(blocksize)
         return [empty_face] * (2 * ndim), [], np.array([0], dtype=np.uint32)

    try:
        logger.info(f"Block {block_index}: Writing segmentation shape {segmentation.shape} to crop {tuple(crop_trimmed)}")
        output_arr[tuple(crop_trimmed)] = segmentation_remapped
    except Exception as e:
        logger.error(f"Block {block_index}: Failed to write segmentation to Zarr {output_zarr_path} at crop {tuple(crop_trimmed)}. Error: {e}", exc_info=True)
        # Return dummy values
        empty_face = np.array([[[0]]], dtype=np.uint32)
        ndim = len(blocksize)
        return [empty_face] * (2 * ndim), [], np.array([0], dtype=np.uint32)


    faces = block_faces(segmentation_remapped)
    if torch.cuda.is_available():
        torch.cuda.empty_cache() # Still potentially useful
    gc.collect() # Explicit GC call

    return faces, boxes, remap

#----------------------- component functions ---------------------------------#
def read_preprocess_and_segment(
    input_zarr, # Now expects opened zarr.Array
    crop,
    preprocessing_steps,
    model_path,
    model_kwargs,   
    eval_kwargs,
    global_p99,
    global_p1
):
    """Read block from zarr array, run all preprocessing steps, run cellpose"""
    logger.debug(f"read_preprocess_and_segment: Reading crop: {crop}")
    try:
        image = input_zarr[crop]
    except Exception as e:
         logger.error(f"Failed to read crop {crop} from input array. Error: {e}", exc_info=True)
         raise

    logger.debug(f"read_preprocess_and_segment: Image shape: {image.shape}, dtype: {image.dtype}")

    # for pp_step in preprocessing_steps:
    #     # Assuming pp_step functions handle potential errors
    #     try:
    #         func = pp_step[0]
    #         kwargs = pp_step[1]
    #         kwargs['crop'] = crop # Add crop info
    #         image = func(image, **kwargs)
    #         # logger.debug(f"Applied preprocessing step {func.__name__}, new shape {image.shape}, dtype {image.dtype}")
    #     except Exception as e:
    #          logger.error(f"Error during preprocessing step {pp_step[0].__name__}: {e}", exc_info=True)
    #          raise # Or handle more gracefully

    # TODO: Add preprocessing steps to list for handling

    # Normalize the image using the global 1% and 99% values
    image = image.astype(np.float32)
    denom = global_p99 - global_p1
    if denom < 1e-8:
        logger.warning(f"Normalization range near zero ({denom:.2e}). Clipping image instead of normalizing.")
        image = np.clip(image, 0, 1)
    else:
        image = (image - global_p1) / (denom + 1e-8)  # Add epsilon for stability
        image = np.clip(image, 0, 1)  # Clip to [0, 1]
    # args = [32, 32, 'db8'] # Example args for destriping, handle if used
    # img_destriped = destripe_image(image, args)

    # Load the model using the provided path
    logger.debug(f"Loading Cellpose model from: {model_path}")
    try:
        # Consider adding device selection ('cuda' or 'cpu') to model_kwargs
        gpu_available = torch.cuda.is_available()
        logger.debug(f"GPU available: {gpu_available}")
        model = models.CellposeModel(gpu=gpu_available, pretrained_model=model_path, **model_kwargs)
        # model = models.Cellpose(gpu=gpu_available, model_type='cyto', **model_kwargs) # Example using Cellpose base class

        logger.debug(f"Cellpose model loaded. Using eval_kwargs: {eval_kwargs}")
        # Before eval, check memory
        if gpu_available:
            logger.debug("GPU Memory before eval: Allocated=%.2fGB, Reserved=%.2fGB",
                         torch.cuda.memory_allocated(0)/1e9, torch.cuda.memory_reserved(0)/1e9)

        # Ensure eval_kwargs does not contain gpu/model_type if using CellposeModel
        eval_kwargs_clean = eval_kwargs.copy()
        eval_kwargs_clean.pop('gpu', None)
        eval_kwargs_clean.pop('model_type', None)
        eval_kwargs_clean.pop('pretrained_model', None)


        masks = model.eval(image, **eval_kwargs_clean)[0]
        # After eval, check memory
        if gpu_available:
            logger.debug("GPU Memory after eval: Allocated=%.2fGB, Reserved=%.2fGB",
                         torch.cuda.memory_allocated(0)/1e9, torch.cuda.memory_reserved(0)/1e9)

        return masks.astype(np.uint32) # Ensure correct dtype
    except Exception as e:
        logger.error(f"Error during Cellpose model loading or evaluation: {e}", exc_info=True)
        raise # Propagate error

def remove_overlaps(array, crop, overlap, blocksize):
    """overlaps only there to provide context for boundary voxels
       and can be removed after segmentation is complete
       reslice array to remove the overlaps"""
    crop_trimmed = list(crop)
    for axis in range(array.ndim):
        if crop[axis].start != 0:
            slc = [slice(None),]*array.ndim
            slc[axis] = slice(overlap, None)
            array = array[tuple(slc)]
            a, b = crop[axis].start, crop[axis].stop
            crop_trimmed[axis] = slice(a + overlap, b)
        if array.shape[axis] > blocksize[axis]:
            slc = [slice(None),]*array.ndim
            slc[axis] = slice(None, blocksize[axis])
            array = array[tuple(slc)]
            a = crop_trimmed[axis].start
            crop_trimmed[axis] = slice(a, a + blocksize[axis])
    return array, crop_trimmed


def bounding_boxes_in_global_coordinates(segmentation, crop):
    """bounding boxes (tuples of slices) are super useful later
       best to compute them now while things are distributed"""
    boxes = scipy.ndimage.find_objects(segmentation)
    boxes = [b for b in boxes if b is not None]
    translate = lambda a, b: slice(a.start+b.start, a.start+b.stop)
    for iii, box in enumerate(boxes):
        boxes[iii] = tuple(translate(a, b) for a, b in zip(crop, box))
    return boxes


def get_nblocks(shape, blocksize):
    """Given a shape and blocksize determine the number of blocks per axis"""
    return np.ceil(np.array(shape) / blocksize).astype(int)


def global_segment_ids(segmentation, block_index, nblocks):
    """pack the block index into the segment IDs so they are
       globally unique. Everything gets remapped to [1..N] later.
       A uint32 is split into 5 digits on left and 5 digits on right.
       This creates limits: 42950 maximum number of blocks and
       99999 maximum number of segments per block"""
    unique, unique_inverse = np.unique(segmentation, return_inverse=True)
    p = str(np.ravel_multi_index(block_index, nblocks))
    remap = [np.uint32(p+str(x).zfill(5)) for x in unique]
    if unique[0] == 0: remap[0] = np.uint32(0)  # 0 should just always be 0
    segmentation = np.array(remap)[unique_inverse.reshape(segmentation.shape)]
    return segmentation, remap


def block_faces(segmentation):
    """slice faces along every axis"""
    faces = []
    for iii in range(segmentation.ndim):
        a = [slice(None),] * segmentation.ndim
        a[iii] = slice(0, 1)
        faces.append(segmentation[tuple(a)])
        a = [slice(None),] * segmentation.ndim
        a[iii] = slice(-1, None)
        faces.append(segmentation[tuple(a)])
    return faces

######################## Distributed Cellpose (Modified) ################################


def distributed_eval(
    input_source, # Path to N5/Zarr or Zarr Array object
    n5_subpath,   # Subpath if input_source is N5 path
    blocksize,    # Tuple
    write_path,   # Output Zarr path
    model_path,
    client: distributed.Client, # Pass active client
    mask=None, # Subpath if mask_source is N5 path
    preprocessing_steps=[],
    model_kwargs={},
    eval_kwargs={},
    temporary_directory=None, # Base for temp files
):
    """
    Evaluate a cellpose model on overlapping blocks of a big image.
    Distributed over workstation or cluster resources with Dask.
    Optionally run preprocessing steps on the blocks before running cellpose.
    Optionally use a mask to ignore background regions in image.
    Either cluster or cluster_kwargs parameter must be set to a
    non-default value; please read these parameter descriptions below.
    If using cluster_kwargs, the workstation and Janelia LSF cluster cases
    are distinguished by the arguments present in the dictionary.

    PC/Mac/Linux workstations and the Janelia LSF cluster are supported;
    running on a different institute cluster will require implementing your
    own dask cluster class. Look at the JaneliaLSFCluster class in this
    module as an example, also look at the dask_jobqueue library. A PR with
    a solid start is the right way to get help running this on your own
    institute cluster.

    If running on a workstation, please read the docstring for the
    LocalCluster class defined in this module. That will tell you what to
    put in the cluster_kwargs dictionary. If using the Janelia cluster,
    please read the docstring for the JaneliaLSFCluster class.

    Parameters
    ----------
    input_zarr : zarr.core.Array
        A zarr.core.Array instance containing the image data you want to
        segment.

    blocksize : iterable
        The size of blocks in voxels. E.g. [128, 256, 256]

    write_path : string
        The location of a zarr file on disk where you'd like to write your results

    mask : numpy.ndarray (default: None)
        A foreground mask for the image data; may be at a different resolution
        (e.g. lower) than the image data. If given, only blocks that contain
        foreground will be processed. This can save considerable time and
        expense. It is assumed that the domain of the input_zarr image data
        and the mask is the same in physical units, but they may be on
        different sampling/voxel grids.

    preprocessing_steps : list of tuples (default: the empty list)
        Optionally apply an arbitrary pipeline of preprocessing steps
        to the image blocks before running cellpose.

        Must be in the following format:
        [(f, {'arg1':val1, ...}), ...]
        That is, each tuple must contain only two elements, a function
        and a dictionary. The function must have the following signature:
        def F(image, ..., crop=None)
        That is, the first argument must be a numpy array, which will later
        be populated by the image data. The function must also take a keyword
        argument called crop, even if it is not used in the function itself.
        All other arguments to the function are passed using the dictionary.
        Here is an example:

        def F(image, sigma, crop=None):
            return gaussian_filter(image, sigma)
        def G(image, radius, crop=None):
            return median_filter(image, radius)
        preprocessing_steps = [(F, {'sigma':2.0}), (G, {'radius':4})]

    model_kwargs : dict (default: {})
        Arguments passed to cellpose.models.Cellpose

    eval_kwargs : dict (default: {})
        Arguments passed to cellpose.models.Cellpose.eval

    cluster : A dask cluster object (default: None)
        Only set if you have constructed your own static cluster. The default
        behavior is to construct a dask cluster for the duration of this function,
        then close it when the function is finished.

    cluster_kwargs : dict (default: {})
        Arguments used to parameterize your cluster.
        If you are running locally, see the docstring for the myLocalCluster
        class in this module. If you are running on the Janelia LSF cluster, see
        the docstring for the janeliaLSFCluster class in this module. If you are
        running on a different institute cluster, you may need to implement
        a dask cluster object that conforms to the requirements of your cluster.

    temporary_directory : string (default: None)
        Temporary files are created during segmentation. The temporary files
        will be in their own folder within the temporary_directory. The default
        is the current directory. Temporary files are removed if the function
        completes successfully.

    Returns
    -------
    Two values are returned:
    (1) A reference to the zarr array on disk containing the stitched cellpose
        segments for your entire image
    (2) Bounding boxes for every segment. This is a list of tuples of slices:
        [(slice(z1, z2), slice(y1, y2), slice(x1, x2)), ...]
        The list is sorted according to segment ID. That is the smallest segment
        ID is the first tuple in the list, the largest segment ID is the last
        tuple in the list.
    """
    logger.info("Starting distributed Cellpose evaluation...")
    logger.info(f"Input source: {input_source}")
    logger.info(f"Output Zarr: {write_path}")
    logger.info(f"Blocksize: {blocksize}")
    cluster = client.cluster


    # --- Open Input Array ---
    # We need the shape and dtype upfront. Open read-only here.
    # Workers will reopen based on path.
    input_is_path = isinstance(input_source, (str, pathlib.Path))
    try:
        if input_is_path:
            if str(input_source).endswith(".n5"):
                store = zarr.N5Store(str(input_source))
                input_arr_handle = zarr.open_array(store=store, path=n5_subpath, mode='r')
                input_path_for_workers = str(input_source)
                input_n5_subpath_for_workers = n5_subpath
            else: # Assume Zarr path
                input_arr_handle = zarr.open(str(input_source), mode='r')
                input_path_for_workers = str(input_source)
                input_n5_subpath_for_workers = None
        else: # Assume already an open Zarr array
            input_arr_handle = input_source
            # Need to get path info if possible for workers
            input_path_for_workers = input_arr_handle.store.path if hasattr(input_arr_handle.store, 'path') else None
            input_n5_subpath_for_workers = input_arr_handle.path if input_path_for_workers and input_path_for_workers.endswith(".n5") else None
            if not input_path_for_workers:
                 logger.warning("Input array provided directly, worker path inference might fail if not standard Zarr/N5.")
                 # This path might not work if input_source was created in memory
                 input_path_for_workers = "memory_array" # Placeholder

        input_shape = input_arr_handle.shape
        input_dtype = input_arr_handle.dtype
        logger.info(f"Opened input: Shape={input_shape}, Dtype={input_dtype}, Chunks={input_arr_handle.chunks}")
        # Ensure blocksize matches dimensionality
        if len(blocksize) != input_arr_handle.ndim:
             raise ValueError(f"Blocksize dimensions ({len(blocksize)}) must match input data dimensions ({input_arr_handle.ndim})")

    except Exception as e:
        logger.error(f"Failed to open input array {input_source}: {e}", exc_info=True)
        raise



    overlap = 90 # should not hardcode this
    logger.info(f"calculated overlap={overlap}")

    block_indices, block_crops = get_block_crops(
        input_shape, blocksize, overlap, mask=mask, # Pass opened mask handle
    )
    logger.info(f"Calculated {len(block_indices)} blocks to process.")
    if not block_indices:
        logger.warning("No blocks selected for processing (potentially due to mask). Exiting.")
        # Create empty output zarr and return
        zarr.open(write_path, mode='w', shape=input_shape, chunks=blocksize, dtype=np.uint32)
        return zarr.open(write_path, mode='r'), []

    # --- Setup Temporary Directory for Stitching Info ---
    # Use a context manager to ensure cleanup
    with tempfile.TemporaryDirectory(prefix='cellpose_stitch_', dir=temporary_directory or os.getcwd()) as temp_dir_path:
        logger.info(f"Using temporary directory for stitching: {temp_dir_path}")
        temp_zarr_path = os.path.join(temp_dir_path, 'segmentation_unstitched.zarr')

        # --- Create Intermediate Zarr Store ---
        try:
            temp_zarr = zarr.open(
                temp_zarr_path, 'w',
                shape=input_shape,
                chunks=blocksize,
                dtype=np.uint32, # Cellpose output is uint32/int32, remap to uint32
            )
            logger.info(f"Created temporary Zarr for unstitched segments: {temp_zarr_path}")
        except Exception as e:
             logger.error(f"Failed to create temporary Zarr store at {temp_zarr_path}: {e}", exc_info=True)
             raise

        # Compute global 1% and 99% percentiles using Dask
        # TODO: only calculate percentiles on array within foreground mask

        dask_array = da.from_zarr(input_arr_handle, component=input_n5_subpath_for_workers)
        global_p1, global_p99 = da.percentile(da.ravel(dask_array), [1,99]).compute()
        global_p1, global_p99 = int(global_p1), int(global_p99)
        logger.info(f"Global 1% percentile: {global_p1}, Global 99% percentile: {global_p99}")
        # --- Submit Blocks for Processing ---
        futures = client.map(
            process_block,
            block_indices,
            block_crops,
            input_zarr_path=input_path_for_workers, # Pass path
            n5_subpath=input_n5_subpath_for_workers, # Pass N5 subpath
            output_zarr_path=temp_zarr_path,         # Pass path
            model_path=model_path,
            preprocessing_steps=preprocessing_steps,
            model_kwargs=model_kwargs,
            eval_kwargs=eval_kwargs,
            blocksize=blocksize,
            overlap=overlap,
            test_mode=False, # Ensure test_mode is False
            global_p99=global_p99,
            global_p1=global_p1,
            # Dask scheduling options:
            # priority=10, # Example priority
            # resources={'GPU': 1} if torch.cuda.is_available() else {}, # If workers have GPU resources defined
            # workers= ... # Target specific workers if needed
        )
        logger.info(f"Submitted {len(futures)} blocks for processing.")

        # --- Gather Results ---
        logger.info("Gathering results (faces, boxes, box_ids)...")
        results = client.gather(futures, errors='skip') # Skip blocks that failed
        logger.info(f"Gathered results from {len(results)} completed blocks.")
        if isinstance(cluster, dask_jobqueue.core.JobQueueCluster): 
            cluster.scale(0)


        # Filter out None results from skipped futures if gather(errors='skip') returns them
        if not results:
             logger.error("No blocks completed successfully.")
             # Consider cleanup or returning empty result
             # For now, raise error as stitching cannot proceed
             raise RuntimeError("Distributed evaluation failed: No blocks completed.")


        logger.info("Processing gathered results for stitching...")
        try:
            faces, boxes_, box_ids_ = list(zip(*results))
            boxes = [box for sublist in boxes_ for box in sublist]
            box_ids = np.concatenate(box_ids_).astype(int)  # unsure how but without cast these are float64
            new_labeling = determine_merge_relabeling(block_indices, faces, box_ids)
            new_labeling_path = temporary_directory + '/new_labeling.npy'
            np.save(new_labeling_path, new_labeling)
            logger.info(f"Saved new labeling map to {new_labeling_path}")

            # TODO: Add GPU release for stitching step

            # if isinstance(cluster, dask_jobqueue.core.JobQueueCluster): 
            #     change_worker_attributes(
            #         cluster,
            #         min_workers=1,
            #         max_workers=8,
            #         ncpus=1,
            #         memory="15GB",
            #         mem=int(15e9),
            #         queue=None,
            #         job_extra_directives=[],
            #     )

        except Exception as e:
             logger.error(f"Error during result processing or relabeling: {e}", exc_info=True)
             raise

        # --- Apply Relabeling with Dask ---
        try:
            logger.info("Applying final relabeling using Dask map_blocks...")
            segmentation_da = da.from_zarr(temp_zarr)

            relabeled = da.map_blocks(
            lambda block: np.load(new_labeling_path)[block],
            segmentation_da,
            dtype=np.uint32,
            chunks=segmentation_da.chunks,
        )
            da.to_zarr(relabeled, write_path, overwrite=True)
            merged_boxes = merge_all_boxes(boxes, new_labeling[box_ids])
        except Exception as e:
            logger.error(f"Error during final relabeling: {e}", exc_info=True)
            raise
        return zarr.open(write_path, mode='r'), merged_boxes



def get_block_crops(shape, blocksize, overlap, mask):
    """Given a voxel grid shape, blocksize, and overlap size, construct
       tuples of slices for every block; optionally only include blocks
       that contain foreground in the mask. Returns parallel lists,
       the block indices and the slice tuples."""
    blocksize = np.array(blocksize)
    if mask is not None:
        ratio = np.array(mask.shape) / shape
        mask_blocksize = np.round(ratio * blocksize).astype(int)

    indices, crops = [], []
    nblocks = get_nblocks(shape, blocksize)
    for index in np.ndindex(*nblocks):
        start = blocksize * index - overlap
        stop = start + blocksize + 2 * overlap
        start = np.maximum(0, start)
        stop = np.minimum(shape, stop)
        crop = tuple(slice(x, y) for x, y in zip(start, stop))

        foreground = True
        if mask is not None:
            start = mask_blocksize * index
            stop = start + mask_blocksize
            stop = np.minimum(mask.shape, stop)
            mask_crop = tuple(slice(x, y) for x, y in zip(start, stop))
            if not np.any(mask[mask_crop]): foreground = False
        if foreground:
            indices.append(index)
            crops.append(crop)
    return indices, crops


def determine_merge_relabeling(block_indices, faces, used_labels):
    """Determine boundary segment mergers, remap all label IDs to merge
       and put all label IDs in range [1..N] for N global segments found"""
    faces = adjacent_faces(block_indices, faces)
    # FIX float parameters
    # print("Used labels:", used_labels, "Type:", type(used_labels))
    used_labels = used_labels.astype(int)
    # print("Used labels:", used_labels, "Type:", type(used_labels))
    label_range = int(np.max(used_labels))

    label_groups = block_face_adjacency_graph(faces, label_range)
    new_labeling = scipy.sparse.csgraph.connected_components(
        label_groups, directed=False)[1]
    # XXX: new_labeling is returned as int32. Loses half range. Potentially a problem.
    unused_labels = np.ones(label_range + 1, dtype=bool)
    unused_labels[used_labels] = 0
    new_labeling[unused_labels] = 0
    unique, unique_inverse = np.unique(new_labeling, return_inverse=True)
    new_labeling = np.arange(len(unique), dtype=np.uint32)[unique_inverse]
    return new_labeling


def adjacent_faces(block_indices, faces):
    """Find faces which touch and pair them together in new data structure"""
    face_pairs = []
    faces_index_lookup = {a:b for a, b in zip(block_indices, faces)}
    for block_index in block_indices:
        for ax in range(len(block_index)):
            neighbor_index = np.array(block_index)
            neighbor_index[ax] += 1
            neighbor_index = tuple(neighbor_index)
            try:
                a = faces_index_lookup[block_index][2*ax + 1]
                b = faces_index_lookup[neighbor_index][2*ax]
                face_pairs.append( np.concatenate((a, b), axis=ax) )
            except KeyError:
                continue
    return face_pairs


def block_face_adjacency_graph(faces, nlabels):
    """Shrink labels in face plane, then find which labels touch across the
    face boundary"""
    # FIX float parameters
    # print("Initial nlabels:", nlabels, "Type:", type(nlabels))
    nlabels = int(nlabels)
    # print("Final nlabels:", nlabels, "Type:", type(nlabels))

    all_mappings = []
    structure = scipy.ndimage.generate_binary_structure(3, 1)
    for face in faces:
        sl0 = tuple(slice(0, 1) if d==2 else slice(None) for d in face.shape)
        sl1 = tuple(slice(1, 2) if d==2 else slice(None) for d in face.shape)
        a = shrink_labels(face[sl0], 1.0)
        b = shrink_labels(face[sl1], 1.0)
        face = np.concatenate((a, b), axis=np.argmin(a.shape))
        mapped = dask_image.ndmeasure._utils._label._across_block_label_grouping(face, structure)
        all_mappings.append(mapped)
    i, j = np.concatenate(all_mappings, axis=1)
    v = np.ones_like(i)
    return scipy.sparse.coo_matrix((v, (i, j)), shape=(nlabels+1, nlabels+1)).tocsr()


def shrink_labels(plane, threshold):
    """Shrink labels in plane by some distance from their boundary"""
    gradmag = np.linalg.norm(np.gradient(plane.squeeze()), axis=0)
    shrunk_labels = np.copy(plane.squeeze())
    shrunk_labels[gradmag > 0] = 0
    distances = scipy.ndimage.distance_transform_edt(shrunk_labels)
    shrunk_labels[distances <= threshold] = 0
    return shrunk_labels.reshape(plane.shape)


def merge_all_boxes(boxes, box_ids):
    """Merge all boxes that map to the same box_ids"""
    merged_boxes = []
    boxes_array = np.array(boxes, dtype=object)
    # FIX float parameters
    # print("Box IDs:", box_ids, "Type:", type(box_ids))
    box_ids = box_ids.astype(int)
    # print("Box IDs:", box_ids, "Type:", type(box_ids))

    for iii in np.unique(box_ids):
        merge_indices = np.argwhere(box_ids == iii).squeeze()
        if merge_indices.shape:
            merged_box = merge_boxes(boxes_array[merge_indices])
        else:
            merged_box = boxes_array[merge_indices]
        merged_boxes.append(merged_box)
    return merged_boxes


def merge_boxes(boxes):
    """Take union of two or more parallelpipeds"""
    box_union = boxes[0]
    for iii in range(1, len(boxes)):
        local_union = []
        for s1, s2 in zip(box_union, boxes[iii]):
            start = min(s1.start, s2.start)
            stop = max(s1.stop, s2.stop)
            local_union.append(slice(start, stop))
        box_union = tuple(local_union)
    return box_union

def get_foreground_mask(input_n5, n5_subpath):
    """Create the foreground mask from the input Zarr array"""
    # Get the shape of the input Zarr array
    logger.info(f"Getting foreground mask for {input_n5} [{n5_subpath}]")
    store = zarr.N5Store(str(input_n5))
    input_arr_handle = zarr.open_array(store=store, path=n5_subpath, mode='r')
    arr_zarr = da.from_zarr(input_arr_handle)
    shape = input_arr_handle.shape
    # Create a mask of all zeros with the same shape as the input Zarr array
    mask = np.zeros(shape, dtype=bool)
    # Set the mask to True where the input Zarr array is above otsu threshold

    thresh = threshold_otsu(arr_zarr)
    mask[arr_zarr > thresh] = True
    logger.info(f"Foreground mask created with shape {mask.shape}")
    return mask


def main():
    parser = argparse.ArgumentParser(description="Distributed Cellpose Segmentation")

    # Input/Output
    parser.add_argument("--input_n5", required=True, help="Path to input N5 store")
    parser.add_argument("--n5_channel_path", required=True, help="Path to dataset within N5 store (e.g., ch0/s0)")
    parser.add_argument("--output_zarr", required=True, help="Path for output Zarr segmentation")
    parser.add_argument("--model_path", required=True, help="Path to pretrained Cellpose model file")

    # Processing Params
    parser.add_argument("--block_size", type=str, required=True, help="Block size (Z,Y,X), comma-separated")
    parser.add_argument("--eval_kwargs", type=str, default="{}", help="String representation of dict for model.eval() kwargs")
    parser.add_argument("--temporary_dir", default=None, help="Base directory for temporary files")

    # Dask SGE Cluster Arguments (GPU specific)
    parser.add_argument("--num_workers", type=int, default=16, help="Number of Dask workers (SGE jobs)")
    parser.add_argument("--cores_per_worker", type=int, default=2, help="Number of cores per worker")
    parser.add_argument("--mem_per_worker", default="60G", help="Memory per worker (e.g., '60G')")
    parser.add_argument("--processes", type=int, default=1, help="Number of Python processes per worker (usually 1 for GPU)")
    parser.add_argument("--project", required=True, help="SGE project code")
    parser.add_argument("--queue", required=True, help="SGE queue name")
    parser.add_argument("--runtime", default="1400000", help="Job runtime (SGE format or seconds)")
    parser.add_argument("--resource_spec", default="gpgpu=1,cuda=1", help="SGE resource specification (e.g., 'gpgpu=1,cuda=1')")
    parser.add_argument("--log_dir", default=None, help="Directory for Dask worker logs")
    parser.add_argument("--conda_env", default="otls-pipeline", help="Conda environment to activate on workers")


    # --- Argument Parsing ---
    if 'snakemake' in globals():
        logger.info("Running under Snakemake.")
        args = argparse.Namespace(
            input_n5=snakemake.input.n5,
            n5_channel_path=snakemake.params.n5_channel_path,
            output_zarr=snakemake.output.zarr,
            block_size=snakemake.params.block_size,
            model_path=snakemake.params.model_path,
            eval_kwargs=snakemake.params.eval_kwargs, # Already a string
            temporary_dir=snakemake.config['dask'].get('log_dir', None), # Use dask log dir for temp? Or specific temp path?
            # Dask params from resources
            num_workers=snakemake.resources.n_gpu_workers,
            cores_per_worker=snakemake.resources.gpu_cores,
            mem_per_worker=snakemake.resources.gpu_worker_memory,
            processes=snakemake.resources.gpu_processes,
            project=snakemake.resources.project,
            queue=snakemake.resources.gpu_queue,
            runtime=str(snakemake.resources.runtime),
            resource_spec=snakemake.resources.gpu_resource_spec,
            log_dir=snakemake.config['dask'].get('log_dir', None), # Reuse dask log dir
            conda_env=snakemake.conda_env_name if hasattr(snakemake, 'conda_env_name') else "otls-pipeline",
            dashboard_port=snakemake.resources.dashboard_port
        )
    else:
        logger.info("Parsing command-line arguments.")
        args = parser.parse_args()

    # --- Parameter Processing ---
    try:
        block_size_tuple = parse_tuple_string(args.block_size)
        eval_kwargs_dict = ast.literal_eval(args.eval_kwargs)
        if not isinstance(eval_kwargs_dict, dict):
            raise ValueError("eval_kwargs must be a dictionary string.")
        logger.info(f"Using eval_kwargs: {eval_kwargs_dict}")

    except Exception as e:
        logger.error(f"Error processing parameters: {e}", exc_info=True)
        sys.exit(1)

    # --- Dask Cluster Setup ---
    cluster = None
    client = None
    try:
        logger.info("Setting up Dask cluster for segmentation...")
        cluster, client = setup_dask_sge_cluster(
            n_workers=args.num_workers,
            cores=args.cores_per_worker,
            processes=args.processes,
            memory=args.mem_per_worker,
            project=args.project,
            queue=args.queue,
            runtime=args.runtime,
            resource_spec=args.resource_spec,
            log_directory=args.log_dir + "/segmentation/dask_worker_logs_" + datetime.datetime.now().strftime('%Y%m%d_%H%M%S'),
            conda_env=args.conda_env,
            dashboard_port=args.dashboard_port
        )
    except Exception as e:
        logger.error(f"Failed to set up Dask cluster: {e}", exc_info=True)
        sys.exit(1)

    # --- Main Processing Block ---
    try:
        start_time = time.time()
        logger.info("--- Starting Distributed Segmentation ---")
        logger.info(f"Input N5: {args.input_n5} [{args.n5_channel_path}]")
        logger.info(f"Output Zarr: {args.output_zarr}")
        logger.info(f"Model: {args.model_path}")
        logger.info(f"Block Size: {block_size_tuple}")

        mask = get_foreground_mask(args.input_n5, 'ch2/s4')
        # Call distributed_eval (no longer needs cluster_kwargs)
        # It now requires the client
        mask_ref, boxes = distributed_eval(
            input_source=args.input_n5, # Pass path
            n5_subpath=args.n5_channel_path,
            blocksize=block_size_tuple,
            write_path=args.output_zarr,
            model_path=args.model_path,
            client=client, # Pass the client
            mask=mask,
            eval_kwargs=eval_kwargs_dict,
            temporary_directory=args.temporary_dir,  
            # Pass model_kwargs if needed: model_kwargs={}
        )

        logger.info("Segmentation complete.")

        # Optionally save boxes if needed:
        # if boxes:
        #    box_file = os.path.splitext(args.output_zarr)[0] + "_boxes.pkl"
        #    try:
        #        with open(box_file, 'wb') as f:
        #            pickle.dump(boxes, f)
        #        logger.info(f"Saved {len(boxes)} merged bounding boxes to {box_file}")
        #    except Exception as e:
        #        logger.error(f"Failed to save bounding boxes: {e}")

        total_time = time.time() - start_time
        logger.info(f"--- Job finished in {total_time:.2f} seconds ---")

    except Exception as e:
        logger.error(f"An error occurred during distributed segmentation: {e}", exc_info=True)

    finally:
        # --- Shutdown Dask ---
        if cluster and client:
            shutdown_dask(cluster, client)
        else:
            logger.warning("Cluster/client not fully initialized, skipping shutdown.")


if __name__ == "__main__":
    main()