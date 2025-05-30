<h1> Analysis suite for light-sheet fluorescence microscopy images </h1>

<h2> Project overview </h2>

This project is a collection of tools for analyzing light-sheet fluorescence microscopy images.






<h2> Environment Setup and job configuration</h2>

To run this pipeline, you will first need to clone the repo:

`$ git clone https://github.com/beliveau-lab/lightsheet-analysis-pipeline`

We will assume that the raw dataset is stored on disk in .h5 format and has a corresponding .xml metadata file. 

Next, we will set up the config.yaml file. First specify the input directory and .xml path:

**Input/Output Settings (from `config.yaml`)**

-   `input_dir`: The folder containing the raw image data (e.g., `.h5` files).
-   `input_xml`: The path to the metadata file (e.g., `.xml`).

Next, specify whether you want to reorient the sample. This currently applies the "SHIFT-Y" transformation from BigDataViewer.

#TODO: add custom transform matrix as an option

-   `reorient_sample`: 
    - `enabled` : Whether to reorient the sample in SHIFT-Y view.


Next, we need to specify the BigStitcher-Spark installation directory:

**BigStitcher-Spark Settings (from `config.yaml`)**

-   `bigstitcher_script_dir`: The location of the BigStitcher-Spark scripts needed to run the alignment process.

The major steps of the alignment process are as follows:

1. Pairwise stitching
2. Global optimization
3. Affine fusion

Each step relies on an Apache Spark cluster for parallel processing. While you can specify a cluster configuration per step, I find that the default setup works well for all steps. Source documentation can be found at:
- https://github.com/JaneliaSciComp/BigStitcher-Spark/blob/main/README.md

**Spark Cluster Configuration (from `config.yaml`)**

-   `runtime`: Maximum time allocated for the alignment job (Format: HH:MM).
-   `n_nodes`: Number of nodes to use in the cluster (1 node = 1 job).
-   `spark_log_base_dir`: The directory where logs (records of the process) from Spark will be saved.
-   `spark_job_timeout`: How long (in seconds) the system should wait for Spark job logs before timing out.
-   `cluster`: Specific settings for how Spark distributes the work:
    -   `executors_per_node`: Number of parallel processes (executors) to run on each node.
    -   `cores_per_executor`: Number of processor cores assigned to each executor.
    -   `overhead_cores_per_worker`: Extra cores reserved on each node for system tasks.
    -   `tasks_per_executor_core`: How many small tasks each core within an executor can handle concurrently.
    -   `cores_driver`: Number of cores dedicated to the main coordinating process (driver).
    -   `gb_per_slot`: Amount of memory (in gigabytes) allocated per processing slot. *Note: This might be related to older cluster managers.*
    -   `ram_per_core`: Amount of memory allocated for each processor core (e.g., "12G").
    -   `project`: The SGE project to submit jobs to.
    -   `queue`: The specific queue to submit the job to (optional).

**Stitching Settings (from `config.yaml`)**

This step finds the spatial relationships between adjacent image tiles.

-   `stitching_channel`: The image channel (e.g., nuclear stain) used to find overlaps between tiles. Channel numbers usually start from 0.
-   `min_correlation`: A threshold (0 to 1) indicating how similar overlapping regions must be to be considered a match. Higher values mean stricter matching.

**Global Optimization**

This step refines the positions of all tiles simultaneously based on the stitching results to create the most accurate overall alignment. 

- TODO: Implement error threshold parameters

**Fusion (from `config.yaml`)**

This step combines the aligned tiles into a single output image file.

-   `channels`: A list of the image channels to include in the final fused image (e.g., `[0, 1, 2]`).
-   `block_size`: The size of the data chunks (in pixels: X, Y, Z) used when writing the output file (e.g., "512,512,512"). This affects performance and compatibility with downstream tools.
-   `intensity`: Defines the range of pixel brightness values in the output.
    -   `min`: The minimum intensity value.
    -   `max`: The maximum intensity value (e.g., 65535 for 16-bit images).
-   `data_type`: The numerical format for storing pixel brightness (e.g., `UINT16` for 16-bit unsigned integers).

The fused output is saved in the input directory as a multichannel, multiresolution .n5 container.

<h3> Distributed Processing (Dask Configuration from `config.yaml`) </h3>

For tasks like segmentation and feature extraction that can be parallelized, the Dask library is used to manage computation across multiple workers (potentially using GPUs for acceleration).

-   `dask`: Container for Dask settings.
    -   `log_dir`: Directory where Dask worker logs are saved.
    -   `runtime`: Maximum runtime for Dask worker jobs.
    -   **GPU Worker Settings:** Controls workers utilizing Graphics Processing Units (GPUs).
        -   `gpu_project`: SGE project for GPU jobs.
        -   `gpu_queue`: SGE queue for GPU jobs.
        -   `num_gpu_workers`: Number of GPU workers to start.
        -   `gpu_cores`: CPU cores allocated per GPU worker.
        -   `gpu_memory`: System memory (RAM) allocated per GPU worker (e.g., "60G").
        -   `gpu_resource_spec`: Specific hardware requirements for GPU workers (e.g., requesting 1 GPU, CUDA compatibility, minimum free CPU memory).
        -   `gpu_processes`: Number of Python processes to run within each GPU worker.
    -   **CPU Worker Settings:** Controls workers utilizing standard Central Processing Units (CPUs).
        -   `cpu_project`: SGE project for CPU jobs.
        -   `cpu_queue`: SGE queue for CPU jobs.
        -   `num_cpu_workers`: Number of CPU worker processes to start.
        -   `cpu_cores`: CPU cores allocated per CPU worker.
        -   `cpu_memory`: System memory (RAM) allocated per CPU worker (e.g., "60G").
        -   `cpu_resource_spec`: Specific hardware requirements for CPU workers (e.g., minimum free system memory).
        -   `cpu_processes`: Number of Python processes to run within each CPU worker.

<h3> Segmentation (from `config.yaml`) </h3>

This step runs dask-distributed nuclear instance segmentation with Cellpose.

-   `segmentation`: Container for segmentation settings.
    -   `output_suffix`: Text added to the input filename to create the output segmentation filename (e.g., "_segmented_normalized.zarr").
    -   `block_size`: Processing chunk size (X, Y, Z) for the segmentation algorithm.
    -   `eval_kwargs`: Specific parameters passed directly to the segmentation model (e.g., Cellpose `eval` function), controlling thresholds, batch sizes, etc.
    -   `cellpose_model_path`: Path to the pre-trained Cellpose model used for segmentation.
    -   `n5_channel_path`: Specifies the path *within* the fused N5/Zarr file to the specific channel data that will be used as input for segmentation (e.g., "ch2/s0" means channel 2, scale 0).

<h3> Feature Extraction (from `config.yaml`) </h3>

After segmentation, this step measures various properties (features) of the segmented objects (e.g., size, shape, intensity in different channels).

-   `feature_extraction`: Container for feature extraction settings.
    -   `output_suffix`: Text added to the input filename to create the output feature extraction filename (e.g., "_features.csv").
    -   `channels`: List of channels from which to extract features for each segmented object.
    -   `n5_path_pattern`: A template defining how to access data for different channels within the N5/Zarr file (e.g., "ch{}/s0", where {} is replaced by the channel number).
    -   `batch_size`: Number of cells to process in a single batch (to control peak memory usage)


<h2> Running the Pipeline </h2>

After configuring the .yaml, you can begin execution of the Snakemake workflow with:

`$ qsub submit_snakemake.sh`

You may need to create a conda environment with snakemake installed, and point to it in the submit_snakemake.sh script.

To perform a dry run of the pipeline, you can modify the submit_snakemake.sh script to run snakemake with the -n flag.





- TODO: discuss spark and dask dashboard access
- TODO: add an environment.yaml file to the repo
- TODO: add dry run command line option to submit_snakemake.sh

