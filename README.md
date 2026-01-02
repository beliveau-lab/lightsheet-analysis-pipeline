<h1> Analysis suite for light-sheet fluorescence microscopy images </h1>

<h2> Project overview </h2>

This project is a collection of tools for analyzing light-sheet fluorescence microscopy images.






<h2> Quickstart</h2>

1. Copy and edit a config:

```
cp config_template.yaml config.yaml
# edit config.yaml
```

2. Dry run (runs in terminal, make sure you are using snakemake >9.0!):

```
$ make dryrun
```

3. Run:

```
$ make run
```

<h2> Environment Setup and job configuration</h2>

To run this pipeline, you will first need to clone the repo:

`$ git clone https://github.com/beliveau-lab/lightsheet-analysis-pipeline`

We will assume that the raw dataset is stored on disk in .h5 format and has a corresponding .xml metadata file. 

Next, we will set up the config.yaml file. First specify the input .xml path and output directory:

**Input/Output Settings (from `config_template.yaml`)**

-   `input_xml`: The path to the XML metadata file (e.g., `/../../dataset.xml`). Make sure you have write permissions on this file! If not, you can make a copy and specify that .xml file instead.
-   `output_dir`: The folder to output resulting files.


Next, specify whether you want to reorient the sample. This currently applies the "SHIFT-Y" transformation from BigDataViewer.

#TODO: add custom transform matrix as an option

-   `reorient_sample`: 
    - `enabled` : Whether to reorient the sample in SHIFT-Y view. Values are True or False.


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
    -   `runtime`: Runtime request for the dask scheduler/workers.
    -   `log_dir`: Directory where Dask worker logs are saved.
    -   `dashboard_port`: Address for the Dask dashboard.

    -   `gpu_worker_config`: Settings for GPU workers
        -   `num_workers`, `processes`, `threads_per_worker`, `memory`, `cores`, `project`, `queue`, `resource_spec`
    -   `cpu_worker_config`: Settings for CPU workers
        -   `num_workers`, `processes`, `threads_per_worker`, `memory`, `cores`, `project`, `queue`, `resource_spec`

<h3> Segmentation (from `config.yaml`) </h3>

This step runs dask-distributed nuclear instance segmentation with Cellpose.

-   `segmentation`: Container for segmentation settings.
    -   `output_suffix`: Text added to the input filename to create the output segmentation filename (e.g., "_segmented_normalized.zarr").
    -   `block_size`: Processing chunk size [Z, Y, X] for the segmentation algorithm (e.g., [512,512,512]).
    -   `eval_kwargs`: Dict representing specific parameters passed directly to the segmentation model (e.g., Cellpose `eval` function), controlling thresholds, batch sizes, etc.
    -   `cellpose_model_path`: Path to the pre-trained Cellpose model used for segmentation.
    -   `segmentation_channel`: Channel on which to run the nuclei segmentation.
    -   `multiresolution`: (Experimental) whether to generate a multiresolution pyramid for the masks, helpful for visualization.

<h3> De-striping (from `config.yaml`) </h3>

-   `destripe`: Container for destripe settings.
    -   `output_suffix`: Text added to the input filename to create the output feature extraction filename (e.g., "_destriped.zarr").
    -   `channels`: List of channels to run destriping on (e.g. [0,1,2]).
    -   `n5_path_pattern`: A template defining how to access data for different channels within the N5/Zarr file (e.g., "ch{}/s0", where {} is replaced by the channel number).
    -   `block_size`: Output chunk size [Z, Y, X] for the resulting .zarr arrays.

<h3> Feature Extraction (from `config.yaml`) </h3>

After segmentation, this step measures various properties (features) of the segmented objects (e.g., size, shape, intensity in different channels).

-   `feature_extraction`: Container for feature extraction settings.
    -   `output_suffix`: Text added to the input filename to create the output feature extraction filename (e.g., "_features.csv").
    -   `channels`: List of channels from which to extract features for each segmented object.
    -   `n5_path_pattern`: A template defining how to access data for different channels within the N5/Zarr file (e.g., "ch{}/s0", where {} is replaced by the channel number).
    -   `batch_size`: Number of cells to process in a single batch (to control peak memory usage)

<h3> Postprocessing (from `config.yaml`) </h3>

Post-processing of the features.csv file.

    # TODO : decide what to do with this...


<h2> Running the Pipeline </h2>

<h3> Performing a full run on a fresh dataset (recommended usage) </h3>

IMPORTANT: It is highly recommended to perform a dryrun before submitting jobs to the cluster!! 

You can do this by running `$ make dryrun` from the repo root folder. You will need to use a snakemake version >9.0, which can be installed via conda or pip. Check the snakemake installation version with `$ snakemake --version`. 

If you are performing a full run on a fresh dataset, a dryrun is likely not necessary, but still good practice. If performing a partial run (for example, data correction + feature extraction only) see the below section "PERFORMING A PARTIAL RUN". 

After configuring the .yaml, you can begin execution of the Snakemake workflow with:

`$ make run`

Environments are managed automatically via `--use-conda` in the profile. If you prefer manual control, create the env from `workflow/envs/*.yml`.

To perform a dry run of the pipeline, you can modify the submit_snakemake.sh script to run snakemake with the -n flag.

- Spark logs: `{bigstitcher_script_dir}/logs/spark`. Dask logs: `dask.log_dir`. Snakemake logs: see `logs/` under the output directory.

<!-- <h3> Performing a partial run (not recommended)</h3>

NOTE: I recommend using the CLI and not Snakemake to perform different steps in isolation for development purposes!

In certain cases, it may be necessary to perform a partial run to avoid regenerating existing data. For example, you have generated the fused data and segmentation masks, and would like to test the effect of different image correction parameters on the resulting features.csv. In this case, we need to run the 'destripe' and 'fix_attenuation' rules in addition to 'feature_extraction' and the helper rules 'rechunk_to_blocks' and 'rechunk_to_planes'. 

First, we need to assess the current state of the existing data. Imagine we have previously performed a full run, and thus have all files from the input .xml to the resulting features.csv. 

Now, we would like to test a new parameter set for the fix_attenuation step. But, if we change the python script fix_attenuation.py and then perform a dryrun, we will  -->

<h2> FAQs / common issues </h2>

1. 