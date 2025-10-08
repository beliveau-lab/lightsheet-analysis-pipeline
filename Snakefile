import os
from pathlib import Path
import random
import xml.etree.ElementTree as ET
from scripts.utils.xml_utils import get_channels, check_xml_stage

# Configuration
configfile: "config.yaml"

# Validate config early (non-fatal if validator is missing). Allow skipping via env var.
if os.environ.get("OTLS_SKIP_VALIDATION") != "1":
    try:
        shell("python scripts/config_validation.py config.yaml | cat")
    except Exception:
        print("Config validator not available; continuing without strict checks.")

# --- Globals / Path Definitions ---
INITIAL_XML = config["input_xml"]

PIPELINE_OUTPUT_DIR = Path(config["output_dir"])
PIPELINE_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

LOG_DIR = PIPELINE_OUTPUT_DIR / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)
LOG_DIR_STR = str(LOG_DIR)

# Helper to generate consistent XML names
def get_processed_xml_path(base_name, stage_suffix, reoriented_prefix_if_applicable=""):
    return PIPELINE_OUTPUT_DIR / f"{reoriented_prefix_if_applicable}{Path(base_name).stem}_{stage_suffix}.xml"

# Touch-mode helper: disable protected() when running with touch executor
TOUCH_MODE = os.environ.get("OTLS_TOUCH_MODE") == "1" or config.get("testing", {}).get("touch_mode", False)
def maybe_protected(target):
    return target if TOUCH_MODE else protected(target)

# Determine the base XML for stitching after potential reorientation
if config.get("reorient_sample", {}).get("enabled", False):
    REORIENTED_XML_NAME = Path(INITIAL_XML).stem + "_reoriented"
    ORIENTED_XML_FOR_STITCHING = get_processed_xml_path(INITIAL_XML, "reoriented")
else:
    REORIENTED_XML_NAME = Path(INITIAL_XML).stem # No reorientation, use original stem
    ORIENTED_XML_FOR_STITCHING = Path(INITIAL_XML) # Stitch the original XML

PAIRWISE_STITCHED_XML = get_processed_xml_path(ORIENTED_XML_FOR_STITCHING.name, "pairwise")
SOLVED_XML = get_processed_xml_path(PAIRWISE_STITCHED_XML.name, "solved")

# Define output file paths (examples)
REORIENT_DONE = PIPELINE_OUTPUT_DIR / "reorient.done"
PAIRWISE_DONE = PIPELINE_OUTPUT_DIR / "pairwise.done"
SOLVED_DONE = PIPELINE_OUTPUT_DIR / "solved.done"
AFFINE_FUSION_N5 = PIPELINE_OUTPUT_DIR / "dataset_fused.n5"
AFFINE_FUSION_DONE = PIPELINE_OUTPUT_DIR / "affine_fusion.done"
SEGMENTED_ZARR = PIPELINE_OUTPUT_DIR / f"dataset_fused{config['segmentation']['output_suffix']}"
FEATURES_CSV = PIPELINE_OUTPUT_DIR / f"dataset_fused{config['feature_extraction']['output_suffix']}"
DESTRIPE_ZARR = PIPELINE_OUTPUT_DIR / f"dataset_fused{config['destripe']['output_suffix']}"
CORRECTED_ZARR = PIPELINE_OUTPUT_DIR / f"dataset_fused{config['fix_attenuation']['output_suffix']}"
RECHUNKED_BLOCKS_ZARR = PIPELINE_OUTPUT_DIR / f"dataset_fused{config['rechunk_to_blocks']['output_suffix']}"
RECHUNKED_PLANES_ZARR = PIPELINE_OUTPUT_DIR / f"dataset_fused{config['rechunk_to_planes']['output_suffix']}"

# Helper function to set common Spark environment variables
def get_spark_env_exports(params):
    # Construct the export commands string directly using params
    # Note: Ensure values like ram_per_core are properly quoted if they contain spaces/special chars
    exports = f"""
        export RUNTIME=\"{params.runtime}\"
        export N_NODES={params.n_nodes}
        export N_EXECUTORS_PER_NODE={params.executors_per_node}
        export N_CORES_PER_EXECUTOR={params.cores_per_executor}
        export N_OVERHEAD_CORES_PER_WORKER={params.overhead_cores}
        export N_TASKS_PER_EXECUTOR_CORE={params.tasks_per_core}
        export N_CORES_DRIVER={params.cores_driver}
        export GB_PER_SLOT={params.gb_per_slot}
        export RAM_PER_CORE=\"{params.ram_per_core}\"
        export SGE_PROJECT=\"{params.project}\"
        export SPARK_JANELIA_ARGS=\"--consolidate_logs -A {params.project}\"
        export WORKER_SLOTS_ARGS=\"-l mfree={params.ram_per_core}\"
    """
    return exports.strip() # Remove leading/trailing whitespace


## XML-related helpers are provided by scripts/utils/xml_utils.py


rule all:
    input:
        csv=FEATURES_CSV

if config.get("reorient_sample", {}).get("enabled", False):
    rule reorient_sample:
        input:
            xml=INITIAL_XML
        output:
            xml=ORIENTED_XML_FOR_STITCHING
        conda:
            "workflow/envs/otls-pipeline.yml"
        script:
            "scripts/reorient_sample.py"

rule pairwise_stitching:
    input:
        xml=ORIENTED_XML_FOR_STITCHING
    output:
        done=PAIRWISE_DONE,
        stitched_xml=PAIRWISE_STITCHED_XML
    params:
        outdir=lambda w, output: os.path.dirname(output.done),
        script_dir=config["bigstitcher_script_dir"],
        runtime=config["spark_cluster"]["runtime"],
        n_nodes=config["spark_cluster"]["n_nodes"],
        executors_per_node=config["spark_cluster"]["executors_per_node"],
        cores_per_executor=config["spark_cluster"]["cores_per_executor"],
        overhead_cores=config["spark_cluster"]["overhead_cores_per_worker"],
        tasks_per_core=config["spark_cluster"]["tasks_per_executor_core"],
        cores_driver=config["spark_cluster"]["cores_driver"],
        gb_per_slot=config["spark_cluster"]["gb_per_slot"],
        ram_per_core=config["spark_cluster"]["ram_per_core"],
        project=config["spark_cluster"]["project"],
        stitching_channel=config["stitching"]["stitching_channel"],
        min_correlation=config["stitching"]["min_correlation"]
    run:
        shell("cp {input.xml} {output.stitched_xml}")
        print(f"Checking for existing pairwise results in {output.stitched_xml}")
        if check_xml_stage(output.stitched_xml, "pairwise"):
            print("Pairwise stitching results found in XML, skipping step")
            shell("touch {output.done}")
        else:
            print("No pairwise results found, running stitching")
            exports = get_spark_env_exports(params)
            shell(f"""
                set -euo pipefail
                mkdir -p {params.outdir}

                export SCRIPT_DIR={params.script_dir}
                {exports}
                export N_CORES_PER_WORKER=$(( ({params.executors_per_node} * {params.cores_per_executor}) + {params.overhead_cores} ))

                bash {params.script_dir}/spark-janelia/PairwiseStitching.sh "{output.stitched_xml}" "{params.stitching_channel}" "{params.min_correlation}" "{params.outdir}" && python scripts/check_spark_job.py --base-log-dir "{params.script_dir}/logs/spark" --job-name "Pairwise Stitching" --timeout {config['spark_job_timeout']} --interval 30 --success-pattern "Saving resulting XML" && touch "{output.done}"
            """)

rule solver:
    input:
        xml=PAIRWISE_STITCHED_XML,
        pairwise=PAIRWISE_DONE
    output:
        done=SOLVED_DONE,
        solved_xml=SOLVED_XML
    params:
        outdir=lambda w, output: os.path.dirname(output.done),
        script_dir=config["bigstitcher_script_dir"],
        runtime=config["spark_cluster"]["runtime"],
        n_nodes=config["spark_cluster"]["n_nodes"],
        executors_per_node=config["spark_cluster"]["executors_per_node"],
        cores_per_executor=config["spark_cluster"]["cores_per_executor"],
        overhead_cores=config["spark_cluster"]["overhead_cores_per_worker"],
        tasks_per_core=config["spark_cluster"]["tasks_per_executor_core"],
        cores_driver=config["spark_cluster"]["cores_driver"],
        gb_per_slot=config["spark_cluster"]["gb_per_slot"],
        ram_per_core=config["spark_cluster"]["ram_per_core"],
        project=config["spark_cluster"]["project"]
    run:
        shell("cp {input.xml} {output.solved_xml}")
        print(f"Checking for existing solver results in {output.solved_xml}")
        if check_xml_stage(output.solved_xml, "solver"):
            print("Solver results found in XML, skipping step")
            shell("touch {output.done}")
        else:
            print("No solver results found, running solver")
            exports = get_spark_env_exports(params)
            shell(f"""
                set -euo pipefail
                mkdir -p {params.outdir}

                export SCRIPT_DIR={params.script_dir}
                {exports}
                export N_CORES_PER_WORKER=$(( ({params.executors_per_node} * {params.cores_per_executor}) + {params.overhead_cores} ))

                bash {params.script_dir}/spark-janelia/Solver.sh -x "{output.solved_xml}" -s STITCHING --method TWO_ROUND_ITERATIVE && python scripts/check_spark_job.py --base-log-dir "{params.script_dir}/logs/spark" --job-name "BigStitcher Solver" --timeout {config['spark_job_timeout']} --interval 30 --success-pattern "Saving resulting XML" && touch "{output.done}"
            """)

rule affine_fusion:
    input:
        xml=SOLVED_XML,
        solved=SOLVED_DONE
    output:
        n5=maybe_protected(directory(AFFINE_FUSION_N5)),
        done=AFFINE_FUSION_DONE
    params:
        outdir=lambda w, output: os.path.dirname(output.n5),
        script_dir=config["bigstitcher_script_dir"],
        runtime=config["spark_cluster"]["runtime"],
        n_nodes=config["spark_cluster"]["n_nodes"],
        executors_per_node=config["spark_cluster"]["executors_per_node"],
        cores_per_executor=config["spark_cluster"]["cores_per_executor"],
        overhead_cores=config["spark_cluster"]["overhead_cores_per_worker"],
        tasks_per_core=config["spark_cluster"]["tasks_per_executor_core"],
        cores_driver=config["spark_cluster"]["cores_driver"],
        gb_per_slot=config["spark_cluster"]["gb_per_slot"],
        ram_per_core=config["spark_cluster"]["ram_per_core"],
        project=config["spark_cluster"]["project"],
        block_size=config["fusion"]["block_size"],
        data_type=config["fusion"]["data_type"],
        min_intensity=config["fusion"]["intensity"]["min"],
        max_intensity=config["fusion"]["intensity"]["max"],
        channels=config["fusion"]["channels"]
    run:
        exports = get_spark_env_exports(params)
        # Ensure output dir exists once
        shell(f"mkdir -p {params.outdir}")
        # Common exports for each channel invocation
        for ch in params.channels:
            shell(f"""
                set -euo pipefail
                export SCRIPT_DIR={params.script_dir}
                {exports}
                export N_CORES_PER_WORKER=$(( ({params.executors_per_node} * {params.cores_per_executor}) + {params.overhead_cores} ))
                bash {params.script_dir}/spark-janelia/AffineFusion.sh -x "{input.xml}" -o "{output.n5}" -d "/ch{ch}/s0" --channelId "{ch}" --blockSize "{params.block_size}" --datatype "{params.data_type}" --minIntensity "{params.min_intensity}" --maxIntensity "{params.max_intensity}" -s N5 --multiRes --preserveAnisotropy && python scripts/check_spark_job.py --base-log-dir "{params.script_dir}/logs/spark" --job-name "Affine Fusion (channel {ch})" --timeout {config['spark_job_timeout']} --interval 30 --success-pattern "done, took"
            """)
        shell(f"touch {output.done}")

rule segmentation:
    input:
        # scheduler=SCHEDULER_FILE,
        done=AFFINE_FUSION_DONE,
        n5=AFFINE_FUSION_N5
    output:
        zarr=maybe_protected(directory(SEGMENTED_ZARR))
    params:
        outdir=lambda w, output: os.path.dirname(output.zarr),
        segmentation_channel=config["segmentation"]["segmentation_channel"],
        block_size=config["segmentation"]["block_size"],
        model_path=config["segmentation"]["cellpose_model_path"],
        eval_kwargs=config["segmentation"]["eval_kwargs"],
        log_dir=LOG_DIR_STR,
    resources:
        # Resources for the Snakemake job submission
        runtime=config["dask"].get("runtime", "1400000"), # Example runtime
        gpu_worker_memory=config["dask"]["gpu_worker_config"]["memory"],
        gpu_resource_spec=config["dask"]["gpu_worker_config"]["resource_spec"],
        project=config["dask"]["gpu_worker_config"]["project"], # SGE project
        n_gpu_workers=config["dask"]["gpu_worker_config"]["num_workers"],
        gpu_cores=config["dask"]["gpu_worker_config"]["cores"],
        gpu_queue=config["dask"]["gpu_worker_config"]["queue"],
        gpu_processes=config["dask"]["gpu_worker_config"]["processes"],
        dashboard_port=config["dask"]["dashboard_port"]
    conda:
        "workflow/envs/otls-pipeline.yml"
    script:
        "scripts/segmentation.py"

rule rechunk_to_planes:
    input:
        n5=AFFINE_FUSION_N5
    output:
        zarr=directory(RECHUNKED_PLANES_ZARR)
    params:
        outdir=lambda w, output: os.path.dirname(output.zarr),
        channels=config["rechunk_to_planes"]["channels"],
        log_dir=LOG_DIR_STR,
    resources:
        # Resources for the Snakemake job submission
        runtime=config["dask"].get("runtime", "1400000"),
        worker_memory=config["dask"]["cpu_worker_config"]["memory"],
        resource_spec=config["dask"]["cpu_worker_config"]["resource_spec"],
        project=config["dask"]["cpu_worker_config"]["project"], # SGE project
        n_workers=config["dask"]["cpu_worker_config"]["num_workers"],
        cores=config["dask"]["cpu_worker_config"]["cores"],
        queue=config["dask"]["cpu_worker_config"]["queue"],
        processes=config["dask"]["cpu_worker_config"]["processes"],
        dashboard_port=config["dask"]["dashboard_port"]
    conda:
        "workflow/envs/otls-pipeline.yml"
    script:
        "scripts/rechunk_to_planes.py"

rule destripe:
    input:
        zarr=RECHUNKED_PLANES_ZARR
    output:
        zarr=directory(DESTRIPE_ZARR)
    params:
        channels=config["destripe"]["channels"],
        outdir=lambda w, output: os.path.dirname(output.zarr),
        log_dir=LOG_DIR_STR,

    resources:
        # Resources for the Snakemake job submission
        worker_memory=config["dask"]["gpu_worker_config"]["memory"],
        resource_spec=config["dask"]["gpu_worker_config"]["resource_spec"],
        project=config["dask"]["gpu_worker_config"]["project"], # SGE project
        n_workers=config["dask"]["gpu_worker_config"]["num_workers"],
        cores=config["dask"]["gpu_worker_config"]["cores"],
        queue=config["dask"]["gpu_worker_config"]["queue"],
        processes=config["dask"]["gpu_worker_config"]["processes"],
        
        # Common params
        runtime=config["dask"].get("runtime", "1400000"),
        dashboard_port=config["dask"]["dashboard_port"],
        dask_resources="GPU=1"

    conda:
        "workflow/envs/otls-pipeline.yml"
    script:
        "scripts/destripe.py"

rule attenuation_fix:
    input:
        zarr=DESTRIPE_ZARR
    output:
        zarr=maybe_protected(directory(CORRECTED_ZARR))
    params:
        channels=config["fix_attenuation"]["channels"],
        outdir=lambda w, output: os.path.dirname(output.zarr),
        log_dir=LOG_DIR_STR,

    resources:
        # Resources for the Snakemake job submission
        worker_memory=config["dask"]["cpu_worker_config"]["memory"],
        resource_spec=config["dask"]["cpu_worker_config"]["resource_spec"],
        project=config["dask"]["cpu_worker_config"]["project"], # SGE project
        n_workers=config["dask"]["cpu_worker_config"]["num_workers"],
        cores=config["dask"]["cpu_worker_config"]["cores"],
        queue=config["dask"]["cpu_worker_config"]["queue"],
        processes=config["dask"]["cpu_worker_config"]["processes"],

        # Common params
        runtime=config["dask"].get("runtime", "1400000"),
        dashboard_port=config["dask"]["dashboard_port"],

    conda:
        "workflow/envs/otls-pipeline.yml"
    script:
        "scripts/fix_attenuation.py"

rule rechunk_to_blocks:
    input:
        zarr=CORRECTED_ZARR
    output:
        zarr=maybe_protected(directory(RECHUNKED_BLOCKS_ZARR))
    params:
        outdir=lambda w, output: os.path.dirname(output.zarr),
        block_size=config["rechunk_to_blocks"]["block_size"],
        channels=config["rechunk_to_blocks"]["channels"],
        log_dir=LOG_DIR_STR,
        zarr_planes=str(RECHUNKED_PLANES_ZARR),
        zarr_destriped=str(DESTRIPE_ZARR),
    resources:
        # Resources for the Snakemake job submission
        runtime=config["dask"].get("runtime", "1400000"), # Example runtime
        project=config["dask"]["cpu_worker_config"]["project"], # SGE project
        queue=config["dask"]["cpu_worker_config"]["queue"],
        n_workers=config["dask"]["cpu_worker_config"]["num_workers"],
        worker_memory=config["dask"]["cpu_worker_config"]["memory"],
        cores=config["dask"]["cpu_worker_config"]["cores"],
        resource_spec=config["dask"]["cpu_worker_config"]["resource_spec"],
        processes=config["dask"]["cpu_worker_config"]["processes"],
        dashboard_port=config["dask"]["dashboard_port"]
    conda:
        "workflow/envs/otls-pipeline.yml"
    script:
        "scripts/rechunk_to_blocks.py"

rule feature_extraction:
    input:
        img_zarr=RECHUNKED_BLOCKS_ZARR,
        mask_zarr=SEGMENTED_ZARR
    output:
        csv=maybe_protected(FEATURES_CSV)
    params:
        outdir=lambda w, output: os.path.dirname(output.csv),
        log_dir=LOG_DIR_STR,
        n5_path_pattern=config["feature_extraction"].get("n5_path_pattern", "ch{}/s0"),
        channels=config["feature_extraction"].get("channels", "0"),
        generate_embeddings=config["feature_extraction"].get("generate_embeddings", False)
    resources:
        # Resources for the Snakemake job submission
        runtime=config["dask"].get("runtime", "1400000"), # Example runtime
        project=config["dask"]["cpu_worker_config"]["project"], # SGE project
        queue=config["dask"]["cpu_worker_config"]["queue"],
        num_workers=config["dask"]["cpu_worker_config"]["num_workers"],
        mem_per_worker=config["dask"]["cpu_worker_config"]["memory"],
        cores_per_worker=config["dask"]["cpu_worker_config"]["cores"],
        resource_spec=config["dask"]["cpu_worker_config"]["resource_spec"],
        processes=config["dask"]["cpu_worker_config"]["processes"],
        dashboard_port=config["dask"]["dashboard_port"]
    conda:
        "workflow/envs/otls-pipeline-cp3.yml"
    script:
        "scripts/feature_extraction.py"