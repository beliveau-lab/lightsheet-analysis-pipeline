import os
from pathlib import Path
import random
import xml.etree.ElementTree as ET

# Configuration
configfile: "config.yaml"

# --- Globals / Path Definitions ---
INITIAL_XML = config["input_xml"]

PIPELINE_OUTPUT_DIR = Path(config["output_dir"])
PIPELINE_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

LOG_DIR = PIPELINE_OUTPUT_DIR / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)

# Helper to generate consistent XML names
def get_processed_xml_path(base_name, stage_suffix, reoriented_prefix_if_applicable=""):
    return PIPELINE_OUTPUT_DIR / f"{reoriented_prefix_if_applicable}{Path(base_name).stem}_{stage_suffix}.xml"

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

# Helper function to get channels from xml file
def get_channels(xml_file):
    # TODO: Implement XML channel detection
    # For now, return dummy channels
    return [0, 1, 2]  # Replace with actual channel detection from XML


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


# Add this helper function at the top with the others
def check_xml_stage(xml_file, stage):
    """Check if a particular processing stage is complete in the XML file"""
    try:
        tree = ET.parse(xml_file)
        root = tree.getroot()
        
        if stage == "pairwise":
            # Check for pairwise stitching results
            results = root.findall(".//PairwiseResult")
            has_results = len(results) > 0
            print(f"Found {len(results)} pairwise results in {xml_file}")
            return has_results
            
        elif stage == "solver":
            # Check for global optimization results with actual transformations
            registrations = root.findall(".//InterpolatedAffineModel3D")
            has_registrations = len(registrations) > 0
            print(f"Found {len(registrations)} view registrations in {xml_file}")
            return has_registrations
        
        return False
    except Exception as e:
        print(f"Error checking XML stage {stage}: {e}")
        return False


rule all:
    input:
        csv=FEATURES_CSV

rule reorient_sample:
    input:
        xml=INITIAL_XML
    output:
        xml=ORIENTED_XML_FOR_STITCHING
    conda:
        "otls-pipeline"
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
            shell("""
                set -x  # Print commands as they execute
                
                if ! mkdir -p {params.outdir}; then
                    echo "Failed to create output directory"
                    exit 1
                fi
                
                # Set environment variables for Spark configuration
                export SCRIPT_DIR={params.script_dir}
                export N_NODES={params.n_nodes}
                export N_EXECUTORS_PER_NODE={params.executors_per_node}
                export N_CORES_PER_EXECUTOR={params.cores_per_executor}
                export N_OVERHEAD_CORES_PER_WORKER={params.overhead_cores}
                export N_TASKS_PER_EXECUTOR_CORE={params.tasks_per_core}
                export N_CORES_DRIVER={params.cores_driver}
                export GB_PER_SLOT={params.gb_per_slot}
                export RUNTIME="{params.runtime}"
                export SGE_PROJECT="{params.project}"
                
                # Calculate total worker slots
                export N_CORES_PER_WORKER=$(( ({params.executors_per_node} * {params.cores_per_executor}) + {params.overhead_cores} ))
                
                # Run stitching
                if ! bash {params.script_dir}/spark-janelia/PairwiseStitching.sh \
                    {output.stitched_xml} \
                    {params.stitching_channel} \
                    {params.min_correlation} \
                    {params.outdir}; then
                    echo "PairwiseStitching.sh failed"
                    exit 1
                fi
                    
                # Find the log directory (most recent)
                BASE_LOG_DIR={params.script_dir}/logs/spark
                SPARK_LOG_DIR=$(find $BASE_LOG_DIR -maxdepth 1 -type d -name "2*" | sort -r | head -n 1)
                
                if [ -z "$SPARK_LOG_DIR" ]; then
                    echo "No log directories found"
                    exit 1
                fi
                
                echo "Looking for logs in: $SPARK_LOG_DIR"
                
                # Wait for job completion (indicated by shutdown log)
                TIMEOUT=43200  # 5 minutes timeout
                ELAPSED=0
                INTERVAL=30  # Check every 10 seconds
                
                while [ ! -f "$SPARK_LOG_DIR/logs/12-shutdown.log" ]; do
                    if [ $ELAPSED -ge $TIMEOUT ]; then
                        echo "Timeout waiting for job completion"
                        exit 1
                    fi
                    echo "Waiting for spark job to complete... ($ELAPSED seconds elapsed)"
                    sleep $INTERVAL
                    ELAPSED=$((ELAPSED + INTERVAL))
                done
                
                # Check for driver log
                DRIVER_LOG="$SPARK_LOG_DIR/logs/04-driver.log"
                if [ ! -f "$DRIVER_LOG" ]; then
                    echo "Driver log not found: $DRIVER_LOG"
                    echo "Available logs:"
                    ls -la "$SPARK_LOG_DIR/logs/"
                    exit 1
                fi
                
                # Check for successful completion
                if grep -q "Saving resulting XML" "$DRIVER_LOG"; then
                    touch {output.done}
                    echo "Stitching completed successfully"
                else
                    echo "Success message not found in driver log"
                    echo "Last 50 lines of driver log:"
                    tail -n 50 "$DRIVER_LOG"
                    exit 1
                fi
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
            shell("""
                set -x  # Print commands as they execute
                
                if ! mkdir -p {params.outdir}; then
                    echo "Failed to create output directory"
                    exit 1
                fi
                
                # Set environment variables for Spark configuration
                export N_NODES={params.n_nodes}
                export N_EXECUTORS_PER_NODE={params.executors_per_node}
                export N_CORES_PER_EXECUTOR={params.cores_per_executor}
                export N_OVERHEAD_CORES_PER_WORKER={params.overhead_cores}
                export N_TASKS_PER_EXECUTOR_CORE={params.tasks_per_core}
                export N_CORES_DRIVER={params.cores_driver}
                export GB_PER_SLOT={params.gb_per_slot}
                export RUNTIME="{params.runtime}"
                export SGE_PROJECT="{params.project}"
                export SCRIPT_DIR={params.script_dir}
                
                # Calculate total worker slots
                export N_CORES_PER_WORKER=$(( ({params.executors_per_node} * {params.cores_per_executor}) + {params.overhead_cores} ))
                
                # Run solver
                if ! bash {params.script_dir}/spark-janelia/Solver.sh \
                    -x {output.solved_xml} \
                    -s STITCHING \
                    --method TWO_ROUND_ITERATIVE; then
                    echo "Solver.sh failed"
                    exit 1
                fi
                    
                # Find the log directory (most recent)
                BASE_LOG_DIR={params.script_dir}/logs/spark
                SPARK_LOG_DIR=$(find $BASE_LOG_DIR -maxdepth 1 -type d -name "2*" | sort -r | head -n 1)
                
                if [ -z "$SPARK_LOG_DIR" ]; then
                    echo "No log directories found"
                    exit 1
                fi
                
                echo "Looking for logs in: $SPARK_LOG_DIR"
                
                # Wait for job completion (indicated by shutdown log)
                TIMEOUT=43200  # 5 minutes timeout
                ELAPSED=0
                INTERVAL=30  # Check every 10 seconds
                
                while [ ! -f "$SPARK_LOG_DIR/logs/12-shutdown.log" ]; do
                    if [ $ELAPSED -ge $TIMEOUT ]; then
                        echo "Timeout waiting for job completion"
                        exit 1
                    fi
                    echo "Waiting for spark job to complete... ($ELAPSED seconds elapsed)"
                    sleep $INTERVAL
                    ELAPSED=$((ELAPSED + INTERVAL))
                done
                
                # Check for driver log
                DRIVER_LOG="$SPARK_LOG_DIR/logs/04-driver.log"
                if [ ! -f "$DRIVER_LOG" ]; then
                    echo "Driver log not found: $DRIVER_LOG"
                    echo "Available logs:"
                    ls -la "$SPARK_LOG_DIR/logs/"
                    exit 1
                fi
                
                # Check for successful completion
                if grep -q "Saving resulting XML" "$DRIVER_LOG"; then
                    touch {output.done}
                    echo "Solver completed successfully"
                else
                    echo "Success message not found in driver log"
                    echo "Last 50 lines of driver log:"
                    tail -n 50 "$DRIVER_LOG"
                    exit 1
                fi
            """)

rule affine_fusion:
    input:
        xml=SOLVED_XML,
        solved=SOLVED_DONE
    output:
        n5=directory(AFFINE_FUSION_N5),
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
    shell:
        """
        set -x  # Print commands as they execute
        
        if ! mkdir -p {params.outdir}; then
            echo "Failed to create output directory"
            exit 1
        fi
        touch {output.done}
        
        # Set environment variables for Spark configuration
        export N_NODES={params.n_nodes}
        export N_EXECUTORS_PER_NODE={params.executors_per_node}
        export N_CORES_PER_EXECUTOR={params.cores_per_executor}
        export N_OVERHEAD_CORES_PER_WORKER={params.overhead_cores}
        export N_TASKS_PER_EXECUTOR_CORE={params.tasks_per_core}
        export N_CORES_DRIVER={params.cores_driver}
        export GB_PER_SLOT={params.gb_per_slot}
        export RUNTIME="{params.runtime}"
        export SGE_PROJECT="{params.project}"
        export SCRIPT_DIR={params.script_dir}
        # Calculate total worker slots
        export N_CORES_PER_WORKER=$(( ({params.executors_per_node} * {params.cores_per_executor}) + {params.overhead_cores} ))
        
        # Run fusion for each channel
        for channel in {params.channels}; do
            echo "Processing channel $channel"
            if ! bash {params.script_dir}/spark-janelia/AffineFusion.sh \
                -x {input.xml} \
                -o {output.n5} \
                -d /ch$channel/s0 \
                --channelId $channel \
                --blockSize {params.block_size} \
                --datatype {params.data_type} \
                --minIntensity {params.min_intensity} \
                --maxIntensity {params.max_intensity} \
                -s N5 \
                --multiRes \
                --preserveAnisotropy; then
                echo "AffineFusion.sh failed for channel $channel"
                exit 1
            fi
            
            # Wait for job completion and check logs
            BASE_LOG_DIR={params.script_dir}/logs/spark
            SPARK_LOG_DIR=$(find $BASE_LOG_DIR -maxdepth 1 -type d -name "2*" | sort -r | head -n 1)
            
            if [ -z "$SPARK_LOG_DIR" ]; then
                echo "No log directories found"
                exit 1
            fi
            
            echo "Looking for logs in: $SPARK_LOG_DIR"
            
            # Wait for job completion (indicated by shutdown log)
            TIMEOUT=43200  # 5 minutes timeout
            ELAPSED=0
            INTERVAL=30  # Check every 10 seconds
            
            while [ ! -f "$SPARK_LOG_DIR/logs/12-shutdown.log" ]; do
                if [ $ELAPSED -ge $TIMEOUT ]; then
                    echo "Timeout waiting for job completion"
                    exit 1
                fi
                echo "Waiting for spark job to complete... ($ELAPSED seconds elapsed)"
                sleep $INTERVAL
                ELAPSED=$((ELAPSED + INTERVAL))
            done
            
            # Check for driver log
            DRIVER_LOG="$SPARK_LOG_DIR/logs/04-driver.log"
            if [ ! -f "$DRIVER_LOG" ]; then
                echo "Driver log not found: $DRIVER_LOG"
                echo "Available logs:"
                ls -la "$SPARK_LOG_DIR/logs/"
                exit 1
            fi
            
            # Check for successful completion
            if ! grep -q "done, took" "$DRIVER_LOG"; then
                echo "Success message not found in driver log for channel $channel"
                echo "Last 50 lines of driver log:"
                tail -n 50 "$DRIVER_LOG"
                exit 1
            fi
            echo "Fusion completed successfully for channel $channel"
        done
        """ 

rule segmentation:
    input:
        # scheduler=SCHEDULER_FILE,
        done=AFFINE_FUSION_DONE,
        n5=AFFINE_FUSION_N5
    output:
        zarr=directory(SEGMENTED_ZARR)
    params:
        outdir=lambda w, output: os.path.dirname(output.zarr),
        n5_channel_path=config["segmentation"].get("n5_channel_path", "ch2/s0"),
        block_size=config["segmentation"]["block_size"],
        model_path=config["segmentation"]["cellpose_model_path"],
        eval_kwargs=config["segmentation"]["eval_kwargs"],
    resources:
        # Resources for the Snakemake job submission
        runtime=config["dask"].get("runtime", "1400000"), # Example runtime
        gpu_worker_memory=config["dask"].get("gpu_memory", "12G"),
        gpu_resource_spec=config["dask"].get("gpu_resource_spec", "gpgpu=1,cuda=1"),
        project=config["dask"]["gpu_project"], # SGE project
        n_gpu_workers=config["dask"]["num_gpu_workers"],
        gpu_cores=config["dask"]["gpu_cores"],
        gpu_queue=config["dask"]["gpu_queue"],
        gpu_processes=config["dask"]["gpu_processes"]
    conda:
        "otls-pipeline"
    script:
        "scripts/segmentation.py"


rule feature_extraction:
    input:
        n5=AFFINE_FUSION_N5,
        zarr=SEGMENTED_ZARR
    output:
        csv=FEATURES_CSV
    params:
        outdir=lambda w, output: os.path.dirname(output.csv),
        log_dir=LOG_DIR,
        n5_path_pattern=config["feature_extraction"].get("n5_path_pattern", "ch{}/s0"),
        channels=config["feature_extraction"].get("channels", "0"),
        batch_size=config["feature_extraction"].get("batch_size", 10000),
    resources:
        # Resources for the Snakemake job submission
        runtime=config["dask"].get("runtime", "1400000"), # Example runtime
        project=config["dask"]["cpu_project"], # SGE project
        queue=config["dask"]["cpu_queue"],
        num_workers=config["dask"]["num_cpu_workers"],
        mem_per_worker=config["dask"].get("cpu_memory", "60G"),
        cores_per_worker=config["dask"].get("cpu_cores", 1),
        resource_spec=config["dask"].get("cpu_resource_spec", "mfree=60G"),
        processes=config["dask"].get("cpu_processes", 2)
    conda:
        "otls-pipeline"
    script:
        "scripts/feature_extraction.py"