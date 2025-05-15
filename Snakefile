import os
from pathlib import Path
import random
import xml.etree.ElementTree as ET

# Configuration
configfile: "config.yaml"

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

# Collect input files
XML_FILE = config["input_xml"]
print(f"Found XML file: {XML_FILE}")

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

# Example helper function
def get_intermediate_xml(base_xml, stage_suffix):
    name, ext = os.path.splitext(base_xml)
    return f"{name}_{stage_suffix}{ext}"

rule all:
    input:
        config["input_dir"] + "/dataset_fused_features.csv"

rule reorient_sample:
    input:
        xml=XML_FILE
    output:
        xml=config["reorient_sample"]["output_xml"]
    conda:
        "dask-cellpose"
    script:
        "scripts/reorient_sample.py"

rule pairwise_stitching:
    input:
        xml=config["reorient_sample"]["output_xml"]
    output:
        done=config["input_dir"] + "/pairwise.done",
        stitched_xml=os.path.splitext(config["reorient_sample"]["output_xml"])[0]+'_pairwise.xml'
    params:
        outdir=lambda w, output: os.path.dirname(output.done),
        script_dir=config["bigstitcher_script_dir"],
        runtime=config["runtime"],
        n_nodes=config["n_nodes"],
        executors_per_node=config["cluster"]["executors_per_node"],
        cores_per_executor=config["cluster"]["cores_per_executor"],
        overhead_cores=config["cluster"]["overhead_cores_per_worker"],
        tasks_per_core=config["cluster"]["tasks_per_executor_core"],
        cores_driver=config["cluster"]["cores_driver"],
        gb_per_slot=config["cluster"]["gb_per_slot"],
        ram_per_core=config["cluster"]["ram_per_core"],
        project="beliveaulab",
        stitching_channel=config["stitching_channel"],
        min_correlation=config["min_correlation"]
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
        xml=os.path.splitext(config["input_xml"])[0]+'_pairwise.xml',
        pairwise=config["input_dir"] + "/pairwise.done"
    output:
        done=config["input_dir"] + "/solved.done",
        solved_xml=os.path.splitext(config["input_xml"])[0]+'_solved.xml'
    params:
        outdir=lambda w, output: os.path.dirname(output.done),
        script_dir=config["bigstitcher_script_dir"],
        runtime=config["runtime"],
        n_nodes=config["n_nodes"],
        executors_per_node=config["cluster"]["executors_per_node"],
        cores_per_executor=config["cluster"]["cores_per_executor"],
        overhead_cores=config["cluster"]["overhead_cores_per_worker"],
        tasks_per_core=config["cluster"]["tasks_per_executor_core"],
        cores_driver=config["cluster"]["cores_driver"],
        gb_per_slot=config["cluster"]["gb_per_slot"],
        ram_per_core=config["cluster"]["ram_per_core"],
        project="beliveaulab"
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
        xml=ancient(os.path.splitext(config["input_xml"])[0]+'_solved.xml'),
        solved=config["input_dir"] + "/solved.done"
    output:
        n5=directory(config["input_dir"] + "/dataset_fused.n5"),
        done=config["input_dir"] + "/affine_fusion.done"
    params:
        outdir=lambda w, output: os.path.dirname(output.n5),
        script_dir=config["bigstitcher_script_dir"],
        runtime=config["runtime"],
        n_nodes=config["n_nodes"],
        executors_per_node=config["cluster"]["executors_per_node"],
        cores_per_executor=config["cluster"]["cores_per_executor"],
        overhead_cores=config["cluster"]["overhead_cores_per_worker"],
        tasks_per_core=config["cluster"]["tasks_per_executor_core"],
        cores_driver=config["cluster"]["cores_driver"],
        gb_per_slot=config["cluster"]["gb_per_slot"],
        ram_per_core=config["cluster"]["ram_per_core"],
        project="beiveaulab",
        block_size=config["block_size"],
        data_type=config["data_type"],
        min_intensity=config["intensity"]["min"],
        max_intensity=config["intensity"]["max"],
        channels=config["channels"]
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
        done=config["input_dir"] + "/affine_fusion.done",
        n5=config["input_dir"] + "/dataset_fused.n5"
    output:
        zarr=directory(config["input_dir"] + "/dataset_fused" + config["segmentation"]["output_suffix"])
    params:
        outdir=lambda w, output: os.path.dirname(output.zarr),
        n5_channel_path=config["segmentation"].get("n5_channel_path", "ch0/s0"),
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
        "dask-cellpose"
    script:
        "scripts/segmentation.py"


rule feature_extraction:
    input:
        n5=config["input_dir"] + "/dataset_fused.n5",
        zarr=config["input_dir"] + "/dataset_fused" + config["segmentation"]["output_suffix"]
    output:
        csv=config["input_dir"] + "/dataset_fused_features.csv"
    params:
        outdir=lambda w, output: os.path.dirname(output.csv),
        log_dir=config["dask"].get("log_dir") + "/dask_worker_logs_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}/",
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
        "dask-cellpose"
    script:
        "scripts/feature_extraction.py"