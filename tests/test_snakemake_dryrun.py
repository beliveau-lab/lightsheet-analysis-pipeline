import os
import subprocess
from pathlib import Path
import yaml


def test_snakemake_dryrun(tmp_path: Path):
    repo_root = Path(__file__).resolve().parents[1]

    # Prepare a minimal BigStitcher dir structure to satisfy validation
    bs = tmp_path / "BigStitcher-Spark" / "spark-janelia"
    bs.mkdir(parents=True)

    # Minimal XML and output dir
    xml = tmp_path / "dataset.xml"
    xml.write_text("<root></root>")
    out_dir = tmp_path / "out"

    cfg = {
        "input_xml": str(xml),
        "output_dir": str(out_dir),
        "bigstitcher_script_dir": str(bs.parent),
        "reorient_sample": {"enabled": False},
        "spark_job_timeout": 10,
        "spark_cluster": {
            "runtime": "140000",
            "n_nodes": 1,
            "executors_per_node": 1,
            "cores_per_executor": 1,
            "overhead_cores_per_worker": 0,
            "tasks_per_executor_core": 1,
            "cores_driver": 1,
            "gb_per_slot": 1,
            "ram_per_core": "1G",
            "project": "testproj",
        },
        "stitching": {"stitching_channel": 0, "min_correlation": 0.5},
        "fusion": {
            "channels": [0],
            "block_size": "64,64,64",
            "intensity": {"min": 0, "max": 255},
            "data_type": "UINT8",
        },
        "dask": {
            "runtime": "140000",
            "log_dir": str(tmp_path / "dask_logs"),
            "dashboard_port": ":8788",
            "gpu_worker_config": {
                "num_workers": 1,
                "processes": 1,
                "threads_per_worker": 1,
                "memory": "1G",
                "cores": 1,
                "project": "testproj",
                "queue": "testq",
                "resource_spec": "mfree=1G",
            },
            "cpu_worker_config": {
                "num_workers": 1,
                "processes": 1,
                "threads_per_worker": 1,
                "memory": "1G",
                "cores": 1,
                "project": "testproj",
                "queue": "testq",
                "resource_spec": "mfree=1G",
            },
        },
        "segmentation": {
            "output_suffix": "_masks.zarr",
            "block_size": [32, 32, 32],
            "eval_kwargs": "{}",
            "cellpose_model_path": str(tmp_path / "dummy_model"),
            "segmentation_channel": 0,
        },
        "rechunk_to_planes": {"output_suffix": "_planes.zarr", "channels": [0]},
        "destripe": {
            "output_suffix": "_destriped.zarr",
            "n5_path_pattern": "ch{}/s0",
            "channels": [0],
        },
        "rechunk_to_blocks": {
            "output_suffix": "_blocks.zarr",
            "channels": [0],
            "block_size": [64, 64, 64],
        },
        "feature_extraction": {
            "output_suffix": "_features.csv",
            "channels": [0],
            "n5_path_pattern": "ch{}/s0",
            "batch_size": 10,
        },
    }

    cfg_path = tmp_path / "config.yaml"
    with open(cfg_path, "w") as f:
        yaml.safe_dump(cfg, f)

    env = os.environ.copy()
    # Do not skip validation; we created the minimum structure
    env["OTLS_SKIP_VALIDATION"] = "0"

    cmd = [
        "snakemake",
        "-n",
        "-p",
        "--configfile",
        str(cfg_path),
        "--executor",
        "dryrun",
    ]

    # Dry-run should succeed and exit 0
    subprocess.check_call(cmd, cwd=repo_root, env=env)

