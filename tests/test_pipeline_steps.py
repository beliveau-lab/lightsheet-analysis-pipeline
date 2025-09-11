import os
import subprocess
from pathlib import Path
import yaml
import pytest


def _write_cfg(tmp_path: Path, base_cfg: dict) -> Path:
    cfg_path = tmp_path / "config.yaml"
    with open(cfg_path, "w") as f:
        yaml.safe_dump(base_cfg, f)
    return cfg_path


def _make_dummy_bigstitcher(bs_root: Path) -> None:
    """Create minimal BigStitcher-Spark stub scripts and logs for tests.

    bs_root: path to BigStitcher-Spark (contains spark-janelia/)
    """
    spark_dir = bs_root / "spark-janelia"
    spark_dir.mkdir(parents=True, exist_ok=True)

    # Common log writer used by all stubs
    common_stub = """#!/bin/bash
set -euo pipefail
BASE_LOG_DIR="${SCRIPT_DIR:-$(cd "$(dirname "$0")/.." && pwd)}/logs/spark"
RUN_DIR="$BASE_LOG_DIR/20250101-000000"
mkdir -p "$RUN_DIR/logs"
echo "Saving resulting XML" > "$RUN_DIR/logs/04-driver.log"
echo "done, took 00:00:01" >> "$RUN_DIR/logs/04-driver.log"
touch "$RUN_DIR/logs/12-shutdown.log"
"""

    # Pairwise and Solver stubs: just write logs and exit
    for name in ("PairwiseStitching.sh", "Solver.sh"):
        (spark_dir / name).write_text(common_stub + "\nexit 0\n")
        (spark_dir / name).chmod(0o755)

    # AffineFusion stub: additionally create the expected N5 output dir
    affine_stub = common_stub + """
# Parse args to find -o (output) and -d (dest)
OUTPUT=""
DEST=""
while [ $# -gt 0 ]; do
  case "$1" in
    -o)
      shift; OUTPUT="$1" ;;
    -d)
      shift; DEST="$1" ;;
    --channelId)
      shift ;; # ignore value
  esac
  shift || true
done

if [ -n "$OUTPUT" ]; then
  mkdir -p "$OUTPUT"
  if [ -n "$DEST" ]; then
    # remove leading slash from DEST if present
    CLEAN_DEST="${DEST#/}"
    mkdir -p "$OUTPUT/$CLEAN_DEST"
  fi
fi
exit 0
"""
    p = spark_dir / "AffineFusion.sh"
    p.write_text(affine_stub)
    p.chmod(0o755)


def _base_minimal_cfg(tmp_path: Path) -> dict:
    xml = tmp_path / "dataset.xml"
    xml.write_text("<root></root>")
    bs_root = tmp_path / "BigStitcher-Spark"
    _make_dummy_bigstitcher(bs_root)
    out_dir = tmp_path / "out"
    return {
        "input_xml": str(xml),
        "output_dir": str(out_dir),
        "bigstitcher_script_dir": str(bs_root),
        "reorient_sample": {"enabled": False},
        "spark_job_timeout": 5,
        "spark_cluster": {
            "runtime": "3000",
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
            "runtime": "3000",
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


def _run(rule: str, cfg_path: Path, repo_root: Path, marks_env: dict | None = None):
    env = os.environ.copy()
    env["OTLS_SKIP_VALIDATION"] = "0"
    if marks_env:
        env.update(marks_env)
    cmd = [
        "snakemake",
        "-p",
        "--configfile",
        str(cfg_path),
        "--profile",
        str(repo_root / "profiles" / "sge"),
        "--until",
        rule,
    ]
    subprocess.check_call(cmd, cwd=repo_root, env=env)


@pytest.mark.spark
def test_pairwise_solver_affine(tmp_path: Path):
    repo_root = Path(__file__).resolve().parents[1]
    cfg = _base_minimal_cfg(tmp_path)
    cfg_path = _write_cfg(tmp_path, cfg)
    _run("affine_fusion", cfg_path, repo_root)


@pytest.mark.dask
def test_rechunk_to_planes(tmp_path: Path):
    repo_root = Path(__file__).resolve().parents[1]
    cfg = _base_minimal_cfg(tmp_path)
    cfg_path = _write_cfg(tmp_path, cfg)
    _run("rechunk_to_planes", cfg_path, repo_root)


@pytest.mark.gpu
def test_destripe(tmp_path: Path):
    repo_root = Path(__file__).resolve().parents[1]
    cfg = _base_minimal_cfg(tmp_path)
    cfg_path = _write_cfg(tmp_path, cfg)
    _run("destripe", cfg_path, repo_root)


@pytest.mark.gpu
def test_segmentation(tmp_path: Path):
    repo_root = Path(__file__).resolve().parents[1]
    cfg = _base_minimal_cfg(tmp_path)
    cfg_path = _write_cfg(tmp_path, cfg)
    _run("segmentation", cfg_path, repo_root)


@pytest.mark.dask
def test_rechunk_to_blocks_and_features(tmp_path: Path):
    repo_root = Path(__file__).resolve().parents[1]
    cfg = _base_minimal_cfg(tmp_path)
    cfg_path = _write_cfg(tmp_path, cfg)
    _run("feature_extraction", cfg_path, repo_root)


