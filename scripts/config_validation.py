import sys
import os
from pathlib import Path
import yaml


REQUIRED_KEYS = [
    ("input_xml", str),
    ("output_dir", str),
    ("bigstitcher_script_dir", str),
]


def fail(msg: str) -> None:
    print(f"CONFIG ERROR: {msg}", file=sys.stderr)
    sys.exit(1)


def main(cfg_path: str) -> None:
    with open(cfg_path, "r") as f:
        cfg = yaml.safe_load(f)

    # Required keys and types
    for key, typ in REQUIRED_KEYS:
        if key not in cfg:
            fail(f"Missing required key '{key}'")
        if not isinstance(cfg[key], typ):
            fail(f"Key '{key}' must be of type {typ.__name__}")

    # Paths
    input_xml = Path(cfg["input_xml"])
    if not input_xml.exists():
        fail(f"input_xml path does not exist: {input_xml}")
    if not os.access(input_xml.parent, os.W_OK):
        fail(f"No write permission in directory of input_xml: {input_xml.parent}")

    bs_dir = Path(cfg["bigstitcher_script_dir"]) / "spark-janelia"
    if not bs_dir.exists():
        fail(f"BigStitcher-Spark scripts not found at: {bs_dir}")

    out_dir = Path(cfg["output_dir"]) if cfg.get("output_dir") else input_xml.parent
    out_dir.mkdir(parents=True, exist_ok=True)

    # Dask nested keys sanity (if present)
    dask_cfg = cfg.get("dask", {})
    for group in ("gpu_worker_config", "cpu_worker_config"):
        wc = dask_cfg.get(group, {})
        if wc:
            for rk in ("num_workers", "memory", "cores", "project"):
                if rk not in wc:
                    fail(f"dask.{group}.{rk} is required when {group} is provided")

    #print("Config validation passed.")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python scripts/config_validation.py <config.yaml>")
        sys.exit(2)
    main(sys.argv[1])


