import tempfile
from pathlib import Path
import yaml
import subprocess


def write_yaml(tmpdir: Path, data: dict) -> Path:
    p = tmpdir / "cfg.yaml"
    with open(p, "w") as f:
        yaml.safe_dump(data, f)
    return p


def test_config_validation_minimal(tmp_path: Path):
    # Prepare minimal filesystem
    xml = tmp_path / "dataset.xml"
    xml.write_text("<root></root>")
    bs = tmp_path / "BigStitcher-Spark" / "spark-janelia"
    bs.mkdir(parents=True)

    cfg = {
        "input_xml": str(xml),
        "output_dir": str(tmp_path / "out"),
        "bigstitcher_script_dir": str(bs.parent),
    }

    cfgp = write_yaml(tmp_path, cfg)
    subprocess.check_call(["python", "scripts/config_validation.py", str(cfgp)])

