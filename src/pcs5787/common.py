from pathlib import Path
import hashlib
import json
import os

ROOT = Path(os.environ.get("PCS5787_ROOT", Path(__file__).resolve().parents[2]))
CFG = json.loads((ROOT / "configs/pipeline.json").read_text())
SOURCE = Path(os.environ.get("PCS5787_SOURCE", ROOT / CFG["source_root"]))


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(2**20), b""):
            h.update(chunk)
    return h.hexdigest()


def save_json(path, value):
    path = ROOT / path
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, default=str) + "\n")
