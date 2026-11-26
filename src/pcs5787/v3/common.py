import json
import hashlib
import pandas as pd
from datetime import datetime,timezone
from ..common import ROOT,sha256
from ..v2.common import atomic_json,atomic_frame,stamp,cutoff,digest,population,training_mask

P=json.loads((ROOT/'configs/v3/protocol.json').read_text())
DATA=ROOT/'data/processed/v3';OUT=ROOT/'reports/v3';CACHE=ROOT/'.scratch/v3/cache';MODELS=ROOT/'models/v3'

def note(stage,message):
    with (OUT/'STATUS.md').open('a') as f:f.write(f'\n## {datetime.now(timezone.utc).isoformat(timespec="seconds")} — {stage}\n\n{message}\n')

def code_identity():
    return {p.name:sha256(p) for p in sorted((ROOT/'src/pcs5787/v3').glob('*.py')) if p.name in ['data.py','models.py','engine.py','common.py']}

def frame_digest(frame):
    """Actual OOF rows matter when sequential updates extend the frozen archive."""
    cols=[c for c in ['region','time_local','name','y','p','train_last'] if c in frame]
    ordered=frame[cols].sort_values([c for c in ['time_local','region','name'] if c in cols])
    return hashlib.sha256(pd.util.hash_pandas_object(ordered,index=False).values.tobytes()).hexdigest()
