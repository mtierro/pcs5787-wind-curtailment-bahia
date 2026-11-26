"""Own-package provenance; never write historical artifacts."""
import argparse
from datetime import datetime,timezone
import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform
import subprocess
import numpy as np
import pandas as pd

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
DEV=ROOT/"reports/h2_feature_selection_dev"
PROTECTED=["src","configs","data","models",".scratch","reports","notebooks","referencias_codex","docs","slides","paper",
           "deliverables","entrega_grupo","exports","importado_grupo","metadata","scripts","tests"]


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda:f.read(8*1024*1024),b''):h.update(chunk)
    return h.hexdigest()


def digest(obj):return hashlib.sha256(json.dumps(obj,sort_keys=True,default=str).encode()).hexdigest()


def json_default(value):
    if isinstance(value,np.generic):return value.item()
    if isinstance(value,np.ndarray):return value.tolist()
    if isinstance(value,(Path,pd.Timestamp)):return str(value)
    raise TypeError(type(value).__name__)


def write_json(path,obj):
    path=Path(path);assert path.resolve().is_relative_to(HERE)
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(obj,ensure_ascii=False,indent=2,default=json_default)+'\n')


def save(frame,name):
    p=HERE/name;assert p.resolve().is_relative_to(HERE);p.parent.mkdir(parents=True,exist_ok=True)
    if p.suffix=='.parquet':frame.to_parquet(p,index=False)
    else:frame.to_csv(p,index=False)


def protected_paths():
    files={p for d in PROTECTED for p in (ROOT/d).rglob('*') if p.is_file() and not p.is_relative_to(HERE)}
    files.update(p for p in ROOT.iterdir() if p.is_file())
    names=subprocess.check_output(['git','ls-files','-z'],cwd=ROOT).decode().split('\0')
    files.update(ROOT/n for n in names if n and (ROOT/n).is_file() and not (ROOT/n).is_relative_to(HERE))
    return sorted(files)


def verify_prior_manifest(folder):
    p=ROOT/folder;lines=(p/'manifest.sha256').read_text().splitlines()
    entries=[line.split('  ',1) for line in lines]
    assert {n for _,n in entries}=={str(f.relative_to(p)) for f in p.rglob('*') if f.is_file() and f.name!='manifest.sha256'}
    for expected,name in entries:assert sha(p/name)==expected,(folder,name)
    return dict(directory=folder,entries=len(entries),manifest_sha256=sha(p/'manifest.sha256'))


def verify_frozen():
    config=json.loads((HERE/'protocol.json').read_text())
    path=ROOT/config['frozen_selection'];assert sha(path)==config['frozen_selection_sha256']
    frozen=json.loads(path.read_text());manifest=json.loads((DEV/'frozen/configuration_manifest.json').read_text())
    assert sha(path)==manifest['selection_sha256']
    for name,expected in frozen['source_hashes'].items():assert sha(DEV/name)==expected,name
    for name,expected in frozen['historical_source_hashes'].items():assert sha(ROOT/name)==expected,name
    for name,expected in frozen['artifacts_sha256'].items():assert sha(DEV/name)==expected,name
    assert sha(ROOT/frozen['feature_definitions_path'])==frozen['feature_definitions_sha256']
    assert sha(ROOT/frozen['semantic_data_access']['source'])==frozen['semantic_data_access']['source_sha256']
    for item in frozen['configurations']:
        assert digest(item['features'])==item['feature_list_sha256']==manifest['feature_lists'][item['role']]
        assert item['missing_indicators']==['missing::'+f for f in item['features']]
    return frozen


def snapshot():
    target=HERE/'provenance/protected_files.csv'
    if target.exists():raise RuntimeError('Existing snapshot must not be replaced')
    frozen=verify_frozen()
    previous=[verify_prior_manifest(p) for p in ['reports/h2_feature_importance','reports/h2_feature_selection_dev']]
    rows=[dict(path=str(p.relative_to(ROOT)),bytes=p.stat().st_size,sha256=sha(p)) for p in protected_paths()]
    save(pd.DataFrame(rows),'provenance/protected_files.csv')
    write_json(HERE/'provenance/registration.json',dict(utc=datetime.now(timezone.utc).isoformat(),
        protocol_sha256=sha(HERE/'protocol.json'),frozen_selection_sha256=sha(DEV/'frozen/selection.json'),
        git_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT).decode().strip(),
        git_branch=subprocess.check_output(['git','branch','--show-current'],cwd=ROOT).decode().strip(),
        initial_status=subprocess.check_output(['git','status','--porcelain'],cwd=ROOT).decode(),
        protected_count=len(rows),protected_bytes=sum(r['bytes'] for r in rows),previous_manifests=previous,
        python=platform.python_version(),platform=platform.platform(),
        dependencies={n:importlib.metadata.version(n) for n in ['lightgbm','numpy','pandas','scipy','scikit-learn','joblib','pyarrow','matplotlib','pytest','threadpoolctl']}))
    write_json(HERE/'provenance/threshold_resolution.json',dict(
        frozen_file_unchanged=True,features_unchanged=True,current_user_instruction='Preserve historical A/B thresholds',
        archived_suggested_april={r['variant']:r['threshold'] for r in frozen['initial_april_thresholds']},
        authoritative_AB='data/processed/v2/final_predictions.parquet: P2, ENE, spatial_recent, lightgbm, seed 5787',
        frozen_reduced_initialization_preserved=True,
        rationale='Latest explicit instruction overrides only proposed A/B threshold initialization; it does not change frozen selections or reduced threshold policy.'))
    print('Registered protocol and protected',len(rows),'files',flush=True)


def registered():
    reg=json.loads((HERE/'provenance/registration.json').read_text())
    assert sha(HERE/'protocol.json')==reg['protocol_sha256'],'Protocol changed after registration'
    verify_frozen()
    return reg


def verify(write=False):
    registered();rows=pd.read_csv(HERE/'provenance/protected_files.csv')
    errors=[r.path for r in rows.itertuples() if not (ROOT/r.path).exists() or sha(ROOT/r.path)!=r.sha256]
    errors+=sorted(set(rows.path)^{str(p.relative_to(ROOT)) for p in protected_paths()})
    result=dict(utc=datetime.now(timezone.utc).isoformat(),protected_files=len(rows),unchanged=not errors,failures=errors,
                git_status=subprocess.check_output(['git','status','--porcelain'],cwd=ROOT).decode())
    if write:write_json(HERE/'provenance/preservation.json',result)
    assert not errors,errors
    print('Unchanged:',len(rows),'historical files',flush=True)
    return result


def manifest(check=False):
    p=HERE/'manifest.sha256';files=sorted(f for f in HERE.rglob('*') if f.is_file() and f!=p)
    if check:
        entries=[line.split('  ',1) for line in p.read_text().splitlines()]
        assert {n for _,n in entries}=={str(f.relative_to(HERE)) for f in files}
        for expected,name in entries:assert sha(HERE/name)==expected,name
    else:p.write_text(''.join(f'{sha(f)}  {f.relative_to(HERE)}\n' for f in files))
    print('Manifest', 'verified' if check else 'written',len(files),'artifacts')


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('action',choices=['snapshot','verify','verify-record','manifest','check-manifest','frozen'])
    action=parser.parse_args().action
    if action=='snapshot':snapshot()
    elif action in ['verify','verify-record']:verify(action=='verify-record')
    elif action=='frozen':registered();print('Frozen configuration unchanged')
    else:manifest(action=='check-manifest')
