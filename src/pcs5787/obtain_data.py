"""Fetch the two public Zenodo products, verify SHA-256, and safely extract.

No authentication is accepted: drafts are not treated as public downloads.
--archives permits the identical checksum/extraction path with local archives.
"""
import argparse,json,hashlib,shutil,stat,tempfile,zipfile
from pathlib import Path,PurePosixPath
from urllib.request import Request,urlopen
from urllib.error import HTTPError
from urllib.parse import urlparse
from .release_tools import sha,verify_manifest

def extract(archive,destination,root_name):
    with zipfile.ZipFile(archive) as z:
        names=[]
        for f in z.infolist():
            p=PurePosixPath(f.filename);names.append(f.filename)
            if p.is_absolute() or '..' in p.parts or '\\' in f.filename or not p.parts or p.parts[0]!=root_name or stat.S_ISLNK(f.external_attr>>16):
                raise ValueError('Unsafe archive member')
        if len(names)!=len(set(names)):raise ValueError('Duplicate archive member')
        if z.testzip() is not None:raise ValueError('Archive CRC error')
        z.extractall(destination)
    verify_manifest(Path(destination)/root_name)

def public_json(url):
    try:
        with urlopen(Request(url,headers={'User-Agent':'PCS5787-reproduction/1.0'}),timeout=60) as r:return json.load(r)
    except HTTPError as e:
        raise RuntimeError(f'Zenodo HTTP {e.code}: this reserved record may not yet be published. No draft is accessed.') from None

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--metadata',type=Path,default=Path('configs/publication/products.json'))
    ap.add_argument('--destination',type=Path,default=Path('data'));ap.add_argument('--archives',type=Path)
    a=ap.parse_args();products=json.loads(a.metadata.read_text())['datasets']
    if a.destination.exists():raise FileExistsError('Choose a new destination; existing datasets are never overwritten')
    a.destination.parent.mkdir(parents=True,exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='pcs5787-download-',dir=a.destination.parent) as tmp:
        tmp=Path(tmp);ready=[]
        for p in products:
            archive=tmp/p['filename']
            if a.archives:shutil.copyfile(a.archives/p['filename'],archive)
            else:
                record=public_json('https://zenodo.org/api/records/'+str(p['record_id']))
                matches=[x for x in record.get('files',[]) if x.get('key')==p['filename']]
                if len(matches)!=1:raise ValueError('Expected single published file not found')
                url=matches[0]['links']['self']
                if urlparse(url).scheme!='https' or urlparse(url).netloc!='zenodo.org':raise ValueError('Unexpected download host')
                with urlopen(Request(url,headers={'User-Agent':'PCS5787-reproduction/1.0'}),timeout=120) as r,archive.open('wb') as f:shutil.copyfileobj(r,f)
            if archive.stat().st_size!=p['bytes'] or sha(archive)!=p['sha256']:raise ValueError('Dataset archive checksum/size mismatch')
            ready.append((archive,p['directory']));print('Verified',p['filename'],flush=True)
        a.destination.mkdir(exist_ok=False)
        for archive,name in ready:extract(archive,a.destination,name)
    print('Both products extracted and all internal manifests verified. No model training.')
if __name__=='__main__':main()
