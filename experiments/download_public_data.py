"""Download pinned public datasets used by round 08; no Kaggle token required.

Run from the repository: python experiments/download_public_data.py
Data is stored under git-ignored data/external, never committed.
"""
import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
import shutil
import stat
import subprocess
import zipfile

SOURCES={
    'open':('crtproduction/name-that-llm-open-pool',18,
            '1017d4f8ed47815b233876846bfc737b9c1d0fbc3a5a5a21718ac9f19968e403'),
    'instyle':('prvsiyan/ntl-public-in-style-pool-v1',2,None),
    'extended':('prvsiyan/prvsiyans-extended-name-that-llm-dataset',13,None),
}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source',choices=SOURCES,default='open')
    parser.add_argument('--output',type=Path)
    args=parser.parse_args()
    dataset,version,expected=SOURCES[args.source]
    root=Path(__file__).resolve().parents[1]
    folder=args.output or root/'data/external'/(dataset.replace('/','__')+'__v'+str(version))
    folder=folder.resolve()
    folder.mkdir(parents=True,exist_ok=True)
    archive=folder/'download.zip'
    url=f'https://www.kaggle.com/api/v1/datasets/download/{dataset}?datasetVersionNumber={version}'
    curl=shutil.which('curl.exe') or shutil.which('curl')
    if not curl: raise RuntimeError('curl is required; alternatively download the pinned dataset from Kaggle.')
    for attempt in range(5):
        if archive.is_file() and zipfile.is_zipfile(archive): break
        print(f'Download {dataset} v{version}; attempt {attempt+1}; resumable',flush=True)
        result=subprocess.run([curl,'-f','-sS','-L','-C','-','--max-time','240',url,'-o',str(archive)])
        if result.returncode not in (0,18,28,56):
            raise RuntimeError(f'Download failed: curl code {result.returncode}; {url}')
    if not zipfile.is_zipfile(archive): raise ValueError(f'Incomplete archive. Rerun to resume: {archive}')
    digest=hashlib.sha256(archive.read_bytes()).hexdigest()
    if expected and digest!=expected:
        raise ValueError('Pinned archive hash differs. Stop and inspect the dataset version.')
    with zipfile.ZipFile(archive) as z:
        for member in z.infolist():
            p=PurePosixPath(member.filename)
            target=(folder/member.filename).resolve()
            if (p.is_absolute() or '..' in p.parts or '\\' in member.filename or
                ':' in member.filename or stat.S_ISLNK(member.external_attr>>16) or
                not target.is_relative_to(folder)):
                raise ValueError(f'Unsafe archive path: {member.filename}')
        if z.testzip() is not None: raise ValueError('Archive CRC check failed.')
        z.extractall(folder)
        names=z.namelist()
    manifest={'dataset':dataset,'version':version,'download_url':url,'archive_sha256':digest,
              'files':{n:hashlib.sha256((folder/n).read_bytes()).hexdigest() for n in names if (folder/n).is_file()}}
    (folder/'download_manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
    print('Verified:',folder,flush=True)


if __name__=='__main__': main()
