from __future__ import annotations
import concurrent.futures as cf
import hashlib, json, os, time
from pathlib import Path
import httpx

REV='afeacb11bcc94dadfd1c8f483ee4377b2b8b614e'
BASE='https://hf-mirror.com/datasets/phamquiluan/RCAEval/resolve'
OUT=Path('/root/autodl-tmp/moe_rca/data/raw/rcaeval_hf_afeacb11')
AUDIT=Path('/root/autodl-tmp/moe_rca/data/audit')
PATHS=[x.strip() for x in (AUDIT/'hf_tree_paths.txt').read_text().splitlines() if x.strip()]
OUT.mkdir(parents=True, exist_ok=True)
limits=httpx.Limits(max_connections=32, max_keepalive_connections=24)
timeout=httpx.Timeout(180.0, connect=30.0)
client=httpx.Client(follow_redirects=True, timeout=timeout, limits=limits, headers={'User-Agent':'moe-rca-data-audit/1.0'})

def sha256_file(p: Path):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(1024*1024), b''): h.update(b)
    return h.hexdigest()

def one(path: str):
    url=f'{BASE}/{REV}/{path}'
    dest=OUT/path
    dest.parent.mkdir(parents=True, exist_ok=True)
    last=None
    for attempt in range(1,5):
        tmp=dest.with_name(dest.name+'.part')
        try:
            head=client.head(url, follow_redirects=False)
            head.raise_for_status()
            repo_commit=head.headers.get('x-repo-commit')
            if repo_commit and repo_commit != REV:
                raise RuntimeError(f'commit mismatch {repo_commit}')
            linked_sha=head.headers.get('x-linked-etag','').strip('"') or None
            linked_size=head.headers.get('x-linked-size')
            expected_size=int(linked_size) if linked_size else None
            if dest.exists() and expected_size is not None and dest.stat().st_size == expected_size:
                got=sha256_file(dest)
                if linked_sha and got == linked_sha:
                    return {'path':path,'bytes':expected_size,'sha256':got,'expected_sha256':linked_sha,'status':'verified-existing'}
            h=hashlib.sha256(); n=0
            with client.stream('GET', url) as r:
                r.raise_for_status()
                with tmp.open('wb') as f:
                    for chunk in r.iter_bytes(1024*1024):
                        if chunk:
                            f.write(chunk); h.update(chunk); n += len(chunk)
            got=h.hexdigest()
            if expected_size is not None and n != expected_size:
                raise RuntimeError(f'size {n} != {expected_size}')
            if linked_sha and got != linked_sha:
                raise RuntimeError(f'sha256 {got} != {linked_sha}')
            os.replace(tmp,dest)
            return {'path':path,'bytes':n,'sha256':got,'expected_sha256':linked_sha,'status':'downloaded'}
        except Exception as e:
            last=repr(e)
            try: tmp.unlink(missing_ok=True)
            except Exception: pass
            if attempt < 4: time.sleep(min(2**attempt,8))
    return {'path':path,'status':'ERROR','error':last}

start=time.time(); results=[]; errors=[]
print(f'START revision={REV} files={len(PATHS)} workers=16', flush=True)
with cf.ThreadPoolExecutor(max_workers=16) as ex:
    futs={ex.submit(one,p):p for p in PATHS}
    for i,fut in enumerate(cf.as_completed(futs),1):
        r=fut.result(); results.append(r)
        if r['status']=='ERROR':
            errors.append(r); print('ERROR',r['path'],r['error'],flush=True)
        if i % 50 == 0 or i == len(PATHS):
            total=sum(x.get('bytes',0) for x in results if x['status']!='ERROR')
            print(f'PROGRESS {i}/{len(PATHS)} bytes={total} errors={len(errors)} elapsed_s={time.time()-start:.1f}',flush=True)
results.sort(key=lambda x:x['path'])
(AUDIT/'hf_download_manifest.json').write_text(json.dumps({'revision':REV,'source_repo':'phamquiluan/RCAEval','transport_endpoint':'https://hf-mirror.com','results':results},indent=2))
with (AUDIT/'hf_download_manifest.tsv').open('w') as f:
    f.write('path\tbytes\tsha256\texpected_sha256\tstatus\n')
    for r in results:
        f.write(f"{r['path']}\t{r.get('bytes','')}\t{r.get('sha256','')}\t{r.get('expected_sha256','')}\t{r['status']}\n")
print(f'DONE files={len(results)} errors={len(errors)} bytes={sum(x.get("bytes",0) for x in results if x["status"]!="ERROR")} elapsed_s={time.time()-start:.1f}',flush=True)
if errors: raise SystemExit(2)
