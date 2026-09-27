from __future__ import annotations
import concurrent.futures as cf
import hashlib, json, os, re, subprocess, time
from pathlib import Path

REV='afeacb11bcc94dadfd1c8f483ee4377b2b8b614e'
BASE='https://hf-mirror.com/datasets/phamquiluan/RCAEval/resolve'
OUT=Path('/root/autodl-tmp/moe_rca/data/raw/rcaeval_hf_afeacb11')
AUDIT=Path('/root/autodl-tmp/moe_rca/data/audit')
PATHS=[x.strip() for x in (AUDIT/'hf_tree_paths.txt').read_text().splitlines() if x.strip()]
OUT.mkdir(parents=True, exist_ok=True)

def sha256_file(p: Path):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(1024*1024), b''): h.update(b)
    return h.hexdigest()

def head_meta(url: str):
    cp=subprocess.run(['curl','-sS','-I','--connect-timeout','15','--max-time','30',url],capture_output=True,text=True,check=True)
    hdr={}
    for line in cp.stdout.splitlines():
        if ':' in line:
            k,v=line.split(':',1); hdr[k.strip().lower()]=v.strip()
    return hdr

def get_file(url: str, tmp: Path):
    cp=subprocess.run(['curl','-L','--fail','--retry','3','--retry-delay','1','--connect-timeout','20','--max-time','900','-sS','-o',str(tmp),url],capture_output=True,text=True)
    if cp.returncode != 0:
        raise RuntimeError(cp.stderr.strip() or f'curl rc={cp.returncode}')

def one(path: str):
    url=f'{BASE}/{REV}/{path}'
    dest=OUT/path; dest.parent.mkdir(parents=True, exist_ok=True)
    last=''
    for attempt in range(1,5):
        tmp=dest.with_name(dest.name+'.part')
        try:
            exp_sha=None; exp_size=None; repo_commit=None
            if path.endswith('.parquet'):
                hdr=head_meta(url)
                repo_commit=hdr.get('x-repo-commit')
                if repo_commit and repo_commit != REV: raise RuntimeError(f'commit mismatch {repo_commit}')
                exp_sha=(hdr.get('x-linked-etag') or '').strip('"') or None
                exp_size=int(hdr['x-linked-size']) if hdr.get('x-linked-size') else None
            if dest.exists() and exp_size is not None and dest.stat().st_size==exp_size:
                got=sha256_file(dest)
                if exp_sha and got==exp_sha:
                    return {'path':path,'bytes':exp_size,'sha256':got,'expected_sha256':exp_sha,'repo_commit':repo_commit,'status':'verified-existing'}
            get_file(url,tmp)
            n=tmp.stat().st_size; got=sha256_file(tmp)
            if exp_size is not None and n!=exp_size: raise RuntimeError(f'size {n} != {exp_size}')
            if exp_sha and got!=exp_sha: raise RuntimeError(f'sha256 {got} != {exp_sha}')
            os.replace(tmp,dest)
            return {'path':path,'bytes':n,'sha256':got,'expected_sha256':exp_sha,'repo_commit':repo_commit,'status':'downloaded'}
        except Exception as e:
            last=repr(e)
            try: tmp.unlink(missing_ok=True)
            except Exception: pass
            if attempt<4: time.sleep(min(2**attempt,8))
    return {'path':path,'status':'ERROR','error':last}

start=time.time(); results=[]; errors=[]
print(f'START revision={REV} files={len(PATHS)} workers=16',flush=True)
with cf.ThreadPoolExecutor(max_workers=16) as ex:
    futs={ex.submit(one,p):p for p in PATHS}
    for i,f in enumerate(cf.as_completed(futs),1):
        r=f.result(); results.append(r)
        if r['status']=='ERROR': errors.append(r); print('ERROR',r['path'],r['error'],flush=True)
        if i%50==0 or i==len(PATHS):
            b=sum(x.get('bytes',0) for x in results if x['status']!='ERROR')
            print(f'PROGRESS {i}/{len(PATHS)} bytes={b} errors={len(errors)} elapsed_s={time.time()-start:.1f}',flush=True)
results.sort(key=lambda x:x['path'])
(AUDIT/'hf_download_manifest.json').write_text(json.dumps({'revision':REV,'source_repo':'phamquiluan/RCAEval','transport_endpoint':'https://hf-mirror.com','results':results},indent=2))
with (AUDIT/'hf_download_manifest.tsv').open('w') as f:
    f.write('path\tbytes\tsha256\texpected_sha256\trepo_commit\tstatus\n')
    for r in results:
        f.write(f"{r['path']}\t{r.get('bytes','')}\t{r.get('sha256','')}\t{r.get('expected_sha256','')}\t{r.get('repo_commit','')}\t{r['status']}\n")
print(f'DONE files={len(results)} errors={len(errors)} bytes={sum(x.get("bytes",0) for x in results if x["status"]!="ERROR")} elapsed_s={time.time()-start:.1f}',flush=True)
if errors: raise SystemExit(2)
