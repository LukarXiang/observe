"""Bounded original-code/data dependency probes; never publish provider candidates as Store inputs."""
import argparse
import importlib.util
import inspect
import io
import json
from pathlib import Path
import re
import shutil
import subprocess

import akshare as ak
import pandas as pd
import requests

from observe.runs import file_sha
from scripts.review_strategy_batch18 import save
from scripts.verify_strategy_batch3 import require

DIRECTORY = Path('data/staging/strategies-batch105/20261011-primary-dependency-probes')
URLS = {
    'author_folder': 'https://drive.google.com/drive/folders/1AKnaQVwp_DdZ6wl27miQTWwuJAcAPDPv?usp=share_link',
    'sw_history': 'https://www.swsresearch.com/swindex/pdf/SwClass2021/StockClassifyUse_stock.xls',
}


def fetch(name, url):
    record = {'name': name, 'url': url, 'timeout_seconds': [5,12], 'follow_redirects': False}
    try:
        with requests.get(url,timeout=(5,12),allow_redirects=False,stream=True) as response:
            record['status_code'] = response.status_code
            if response.status_code != 200:
                record['status'] = 'http_status_blocked'; return record
            path = DIRECTORY/(name+'.response'); size = 0
            with path.open('xb') as stream:
                for chunk in response.iter_content(65536):
                    size += len(chunk); require(size <= 16*1024*1024,'Response exceeds bounded16MiB')
                    stream.write(chunk)
            record.update(status='response',file=str(path),bytes=size,sha256=file_sha(path))
    except Exception as exc:
        message = re.sub(r'(https?://)[^/\s@]+@',r'\1[redacted]@',str(exc))
        record.update(status='failed',error={'type':type(exc).__name__,'message':message[:1500]})
    return record


def probe():
    require(not DIRECTORY.exists(),'Immutable dependency probes exist'); DIRECTORY.mkdir(parents=True)
    installed = Path(inspect.getfile(ak.stock_industry_clf_hist_sw))
    repo_source = Path('repo/akshare/akshare/stock/stock_industry_sw.py')
    for name,path in [('installed-api.py',installed),('repo-api.py',repo_source)]:
        shutil.copyfile(path,DIRECTORY/name)
    version = subprocess.run(['git','-C','repo/akshare','rev-parse','HEAD'],capture_output=True,text=True,check=False)
    sources = {'akshare_version':ak.__version__,'installed_api_sha256':file_sha(installed),
        'repo_api_sha256':file_sha(repo_source),'repo_commit':version.stdout.strip() if version.returncode == 0 else None,
        'existing_function':'stock_industry_clf_hist_sw','signature':str(inspect.signature(ak.stock_industry_clf_hist_sw)),
        'missing_local_packages':[name for name in ('knight_scr','scr') if importlib.util.find_spec(name) is None]}
    results = []
    for name,url in URLS.items():
        record = fetch(name,url); results.append(record); save(DIRECTORY/(name+'-request.json'),record)
        if name == 'sw_history' and record['status'] == 'response':
            try:
                frame = pd.read_excel(io.BytesIO(Path(record['file']).read_bytes()),dtype={'股票代码':str,'行业代码':str})
                frame.to_parquet(DIRECTORY/'sw-history-candidate.parquet',index=False)
                record['candidate'] = {'rows':len(frame),'columns':list(frame.columns),
                    'file':str(DIRECTORY/'sw-history-candidate.parquet'),
                    'sha256':file_sha(DIRECTORY/'sw-history-candidate.parquet'),'strict_usable':False}
            except Exception as exc:
                record['parse_error'] = {'type':type(exc).__name__,'message':str(exc)[:1000]}
    result = {'status':'dependency_inventory_only','references':sources,'requests':results,
        'actual_requests_attempted':len(results),'responses_received':sum('status_code' in r for r in results),
        'complete_original_SW_membership_proven':False,'original_nhnl_implementation_proven':False,
        'new_publications':0,'new_trading_backtests':0,'library_goal_complete':False}
    save(DIRECTORY/'checked.json',result); return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument('action',choices=('probe',))
    parser.parse_args(); print(json.dumps(probe(),ensure_ascii=False,allow_nan=False))
