"""Archive source40 inputs, official API semantics and indicator dependency evidence."""
import argparse
import ast
import importlib.metadata
from pathlib import Path
import tomllib

from bs4 import BeautifulSoup
import pandas as pd
import requests

from observe.data.store import Store
from observe.execution import sessions
from observe.runs import environment, file_sha, write_json
from observe.strategy_catalog import read_source
from scripts.archive_strategy_progress import archive


SOURCE = Path('repo/量化策略源代码/2024年度精选策略2/40.RSI学习贴.txt')


def probe(root, directory):
    output = directory / 'input-dependency-probe.json'
    for path in (output, directory / 'joinquant-api-response.json', directory / 'talib-pypi-response.json'):
        if path.exists(): raise FileExistsError(path)
    text, encoding = read_source(SOURCE); tree = ast.parse(text)
    pool = next(ast.literal_eval(n.value) for n in ast.walk(tree) if isinstance(n, ast.Assign)
                and any(isinstance(t, ast.Attribute) and t.attr == 'pool' for t in n.targets))
    pool = [i[:6] + ('.SH' if i.endswith('XSHG') else '.SZ') for i in pool]; unique = list(dict.fromkeys(pool))
    store = Store(root); state = store.state('20261005-152153-eb05')
    bars = store.load_state(state, 'bars_1d', filters = [('instrument', 'in', unique)])
    calendar = sessions(store.load_state(state, 'calendar')); bars['date'] = pd.to_datetime(bars.date).dt.date
    coverage = store.load_state(state, 'adj_coverage', filters = [('instrument', 'in', unique)])
    by_stock = bars.groupby('instrument').agg(rows = ('date', 'size'), first = ('date', 'min'), last = ('date', 'max'), trading = ('is_trading', 'sum'))
    ready = {i: bars[bars.instrument.eq(i) & bars.is_trading].sort_values('date').iloc[60].date for i in unique}
    missing = {i: [str(d) for d in calendar if d not in set(bars[bars.instrument.eq(i)].date)] for i in unique}
    url = 'https://www.joinquant.com/help/api/getContent?name=api'
    response = requests.get(url, timeout = 30); response.raise_for_status(); doc = response.json()
    if doc.get('code') != '00000' or not isinstance(doc.get('data'), str): raise ValueError('Official API documentation unavailable')
    raw_file = directory / 'joinquant-api-response.json'; write_json(raw_file, doc)
    content = BeautifulSoup(doc['data'], 'html.parser').get_text('\n')
    start = content.index('attribute_history(security, count')
    excerpt = content[start:content.index('get_bars', start)]
    if 'skip_paused=True' not in excerpt or '默认跳过停牌日期' not in excerpt: raise ValueError('Skip-paused signature not found')
    matching = content[content.index('回测模式整个过程是同步进行'):content.index('目前官网的模拟盘过程同回测')]
    before = tomllib.loads((directory / 'uv.lock.before').read_text()); now = tomllib.loads(Path('uv.lock').read_text())
    previous = {p['name']: p for p in before['package']}; current = {p['name']: p for p in now['package']}
    pypi = requests.get('https://pypi.org/pypi/TA-Lib/0.8.1/json', timeout = 30); pypi.raise_for_status()
    write_json(directory / 'talib-pypi-response.json', pypi.json())
    import talib
    dependency = {'wrapper_version': talib.__version__, 'c_version': talib.__ta_version__.decode(),
                  'compatibility': talib.get_compatibility(), 'rsi_unstable_period': talib.get_unstable_period('RSI'),
                  'installed_extension': str(Path(talib._ta_lib.__file__).resolve()), 'extension_sha256': file_sha(talib._ta_lib.__file__),
                  'lock_added_packages': sorted(set(current) - set(previous)), 'lock_removed_packages': sorted(set(previous) - set(current)),
                  'lock_changed_existing': sorted(n for n in set(previous) & set(current) if previous[n] != current[n]),
                  'locked_version_changes': {n: [previous[n]['version'], current[n]['version']] for n in set(previous) & set(current) if previous[n]['version'] != current[n]['version']},
                  'environment_preserved_mismatches': {p: {'installed': importlib.metadata.version(p), 'locked': current[p]['version']} for p in ('filelock', 'httpx')},
                  'installation': 'UV_PROJECT_ENVIRONMENT external WSL venv; uv pip install --python existing interpreter --no-deps --only-binary=:all: TA-Lib==0.8.1',
                  'sync_dry_run': 'Would also update preexisting filelock/httpx and reinstall editable observe; selective installation did not do this'}
    result = {'status': 'ok', 'environment': environment(), 'source_sha256': file_sha(SOURCE), 'encoding': encoding,
              'snapshot': state['snapshot_id'] if 'snapshot_id' in state else '20261005-152153-eb05', 'batch': state['batch_id'],
              'pool': pool, 'unique_pool': unique, 'source_entries': len(pool), 'distinct_instruments': len(unique),
              'bars': by_stock.reset_index().to_dict('records'), 'missing_sessions': missing,
              'first_complete_61_trading_rows': ready, 'earliest_common_decision': str(max(ready.values())),
              'suspended_rows': bars[~bars.is_trading][['date', 'instrument']].to_dict('records'), 'adjustment_coverage': coverage.to_dict('records'),
              'official_api': {'url': url, 'response_sha256': file_sha(raw_file), 'attribute_history': excerpt, 'synchronous_execution': matching,
                               'scope': 'Current official documentation evidence; not proof of original-platform historical execution or provider equivalence'},
              'dependency': dependency, 'pyproject_sha256': file_sha('pyproject.toml'), 'lock_sha256': file_sha('uv.lock'),
              'pypi_metadata_sha256': file_sha(directory / 'talib-pypi-response.json'),
              'limitations': ['Static pool selected by later strategy source is ex-post for 2021 experiments; not a proved historical stock-pool announcement',
                              'Missing trading inputs block; suspended rows are skipped as official attribute_history default, not filled',
                              'No standard data published and no old partitions changed']}
    write_json(output, result); archive(root, '第七批输入/依赖证据落盘：41股完整覆盖，官方确认跳过停牌与同步成交，TA-Lib可选依赖安装')
    return {'status': 'ok', 'output': str(output), 'earliest_common_decision': result['earliest_common_decision'], 'dependency': dependency}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description = __doc__); parser.add_argument('--root', default = 'data')
    parser.add_argument('--directory', default = 'data/staging/strategies-batch7/20261005-rsi-slots')
    args = parser.parse_args(); print(probe(args.root, Path(args.directory)))
