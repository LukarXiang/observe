"""Archive the next three complete source reviews and existing weight API limits."""
import argparse
import inspect
import json
from pathlib import Path

import akshare as ak

from observe.data.sources.baostock import BaoStock
from observe.data.store import Store
from observe.runs import file_sha, write_json
from observe.strategy_catalog import REVIEWS, catalog_strategies, read_source, strategy_id
from scripts.archive_strategy_progress import archive

SOURCES = ('2020年度精选策略/79 DMI——大盘择时.txt',
           '2021年度精选策略/3.一个简单而持续稳定的懒人超额收益策略.txt',
           '2022年度精选策略/66.【配对交易】基于zscore的配对交易策略（胜率88.2%）.txt')


def review(root, directory):
    directory.mkdir(parents = True, exist_ok = False)
    evidence = []
    for relative in SOURCES:
        source = Path('repo/量化策略源代码') / relative; text, encoding = read_source(source); sid = strategy_id(relative)
        destination = directory / f'{sid}.source'; destination.write_bytes(source.read_bytes())
        evidence.append({'strategy_id': sid, 'source_path': str(source), 'source_sha256': file_sha(source), 'encoding': encoding,
                         'complete_lines_read': len(text.splitlines()), 'source_copy': str(destination), 'review': REVIEWS[relative]})
    weight = ak.index_stock_cons_weight_csindex
    api = {'akshare_weight': {'name': weight.__name__, 'signature': str(inspect.signature(weight)), 'doc': inspect.getdoc(weight),
                             'historical_date_parameter': False, 'policy': 'Latest only; do not request latest values as missing historical inputs'},
           'baostock_constituents': {'name': 'index_constituents', 'source': inspect.getsource(BaoStock.index_constituents),
                                    'policy': 'Membership API, not index weights'}, 'network_requests': 0}
    write_json(directory / 'existing-weight-api.json', api)
    state = Store(root).state('20261005-152153-eb05'); filters = [('instrument', 'in', ['002415.SZ', '000651.SZ'])]
    bars = Store(root).load_state(state, 'bars_1d', filters = filters)
    coverage = Store(root).load_state(state, 'adj_coverage', filters = filters)
    write_json(directory / 'review.json', {'sources': evidence, 'snapshot': '20261005-152153-eb05', 'published_batch': state['batch_id'],
                                         'pair_bars': bars.groupby('instrument').agg(rows = ('date', 'size'), first = ('date', 'min'), last = ('date', 'max')).reset_index().to_dict('records'),
                                         'pair_adjustment_coverage': coverage.to_dict('records'), 'weight_api_evidence_sha256': file_sha(directory / 'existing-weight-api.json'),
                                         'reproduction_experiments': [], 'note': 'Read/review evidence only; no substitute inputs and no strategy results claimed'})
    catalog = catalog_strategies(root, 'repo/量化策略源代码')
    progress = archive(root, '下一批三份人工审查归档：DMI79不可交易指数；月初权重策略缺历史weight；配对66兼容修正待确认')
    return {'status': 'ok', 'directory': str(directory), 'catalog': catalog, 'progress': progress['output']}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description = __doc__); parser.add_argument('--root', default = 'data')
    parser.add_argument('--directory', default = 'data/staging/strategies-batch8/20261005-next-reviews')
    args = parser.parse_args(); print(json.dumps(review(args.root, Path(args.directory)), ensure_ascii = False))
