"""Archive the eight complete daily-candidate source reviews and EMA evidence."""
import argparse
import json
from pathlib import Path

from observe.runs import file_sha, write_json
from observe.strategy_catalog import REVIEWS, catalog_strategies, read_source, strategy_id
from scripts.archive_strategy_progress import archive

SOURCES = (
    '2020年度精选策略/58 MACD金叉买入，死叉卖出.txt',
    '2020年度精选策略/08 MACD单因子多头策略.txt',
    '2022年度精选策略/75.量价MACD组合择时——极速版.txt',
    '2022年度精选策略/35.量价MACD组合择时-帮你把握股市中的大趋势.txt',
    '2022年度精选策略/27.海龟交易体系-多资产版.txt',
    '2021年度精选策略/27.基于RSRS的趋势交易与网格交易相结合的尝试.txt',
    '2021年度精选策略/67.RSRS择时改进-【成交量加权-钝化-右偏】.txt',
    '2020年度精选策略/46 Get API 新技能，研究中写策略并回测.txt',
)


def review(root, directory):
    if not (directory.parent / 'baseline.json').exists(): raise FileNotFoundError('Establish batch checkpoint first')
    directory.mkdir(exist_ok = False); rows = []
    for relative in SOURCES:
        source = Path('repo/量化策略源代码') / relative; text, encoding = read_source(source); sid = strategy_id(relative)
        copied = directory / f'{sid}.source'; copied.write_bytes(source.read_bytes())
        rows.append({'strategy_id': sid, 'source_path': str(source), 'source_sha256': file_sha(source), 'encoding': encoding,
                     'complete_lines_read': len(text.splitlines()), 'review': REVIEWS[relative],
                     'source_copy': str(copied), 'source_copy_sha256': file_sha(copied)})
    ema = Path('data/staging/strategies-batch11/20261005-ema-evidence/manifest.json')
    write_json(directory / 'review.json', {'sources': rows, 'snapshot': '20261005-152153-eb05',
               'ema_evidence': {'file': str(ema), 'sha256': file_sha(ema), 'strict_implementation': 'blocked', 'variant_confirmation': 'pending'},
               'strategy_results': [], 'note': 'Complete manual reviews; no new backtest for these eight sources'})
    catalog = catalog_strategies(root, 'repo/量化策略源代码')
    progress = archive(root, '第十一批新增8份完整源码审查及EMA官方证据，缺失输入/原缺陷逐份登记，近似方案等待确认')
    return {'status': 'ok', 'reviewed': len(rows), 'catalog': catalog, 'progress': progress['output']}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description = __doc__); parser.add_argument('--root', default = 'data')
    parser.add_argument('--directory', type = Path, default = Path('data/staging/strategies-batch11/20261005-daily-stock-reviews/source-reviews'))
    args = parser.parse_args(); print(json.dumps(review(args.root, args.directory), ensure_ascii = False))
