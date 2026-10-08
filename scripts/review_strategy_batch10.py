"""Archive the nineteen source reviews read during the ETF conversion batch."""
import argparse
import json
from pathlib import Path
import subprocess

from observe.data.store import Store
from observe.runs import file_sha, write_json
from observe.strategy_catalog import REVIEWS, catalog_strategies, read_source, strategy_id
from scripts.archive_strategy_progress import archive

SOURCES = (
    '2021年度精选策略/19.简单到发指的股指策略(2013到现在收益10949%).txt',
    '2024年度精选策略2/99.计算每日全A市场个股创新高比例(改).txt',
    '2024年度精选策略1/70.致敬市场(12)——赚钱的策略.txt',
    '2020年度精选策略/69 阻力支撑相对强度（RSRS）指标择时策略.txt',
    '聚宽2025年精选/94别人“拿的住”的股票最赚钱.txt',
    '聚宽2025年精选/14安全摸狗策略.txt',
    '2020年度精选策略/30 基于LSTM模型预测股票价格走势.txt',
    '2020年度精选策略/11 酒股地中短线策略.txt',
    '2021年度精选策略/94.小资金短线策略.txt',
    '2021年度精选策略/23.玩趋势交易的看进来~~~~2.txt',
    '2020年度精选策略/50 简单配对交易.txt',
    '2022年度精选策略/93.海龟交易体系（股票版）.txt',
    '2024年度精选策略2/92.利用myTT库整合通达信公式——以“飞鹰优选”选股公式为例.txt',
    '2022年度精选策略/55.【策略研发】三进兵策略（变形版）.txt',
    '2024年度精选策略2/33.胜率88.9%之君正集团策略-大阳分歧反包.txt',
    '2021年度精选策略/76.日内交易策略R-breaker - 300346.XSHE.txt',
    '2020年度精选策略/18 均值回归策略分享.txt',
    '2022年度精选策略/67.来自JPMorgan的VaR+EXPMA风险仓位控制.txt',
    '2021年度精选策略/6.iAlpha 基金投资策略.txt',
)


def review(root, directory):
    if not (directory.parent / 'baseline.json').exists(): raise FileNotFoundError('Establish batch checkpoint first')
    directory.mkdir(exist_ok = False); evidence = []
    for relative in SOURCES:
        source = Path('repo/量化策略源代码') / relative; text, encoding = read_source(source); sid = strategy_id(relative)
        target = directory / f'{sid}.source'; target.write_bytes(source.read_bytes())
        evidence.append({'strategy_id': sid, 'source_path': str(source), 'source_sha256': file_sha(source),
                         'encoding': encoding, 'complete_lines_read': len(text.splitlines()), 'review': REVIEWS[relative],
                         'source_copy': str(target), 'source_copy_sha256': file_sha(target)})
    search = subprocess.run(['rg', '--files', '--hidden', 'repo', 'data', '-g', '*myTT*', '-g', '*MyTT*',
                             '-g', '*technical_analysis*'], capture_output = True, text = True, check = False)
    if search.returncode not in (0, 1): raise RuntimeError(search.stderr)
    store = Store(root); snapshot = '20261005-152153-eb05'; state = store.state(snapshot)
    previous = directory.parent / 'source-reviews/review.json'
    correction = {'file': str(previous), 'sha256': file_sha(previous),
                  'changes': ['2020/11 original index union and position.price return denominator',
                              '2020/18 fixed five-stock pool', '2024/70 treasury resale loop',
                              '2025/94 index volume gate and minute weighting']} if previous.exists() and previous.parent != directory else None
    write_json(directory / 'review.json', {'sources': evidence, 'snapshot': snapshot, 'published_batch': state['batch_id'],
               'supersedes_review': correction,
               'dependency_search': {'command': search.args, 'matches': search.stdout.splitlines(), 'returncode': search.returncode,
                                     'scope': 'Filename evidence only; candidate libraries are not assumed equivalent'},
               'strategy_results': [], 'note': 'Manual source reviews only; no substitute input or new backtest'})
    catalog = catalog_strategies(root, 'repo/量化策略源代码')
    progress = archive(root, '第十批19份完整源码规则与缺口已归档，ETF/分钟/Tick/期货及历史池各项依赖分别登记')
    return {'status': 'ok', 'reviewed': len(evidence), 'catalog': catalog, 'progress': progress['output']}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description = __doc__); parser.add_argument('--root', default = 'data')
    parser.add_argument('--directory', default = 'data/staging/strategies-batch10/20261005-ledger-conversion/source-reviews')
    args = parser.parse_args(); print(json.dumps(review(args.root, Path(args.directory)), ensure_ascii = False))
