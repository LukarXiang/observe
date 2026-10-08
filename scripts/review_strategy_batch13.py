"""Archive dependency reviews and recompute a source's non-trading example."""
import argparse
import ast
from datetime import date
import hashlib
import json
import math
from pathlib import Path
import socket
import subprocess
from unittest.mock import patch

import numpy as np
import pandas as pd
from scipy.signal import argrelextrema

from observe.data.prices import with_adjusted
from observe.data.store import Store
from observe.runs import file_sha, write_json
from observe.strategy_catalog import REVIEWS, catalog_strategies, read_source, strategy_id
from scripts.archive_strategy_progress import archive
from scripts.run_strategy_batch4 import protect, read
from scripts.verify_strategy_batch3 import require

SNAPSHOT = '20261005-152153-eb05'
SOURCES = (
    '2020年度精选策略/97 银行翻倍策略.txt',
    '2020年度精选策略/39 因子分析——利用JoinQuant因子分析模块选取因子并封装为策略.txt',
    '2020年度精选策略/41 多因子模型（三）-交易回测.txt',
    '2022年度精选策略/23盘口信息获取与判断.txt',
    '2023年度精选策略/8.iAlpha 基金投资策略.txt',
    '2023年度精选策略/82.势的度量.txt',
    '2023年度精选策略/69.选取向上趋势的股票.txt',
    '2024年度精选策略2/94.FoF, all in.txt',
    '聚宽2025年精选/100使用K-means 聚类对基金分类.py',
    '聚宽2025年精选/20小盘股动态调仓，100只10年17倍.py',
    '2023年度精选策略/11.致敬市场--5 定投ETF之躺平赢.txt',
    '2020年度精选策略/99 一个中信证券的向导策略.txt',
)


def review(root, directory):
    require((directory / 'baseline.json').is_file(), 'Batch baseline missing')
    latest = read(Path(root) / 'catalog/strategies/latest.json')
    old = {r['path']: r for r in read(Path(latest['directory']) / 'catalog.json')}
    for name in SOURCES:
        require(old[name]['review_status'] != '人工审查完成', 'Source already reviewed')
        require(file_sha(Path('repo/量化策略源代码') / name) == old[name]['bytes_sha256'], 'Original source changed')
    copied_dir = directory / 'source-reviews'; copied_dir.mkdir(exist_ok=False)
    records = []
    for name in SOURCES:
        source = Path('repo/量化策略源代码') / name; text, encoding = read_source(source)
        copied = copied_dir / f'{strategy_id(name)}.source'
        with copied.open('xb') as stream: stream.write(source.read_bytes())
        records.append({'strategy_id': strategy_id(name), 'source_path': str(source), 'source_sha256': file_sha(source),
            'source_copy': str(copied), 'source_copy_sha256': file_sha(copied), 'encoding': encoding,
            'complete_lines_read': len(text.splitlines()), 'review': REVIEWS[name]})
    search = subprocess.run(['rg', '--files', '--hidden', 'repo', 'data', '-g', 'MyPackage_Final.pkl', '-g', 'wizard.py'],
                            capture_output=True, text=True, check=False)
    require(search.returncode in (0, 1), search.stderr)
    catalog = catalog_strategies(root, 'repo/量化策略源代码')
    current = read(Path(catalog['output']) / 'catalog.json')
    changed = [{'path': r['path'], 'old_apis': old[r['path']]['apis'], 'new_apis': r['apis']}
               for r in current if old[r['path']]['apis'] != r['apis']]
    result = {'sources': records, 'snapshot': SNAPSHOT, 'strategy_results': [],
        'dependency_search': {'argv': search.args, 'returncode': search.returncode, 'matches': search.stdout.splitlines(), 'pickle_executed': False},
        'static_detection_changes': changed, 'catalog': catalog,
        'limits': ['Six research/fragment sources have no trading rules', 'Continuous financial API detection does not supply historical ROE',
                   'No new strategy backtest or replacement input published']}
    write_json(copied_dir / 'review.json', result)
    return {'status': 'ok', 'reviewed': len(records), 'api_changes': len(changed),
            'progress': archive(root, '第十三批12份完整审查及连续财务API漏检修复已归档，研究脚本与交易策略逐份区分')['output']}


def independent_scores(closes):
    sign = lambda value: (value > 0) - (value < 0)
    normalized = []; cumulative = 0.0
    for i, close in enumerate(closes):
        if i >= 4:
            monotone = sign(close / closes[i - 1] - 1)
            mean = sum(closes[i - 4:i + 1]) / 5
            ma = sign(close - mean)
            if monotone == 0: monotone = -ma
            if ma == 0: ma = monotone
            cumulative += (monotone + ma) / 2
        normalized.append(cumulative)
    return normalized


def independent_trend(values):
    points = [0] + [i for i in range(1, len(values) - 1)
                    if values[i] > max(values[i - 1], values[i + 1]) or values[i] < min(values[i - 1], values[i + 1])] + [len(values) - 1]
    continuous = sum((values[b] - values[a]) ** 2 for a, b in zip(points[:-1], points[1:], strict=True))
    absolute = (values[-1] - values[0]) ** 2 if max(values) > max(values[1:-1]) and min(values) < min(values[1:-1]) else continuous
    return continuous, absolute, absolute / (len(values) - 1) ** 1.5


def study(root, directory):
    output = directory / 'trend-research.json'
    if output.exists(): raise FileExistsError(output)
    record = next(r for r in read(directory / 'source-reviews/review.json')['sources'] if r['source_path'].endswith(SOURCES[5]))
    source = Path(record['source_copy']); require(file_sha(source) == record['source_sha256'], 'Reviewed study source changed')
    text, _ = read_source(source); tree = ast.parse(text)
    names = ('normalize_compound', 'continuous_score', 'absolute_score', 'ultimate_score')
    functions = {node.name: node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in names}
    require(set(functions) == set(names), 'Original functions missing')
    # Only the four fully reviewed function definitions run; no imports, notebook cells or provider calls.
    selected = ast.Module(body=[functions[name] for name in names], type_ignores=[])
    namespace = {'np': np, 'pd': pd, 'argrelextrema': argrelextrema}
    exec(compile(selected, str(source), 'exec'), namespace)
    examples = [[0,1,2,3,4,5,6,7,8,9,10], [0,1,2,3,4,5,4,5,6,7,8], [0,1,2,3,4,5,6,5,4,3,2],
                [0,1,2,3,2,3,4,3,4,5,6], [0,1,2,1,0,1,2,1,0,1,2], [0,1,0,1,0,1,0,1,0,1,0]]
    example_results = []
    for values in examples:
        array = np.array(values, dtype=float)
        actual = [float(namespace[name](array)) for name in names[1:]]
        require(np.allclose(actual, independent_trend(values), rtol=0, atol=1e-12), 'Original array arithmetic differs')
        example_results.append({'input': values, 'continuous': actual[0], 'absolute': actual[1], 'ultimate': actual[2]})
    store = Store(root); state = store.state(SNAPSHOT); instruments = REVIEWS[SOURCES[5]]['rules']['pool']
    filters = [('instrument', 'in', instruments)]
    prices = with_adjusted(store.load_state(state, 'bars_1d', filters=filters),
        store.load_state(state, 'adj_factors', filters=filters), store.load_state(state, 'adj_coverage', filters=filters))
    prices['date'] = pd.to_datetime(prices.date).dt.date
    rows = []; inputs = []
    for instrument in instruments:
        window = prices[prices.instrument.eq(instrument) & prices.date.le(date(2021, 3, 8))].sort_values('date').tail(30).copy()
        require(len(window) == 30 and window.date.max() == date(2021, 3, 8) and not window.date.duplicated().any(), 'Original date/window unavailable')
        require(window.adjustment_status.eq('usable').all() and np.isfinite(window.close_adj).all() and (window.close_adj > 0).all(), 'Adjustment input unavailable')
        window['anchored_close'] = window.close_adj / float(window.back_factor.iloc[-1])
        normalized = namespace['normalize_compound'](window.anchored_close)
        oracle = independent_scores(window.anchored_close.tolist())
        require(np.array_equal(normalized, oracle), 'Compound normalization differs')
        score = float(namespace['ultimate_score'](normalized))
        require(math.isclose(score, independent_trend(oracle)[2], rel_tol=0, abs_tol=1e-12), 'Stock trend differs')
        window['normalized'] = normalized; inputs.append(window)
        rows.append({'instrument': instrument, 'first': str(window.date.min()), 'last': str(window.date.max()),
                     'rows': len(window), 'paused_rows': int((~window.is_trading.astype(bool)).sum()), 'score': score,
                     'normalization': normalized.tolist()})
    data = directory / 'trend-inputs.parquet'; require(not data.exists(), 'Study input archive exists')
    pd.concat(inputs, ignore_index=True).to_parquet(data, index=False)
    result = {'status': 'research_recomputed_with_limits', 'snapshot': SNAPSHOT, 'source_sha256': record['source_sha256'],
        'selected_function_ast_sha256': hashlib.sha256(ast.dump(selected).encode()).hexdigest(),
        'examples': example_results, 'stocks': rows, 'ranking': sorted(rows, key=lambda item: item['score'], reverse=True),
        'input_file': str(data), 'input_sha256': file_sha(data), 'independent_check': 'Scalar signs, strict local extrema and endpoint condition; 6 arrays and all 90 closes checked',
        'not_a_backtest': True, 'limitations': ['Original source supplies no trading, holdings or cost policy',
            'Vendor adjusted prices anchored to sample end; no original JoinQuant price equality proof',
            'Exact comparisons may be sensitive to floating-point adjustment scaling', 'Static endpoint/local-extrema statistics are retrospective, not causal trading signals']}
    write_json(output, result)
    archive(root, '第十三批势的度量原6数组与三股90条价格研究例子复算已落盘，不计入交易回测或策略复现')
    return {'status': result['status'], 'stocks': rows, 'output': str(output)}


def verify_study(root, directory):
    output = directory / 'trend-offline-verification.json'
    if output.exists(): raise FileExistsError(output)
    original = directory / 'trend-research.json'; before = file_sha(original)
    reference = read(original); repeated = directory / 'trend-offline'; repeated.mkdir(exist_ok=False)
    reviews = repeated / 'source-reviews'; reviews.mkdir()
    with (reviews / 'review.json').open('xb') as stream: stream.write((directory / 'source-reviews/review.json').read_bytes())
    def forbidden(*args, **kwargs): raise AssertionError('Research verification network disabled')
    with patch.object(socket.socket, 'connect', forbidden), patch.object(socket.socket, 'connect_ex', forbidden):
        study(root, repeated)
    again = read(repeated / 'trend-research.json')
    expected = {k: v for k, v in reference.items() if k != 'input_file'}
    actual = {k: v for k, v in again.items() if k != 'input_file'}
    require(expected == actual and file_sha(original) == before, 'Offline research differs or original changed')
    pd.testing.assert_frame_equal(pd.read_parquet(reference['input_file']), pd.read_parquet(again['input_file']))
    result = {'status': 'match', 'differences': 0, 'original_unchanged': True, 'not_a_strategy_reproduction': True,
        'original_file': str(original), 'original_sha256': before,
        'recomputed_file': str(repeated / 'trend-research.json'), 'recomputed_sha256': file_sha(repeated / 'trend-research.json'),
        'input_sha256': reference['input_sha256'], 'network': 'socket.connect and connect_ex forbidden'}
    write_json(output, result)
    return result


def finish(root, directory):
    output = Path('docs/handoff/2026-10-05-batch13-verification.json')
    if output.exists(): raise FileExistsError(output)
    reviews = read(directory / 'source-reviews/review.json')
    require(len(reviews['sources']) == len(SOURCES), 'Source reviews incomplete')
    for record in reviews['sources']:
        require(file_sha(Path(record['source_path'])) == record['source_sha256'] == file_sha(Path(record['source_copy'])), 'Source evidence changed')
    research = read(directory / 'trend-research.json')
    require(research['not_a_backtest'] and file_sha(Path(research['input_file'])) == research['input_sha256'], 'Research input changed')
    checks = read(directory / 'checks.json')
    require(checks['commands'] and all(r['returncode'] == 0 for r in checks['commands']), 'Checks failed')
    evidence = checks['evidence_sha256']
    expected_evidence = {'source-reviews/review.json', 'trend-research.json', 'trend-offline-verification.json'}
    require(set(evidence) == expected_evidence, 'Required checked evidence missing')
    for name, sha in evidence.items(): require(file_sha(directory / name) == sha, 'Checked evidence changed')
    offline = read(directory / 'trend-offline-verification.json')
    require(offline['status'] == 'match' and offline['differences'] == 0 and offline['original_unchanged'], 'Offline research did not match')
    require(file_sha(directory / 'trend-research.json') == offline['original_sha256'], 'Original research fingerprint differs')
    require(file_sha(Path(offline['recomputed_file'])) == offline['recomputed_sha256'], 'Recomputed research changed')
    recomputed = read(Path(offline['recomputed_file']))
    require(file_sha(Path(recomputed['input_file'])) == recomputed['input_sha256'] == offline['input_sha256'], 'Recomputed input changed')
    files = []; repository = Path.cwd().resolve()
    for name, sha in checks['implementation_sha256'].items():
        source = Path(name).resolve(); relative = source.relative_to(repository)
        require(file_sha(source) == sha and relative not in {r for _, r in files}, 'Checked implementation changed or duplicated')
        files.append((source, relative))
    required = {Path(__file__).resolve(), repository / 'src/observe/strategy_catalog.py', repository / 'tests/unit/test_catalog_dependencies.py'}
    require(required.issubset({p for p, _ in files}), 'Required checked code missing')
    baseline = read(directory / 'baseline.json')
    require(Store(root).published() == baseline['published'], 'Published inputs changed')
    protection = protect(root, directory)
    frozen = directory / 'implementation-final'; frozen.mkdir(exist_ok=False)
    for source, relative in files:
        target = frozen / relative; target.parent.mkdir(parents=True, exist_ok=True)
        with target.open('xb') as stream: stream.write(source.read_bytes())
    result = {'status': 'ok_with_deferred_strategies', 'reviews': reviews, 'research': research, 'research_offline': offline, 'checks': checks,
        'protection': protection, 'scope': 'Twelve source reviews, continuous financial API detection fix, one non-trading research example',
        'progress': archive(root, '第十三批验收：12份审查、连续财务漏检修复及三股研究复算；旧输入只读，全库继续')}
    write_json(output, result)
    return {'status': result['status'], 'output': str(output), 'progress': result['progress']['output']}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['review', 'study', 'verify-study', 'finish'])
    parser.add_argument('--root', default='data')
    parser.add_argument('--directory', type=Path, default=Path('data/staging/strategies-batch13/20261005-dependency-and-research'))
    args = parser.parse_args()
    print(json.dumps({'review': review, 'study': study, 'verify-study': verify_study, 'finish': finish}[args.action](args.root, args.directory), ensure_ascii=False), flush=True)
