"""策略库统一记录：每个策略最近一次完成的回测 + 目录信息 + 规格里的实现思路与差异。

输出 strategies/results.csv（逐策略一行）与 docs/09-策略库回测汇总.md。只读实验目录，不重跑。
"""
import json
from pathlib import Path

import pandas as pd
import yaml

from ..runs import RunRegistry

DONE = ('success', 'success_limited', 'blocked')
METRICS = ('total_return', 'annual_return', 'annual_vol', 'sharpe', 'max_drawdown')


def _read(path):
    p = Path(path); return json.loads(p.read_text(encoding = 'utf-8')) if p.exists() else {}


def latest_runs(root):
    """{策略编号: 实验登记行}，每个策略取 finished_at 最晚的已完成实验"""
    reg = RunRegistry(Path(root) / 'runs'); out = {}
    for r in reg.list(limit = 100000, kind = 'strategy'):
        if r['status'] not in DONE or not r['finished_at']: continue
        sid = r['name']
        if sid not in out or r['finished_at'] > out[sid]['finished_at']: out[sid] = r
    return out


def collect(root, catalog = 'strategies/catalog.csv', specs = 'strategies/specs'):
    cat = pd.read_csv(catalog, dtype = str, keep_default_na = False).set_index('id') if Path(catalog).exists() else pd.DataFrame()
    runs = latest_runs(root); rows = []
    for path in sorted(Path(specs).glob('*.yaml')):
        spec = yaml.safe_load(path.read_text(encoding = 'utf-8')); sid = spec['id']; r = runs.get(sid)
        row = {'id': sid, 'title': spec['title'], 'archetype': spec['archetype'], 'source_file': spec['source']['file'], 'url': spec['source'].get('url', ''),
               'spec': path.as_posix(), 'deviations': len(spec.get('deviations', [])), 'run_status': r['status'] if r else 'not_run', 'run_id': r['run_id'] if r else ''}
        if r:
            out = Path(r['path']); m = _read(out / 'metrics.json'); t = _read(out / 'trading.json'); s = _read(out / 'status.json')
            summ = s.get('summary', {})
            row.update(start = summ.get('start'), end = summ.get('end'), **{k: m.get(k) for k in METRICS},
                       turnover_daily = t.get('turnover_two_sided_daily_mean'), fee_ratio = t.get('fee_ratio_to_initial'), cash_share = t.get('cash_share_mean'),
                       rebalances = summ.get('rebalances'), snapshot_id = r['snapshot_id'], git_commit = r['git_commit'],
                       **{f'ret_{y}': v for y, v in (m.get('yearly_returns') or {}).items()})
            if r['status'] == 'blocked': row['blocked'] = '; '.join(x.get('kind', '') for x in s.get('blocked', []))
        if sid in cat.index: row['catalog_status'] = cat.at[sid, 'status']
        rows.append(row)
    return pd.DataFrame(rows).reindex(columns = list(dict.fromkeys(['id', 'run_status', *METRICS, *[k for row in rows for k in row]])))


def _pct(x): return '' if x is None or pd.isna(x) else f'{x:.1%}'
def _num(x): return '' if x is None or pd.isna(x) else f'{x:.2f}'


def markdown(df, specs = 'strategies/specs'):
    lines = ['# 策略库回测汇总', '', '由 `observe strategy report` 生成，不要手工编辑。每行是该策略最近一次完成的回测；证据级别一律为「探索」。',
             '回测口径：本项目执行规则集的费用、0.1% 滑点、次日开盘成交、退市持仓按最后估值价折现；与原文的差异逐条列在各策略小节。', '']
    years = sorted(c for c in df.columns if c.startswith('ret_'))
    head = ['编号', '标题', '原型', '状态', '区间', '年化', '最大回撤', '夏普', *[c[4:] for c in years]]
    lines += ['| ' + ' | '.join(head) + ' |', '|' + '---|' * len(head)]
    for _, r in df.sort_values('annual_return', ascending = False, na_position = 'last').iterrows():
        span = f"{r.get('start') or ''}..{r.get('end') or ''}" if r.get('start') else ''
        cells = [r['id'], r['title'], r['archetype'], r['run_status'], span, _pct(r.get('annual_return')), _pct(r.get('max_drawdown')), _num(r.get('sharpe')),
                 *[_pct(r.get(c)) for c in years]]
        lines.append('| ' + ' | '.join(str(c).replace('|', '/') for c in cells) + ' |')
    lines += ['', '## 各策略实现思路与差异', '']
    for _, r in df.sort_values('id').iterrows():
        spec = yaml.safe_load(Path(r['spec']).read_text(encoding = 'utf-8'))
        lines += [f"### {r['id']} {r['title']}", '', f"- 来源：`{spec['source']['file']}`" + (f"（{spec['source']['url']}）" if spec['source'].get('url') else ''),
                  f"- 规格：`{r['spec']}`；最近实验：`{r['run_id'] or '未运行'}`", '', spec['idea'].strip(), '']
        if spec.get('deviations'): lines += ['与原文的差异：', ''] + [f'- {d}' for d in spec['deviations']] + ['']
    return '\n'.join(lines).rstrip() + '\n'


def write_report(root, out_csv = 'strategies/results.csv', out_md = 'docs/09-策略库回测汇总.md', **kw):
    df = collect(root, **kw)
    Path(out_csv).parent.mkdir(parents = True, exist_ok = True); df.to_csv(out_csv, index = False, encoding = 'utf-8', lineterminator = '\n')
    Path(out_md).write_bytes(markdown(df, kw.get('specs', 'strategies/specs')).encode('utf-8'))
    return {'strategies': len(df), 'run': int((df.run_status != 'not_run').sum()) if len(df) else 0, 'csv': out_csv, 'markdown': out_md}
