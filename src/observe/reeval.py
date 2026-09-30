"""只读重新评价（评价口径修正后）：复用已有成对实验（paired 目录）里保存的预测、标签、因子、股票池、切分与净值，
用修正后的统一评价函数重新计算，结果写到**新目录**并引用原实验编号与原产物哈希；不重新训练、不重新下载、不改动任何源文件。

内容：旧新指标逐项对照（哪些没变、哪些变了、原因）；成对预测差异；固定原名单的前 N 名及覆盖；开发区间因子分析的留出边界影响（分实现与边界两步归因）；
多个实验在共同测试日期上的比较；选参重放（证明修复没有改变训练选择，因此预测不受影响）；组合阻断与前缀诊断。"""
import hashlib, json
from pathlib import Path

import pandas as pd

from .dataset import cross_sectional_preprocess, dev_labels
from .evaluation.paired import diff_stats
from .evaluation.ranking import rank_ic_table, spearman, topn_table
from .features import load_factor_set
from .models import day_weights
from .paired import OK, PairedConfig, evaluate
from .replay import ReproduceRefused, _read
from .research import ResearchConfig, _factor_eval, selection_stage
from .runs import RunStatus, canonical, create_run_dir, environment, file_sha, write_json

ARM_INPUTS = ('predictions.parquet', 'labels.parquet', 'factors.parquet', 'universe.parquet', 'split_plan.parquet', 'factor_set.yaml', 'model_eval.json', 'factor_eval.json', 'config.json')
KEY = ['decision_date', 'instrument']


def _verify(run):
    """源目录的 manifest 记录的每个文件都在、哈希一致，否则拒绝（不在被篡改或不完整的产物上做评价）"""
    m = _read(Path(run) / 'manifest.json'); bad = sorted(n for n, sha in m.get('files', {}).items() if not (Path(run) / n).exists() or file_sha(Path(run) / n) != sha)
    if bad: raise ReproduceRefused(f'{run} 的产物与 manifest 不一致：{bad[:5]}')
    return m


def load_source(path):
    src = Path(path); m = _verify(src); doc = _read(src / 'config.json')
    if doc.get('kind') != 'paired': raise ReproduceRefused(f'{src} 不是成对实验目录')
    sub = _read(src / 'subruns.json'); root = src.parent
    for item in list(sub['arms'].values()) + sub['backtests']: item['output'] = str(root / item['run_id'])          # 按编号在同一实验根目录下解析，不依赖当时的工作目录
    for item in list(sub['arms'].values()) + sub['backtests']: _verify(item['output'])
    cfg = PairedConfig.model_validate(doc['config']); plan = pd.read_parquet(Path(sub['arms']['base']['output']) / 'split_plan.parquet')
    return {'dir': src, 'doc': doc, 'cfg': cfg, 'sub': sub, 'plan': plan, 'old': _read(src / 'paired_eval.json'), 'manifest': m}


def _sha_inputs(s):
    return {'paired_dir': s['dir'].name, 'config_hash': _read(s['dir'] / 'status.json').get('config_hash'), 'manifest_sha256': file_sha(s['dir'] / 'manifest.json'), 'paired_eval_sha256': file_sha(s['dir'] / 'paired_eval.json'),
            'arms': {k: {'run_id': v['run_id'], **{n: file_sha(Path(v['output']) / n) for n in ARM_INPUTS}} for k, v in s['sub']['arms'].items()},
            'backtests': [{'model': b['model'], 'arm': b['arm'], 'run_id': b['run_id'], 'status': b['status'], 'equity_sha256': file_sha(Path(b['output']) / 'equity.json') if (Path(b['output']) / 'equity.json').exists() else None,
                           'status_sha256': file_sha(Path(b['output']) / 'status.json')} for b in s['sub']['backtests']]}


def _holdout(plan):
    h = plan.holdout_start.iloc[0]; return None if pd.isna(h) else pd.Timestamp(h).date()


# 旧新对照 ------------------------------------------------------------------------------------------------
def _pull(ev):
    """paired_eval 里可对照的标量：{路径: 值}"""
    out = {}
    for m, v in ev.get('models', {}).items():
        for sec in ('rank_ic', next((k for k in v if k.endswith('_mean_label')), 'rank_ic')):
            d = v.get(sec) or {}
            for f in ('a_mean', 'b_mean', 'diff_mean'): out[f'models/{m}/{sec}/{f}'] = d.get(f)
            for i, f in enumerate(('lo', 'hi')): out[f'models/{m}/{sec}/diff_ci95_{f}'] = (d.get('diff_ci95') or [None, None])[i]
            out[f'models/{m}/{sec}/days'] = d.get('days')
    out['valid_primary_comparison'] = ev.get('valid_primary_comparison')
    for n, v in ev.get('new_factors', {}).items():
        out[f'new_factors/{n}/rank_ic_mean'] = v.get('rank_ic_mean'); out[f'new_factors/{n}/direction_adjusted_ic'] = v.get('direction_adjusted_ic')
        for i, f in enumerate(('lo', 'hi')): out[f'new_factors/{n}/rank_ic_ci95_{f}'] = (v.get('rank_ic_ci95') or [None, None])[i]
    for m, slot in ev.get('portfolio', {}).items():
        for arm, r in slot['runs'].items(): out[f'portfolio/{m}/{arm}/status'] = r.get('status'); out[f'portfolio/{m}/{arm}/total_return'] = r.get('total_return')
        pair = slot.get('pair') or slot.get('diagnostic_prefix') or {}
        out[f'portfolio/{m}/{"pair" if "pair" in slot else "diagnostic_prefix"}/daily_diff_mean'] = (pair.get('daily_return') or {}).get('diff_mean')
    return out


REASONS = {'rank_ic': '联合有效配对的秩 IC；本样本预测与标签均为有限值，预期不变', 'top': '前 N 名名单先由预测固定再关联标签（旧：先滤掉无效标签再取前 N）',
           'new_factors': '联合配对 + 开发区间标签成熟规则（旧：因子行先分别排名、只按决策日过滤）', 'portfolio': '组合指标来自原回测产物，未重算；阻断的运行不再列绩效，阻断前的共同区间只作诊断前缀'}


def compare_old_new(old, new, tol = 1e-12):
    a, b = _pull(old), _pull(new); rows = []
    for k in sorted(set(a) | set(b)):
        x, y = a.get(k), b.get(k); sec = 'portfolio' if k.startswith('portfolio') else 'new_factors' if k.startswith('new_factors') else 'rank_ic' if '/rank_ic/' in k else 'top'
        same = (x == y) or (isinstance(x, (int, float)) and isinstance(y, (int, float)) and abs(x - y) <= tol)
        rows.append({'metric': k, 'old': x, 'new': y, 'changed': not same, 'delta': None if same or not all(isinstance(v, (int, float)) for v in (x, y)) else y - x, 'reason': None if same else REASONS[sec]})
    by = {}
    for r in rows:
        sec = r['metric'].split('/')[0] + ('/' + r['metric'].split('/')[2] if r['metric'].startswith('models') else ''); d = by.setdefault(sec, {'metrics': 0, 'changed': 0, 'max_abs_delta': 0.0})
        d['metrics'] += 1; d['changed'] += int(r['changed']); d['max_abs_delta'] = max(d['max_abs_delta'], abs(r['delta']) if r['delta'] is not None else 0.0)
    return {'summary': by, 'rows': rows}


# 每日序列、共同测试日期 ---------------------------------------------------------------------------------
def daily_series(s):
    """{(臂, 模型): {'ic': 序列, 'top': 序列}}：预测 × 开发区间可用标签，与成对评价同一实现"""
    lab = pd.read_parquet(Path(s['sub']['arms']['base']['output']) / 'labels.parquet'); labd = dev_labels(lab, _holdout(s['plan'])); y = labd[KEY + ['value']]; out, tops = {}, {}
    for arm, item in s['sub']['arms'].items():
        pred = pd.read_parquet(Path(item['output']) / 'predictions.parquet')
        for model, g in pred.groupby('model_id'):
            t = topn_table(g, labd, s['cfg'].models.top_n); tops[(arm, model)] = t
            out[(arm, model)] = {'ic': rank_ic_table(g.merge(y, on = KEY, how = 'left'), min_n = s['cfg'].models.min_names).ic, 'top': t.mean_label}
    return out, tops


def _returns(equity):
    e = pd.DataFrame(equity).set_index('date').equity.astype(float); e.index = pd.to_datetime(e.index).date; return e.pct_change().dropna()


def common_dates(sources, series, block, n_boot, seed):
    """各实验测试日期的交集上：实验内 B−A、同一臂在两个实验之间的差（两个实验只差训练窗或惩罚口径时即该因素的效应）、两个实验的 B−A 之差"""
    labels = list(sources); dates = [set(pd.Index(series[l][0][('base', 'ridge')]['ic'].index)) for l in labels]; common = pd.Index(sorted(set.intersection(*dates)))
    out = {'labels': labels, 'common_days': int(len(common)), 'first': str(common.min()) if len(common) else None, 'last': str(common.max()) if len(common) else None,
           'test_days_by_experiment': {l: int(len(d)) for l, d in zip(labels, dates)}, 'within_experiment': {}, 'between_experiment_arm_difference': {}, 'effect_difference': {}}
    for model in ('ridge', 'equal_blend', 'single_factor'):
        for metric in ('ic', 'top'):
            for l in labels:
                a, b = (series[l][0][(arm, model)][metric].reindex(common) for arm in ('base', 'extended'))
                out['within_experiment'].setdefault(model, {}).setdefault(metric, {})[l] = diff_stats(a, b, block, n_boot, seed, common, 2 * block)
            if len(labels) == 2:
                l0, l1 = labels
                for arm in ('base', 'extended'):
                    a, b = (series[l][0][(arm, model)][metric].reindex(common) for l in (l1, l0))          # B 项 = 第一个实验，A 项 = 第二个实验：差 = 第一个 − 第二个
                    out['between_experiment_arm_difference'].setdefault(model, {}).setdefault(metric, {})[arm] = {'first_minus_second': f'{l0} − {l1}', **diff_stats(a, b, block, n_boot, seed, common, 2 * block)}
                d = [(series[l][0][('extended', model)][metric].reindex(common) - series[l][0][('base', model)][metric].reindex(common)) for l in labels]
                out['effect_difference'].setdefault(model, {}).setdefault(metric, diff_stats(d[1], d[0], block, n_boot, seed, common, 2 * block) | {'note': f'加分钟效应（B−A）：{labels[0]} 减 {labels[1]}'})
    return out, common


def portfolio_common(sources, common, block, n_boot, seed):
    out = {}
    for model in ('equal_blend', 'ridge'):
        for l, s in sources.items():
            runs = {b['arm']: b for b in s['sub']['backtests'] if b['model'] == model}
            if set(runs) != {'base', 'extended'} or not all((Path(b['output']) / 'equity.json').exists() for b in runs.values()): continue
            r = {arm: _returns(_read(Path(b['output']) / 'equity.json')) for arm, b in runs.items()}; blocked = [b for b in runs.values() if b['status'] not in OK]
            bad = []
            for b in blocked:
                iss = _read(Path(b['output']) / 'status.json').get('issues') or []; bad += [i['date'] for i in iss]
            until = pd.Timestamp(min(bad)).date() if bad else None
            grid = pd.Index([d for d in common if until is None or d < until])
            a, b_ = (r['base'].reindex(grid), r['extended'].reindex(grid)); comp = lambda x: float((1 + x.dropna()).prod() - 1)
            entry = {'common_days': int(len(common)), 'evaluable_days': int(a.notna().sum()), 'excluded_days': int(len(common) - a.notna().sum()), 'base_total_return': comp(a), 'extended_total_return': comp(b_),
                     'daily_return': diff_stats(a, b_, block, n_boot, seed, grid), 'valid_portfolio_comparison': not blocked}
            if blocked: entry['diagnostic_prefix'] = True; entry['truncation_reason'] = '账本阻断（持仓退市）：' + ','.join(f"{b['arm']}@{min(bad)}" for b in blocked); entry['full_period_metrics'] = 'unavailable'
            out.setdefault(model, {})[l] = entry
    return out


# 选参重放、留出边界 ------------------------------------------------------------------------------------
def replay_selection(arm_dir):
    """用保存的因子、标签、切分重放选参阶段（不写任何产物、不做最终重拟合与预测），与原实验保存的验证秩 IC 逐项对照。
    一致 ⇒ 修复没有改变候选打分与 alpha 选择 ⇒ 训练结果与预测不受影响"""
    a = Path(arm_dir); cfg = ResearchConfig.model_validate(_read(a / 'config.json')['config']); fset = load_factor_set(a / 'factor_set.yaml')
    names = [f['name'] for f in fset['factors']]; directions = {f['name']: f['direction'] for f in fset['factors']}
    fac, lab, plan = (pd.read_parquet(a / f'{n}.parquet') for n in ('factors', 'labels', 'split_plan'))
    X = cross_sectional_preprocess(fac.assign(eligible = True), names).drop(columns = 'eligible'); saved = _read(a / 'model_eval.json')['selection']; rows, scale = [], []
    for sp in plan.itertuples(index = False):
        kept, _, sel, val, _, candidates, tried, best = selection_stage(sp, X, lab, names, directions, cfg.models)
        scale.append(fit_scale(sp.split_id, sel, val, kept, candidates, tried, best))
        old = {x['candidate']: x['valid_rank_ic'] for x in saved if x['split_id'] == sp.split_id}
        rows += [{'split_id': int(sp.split_id), 'candidate': t['candidate'], 'old': old.get(t['candidate']), 'new': t['valid_rank_ic'],
                  'abs_diff': None if old.get(t['candidate']) is None or t['valid_rank_ic'] is None else abs(old[t['candidate']] - t['valid_rank_ic'])} for t in tried]
        rows.append({'split_id': int(sp.split_id), 'candidate': 'selected_ridge', 'old': _read(a / 'models' / f'split{sp.split_id:02d}_ridge.json')['info']['selected'], 'new': best, 'abs_diff': None})
    diffs = [r['abs_diff'] for r in rows if r['abs_diff'] is not None]
    return {'fit_scale': scale, 'candidates_compared': len(diffs), 'max_abs_diff': max(diffs) if diffs else None, 'selected_alpha_same': all(r['old'] == r['new'] for r in rows if r['candidate'] == 'selected_ridge'),
            'identical': bool(all(d == 0 for d in diffs) and all(r['old'] == r['new'] for r in rows if r['candidate'] == 'selected_ridge')), 'rows': rows}


def fit_scale(split_id, sel, val, kept, candidates, tried, best):
    """一个拟合阶段的正则强度记录：每天总样本权重为 1，所以 sum_weight = 选参样本的决策日数；alpha / sum_weight 才是与样本量无关的相对强度。
    另给验证曲线，以及验证窗上相邻 alpha 的预测在每日横截面上的平均秩相关（趋于 1 说明再加大 alpha 排名不再变化）"""
    w = day_weights(sel.date); ridge = [(k, m) for k, m in candidates if k.startswith('ridge')]; score = {k: m.predict(val[m.features]) for k, m in ridge}
    def stab(k1, k2):
        f = val[['date']].assign(x = score[k1], y = score[k2]); v = f.groupby('date').apply(lambda g: spearman(g.x, g.y), include_groups = False); return _mean(v)
    curve = [{'candidate': k, 'alpha': m.penalty['alpha_used'], 'alpha_over_sum_weight': m.penalty['alpha_used'] / float(w.sum()), 'valid_rank_ic': next(t['valid_rank_ic'] for t in tried if t['candidate'] == k)} for k, m in ridge]
    return {'split_id': int(split_id), 'sum_weight': float(w.sum()), 'decision_days': int(sel.date.nunique()), 'rows': int(len(sel)), 'features': len(kept), 'selected': best,
            'selected_at_grid_edge': best == ridge[-1][0], 'curve': curve, 'adjacent_alpha_rank_agreement': [{'from': ridge[i][0], 'to': ridge[i + 1][0], 'mean_daily_spearman': stab(ridge[i][0], ridge[i + 1][0])} for i in range(len(ridge) - 1)]}


def _mean(v):
    v = pd.Series(v).dropna(); return None if not len(v) else float(v.mean())


def boundary_effect(arm_dir, plan):
    """开发区间因子评价：原保存值 → 新实现+旧边界（只看决策日）→ 新实现+新边界；并列出曾被诊断使用的边界标签"""
    a = Path(arm_dir); fset = load_factor_set(a / 'factor_set.yaml'); names = [f['name'] for f in fset['factors']]; dirs = {f['name']: f['direction'] for f in fset['factors']}
    fac, lab = pd.read_parquet(a / 'factors.parquet'), pd.read_parquet(a / 'labels.parquet'); H = _holdout(plan)
    dev = sorted(d for d in set(fac.date) if H is None or pd.Timestamp(d).date() < H); min_n = ResearchConfig.model_validate(_read(a / 'config.json')['config']).models.min_names
    old = _read(a / 'factor_eval.json')['factors']; mid = _factor_eval(fac, lab, names, dirs, dev, min_n, None)['factors']; new = _factor_eval(fac, lab, names, dirs, dev, min_n, H)['factors']
    used = lab[lab.valid.astype(bool) & (pd.to_datetime(lab.decision_date) < pd.Timestamp(H)) & (pd.to_datetime(lab.matured_at) >= pd.Timestamp(H))] if H is not None else lab.iloc[:0]
    rows = [{'factor': n, 'saved_old': old[n]['rank_ic_mean'], 'new_impl_old_boundary': mid[n]['rank_ic_mean'], 'new_impl_new_boundary': new[n]['rank_ic_mean'],
             'implementation_effect': None if old[n]['rank_ic_mean'] is None else mid[n]['rank_ic_mean'] - old[n]['rank_ic_mean'], 'boundary_effect': new[n]['rank_ic_mean'] - mid[n]['rank_ic_mean']} for n in names]
    return {'holdout_start': None if H is None else str(H), 'boundary_labels_used_by_saved_diagnostics': {'label_rows': int(len(used)), 'decision_dates': sorted({str(pd.Timestamp(d).date()) for d in used.decision_date}),
            'exit_dates': sorted({str(pd.Timestamp(d).date()) for d in used.matured_at})}, 'factors': rows, 'max_abs_implementation_effect': max((abs(r['implementation_effect']) for r in rows if r['implementation_effect'] is not None), default=None),
            'max_abs_boundary_effect': max(abs(r['boundary_effect']) for r in rows)}


def attribution_check(s):
    """逐个回测核对持仓归因：旧代码取「positions_daily 的最大日期」作期末持仓，新代码取净值日历的截止日；两者是否相同，以及与账户净值勾稽的残差各由什么构成"""
    from .diagnose import attribute_run       # diagnose 依赖本模块的读取函数，这里延迟导入避免循环
    out = []
    for b in s['sub']['backtests']:
        d = Path(b['output']); bad = [i['date'] for i in (_read(d / 'status.json').get('issues') or [])]; until = min(bad) if bad else None
        pos = sorted({x['date'] for x in _read(d / 'positions_daily.json') if until is None or x['date'] < until}); r = attribute_run(d, None, until, s['cfg'].initial_cash)
        row = {'model': b['model'], 'arm': b['arm'], 'status': b['status'], 'until_exclusive': until, 'legacy_last_position_date': pos[-1] if pos else None, 'equity_calendar_end': r.get('end')}
        row['same_cutoff'] = row['legacy_last_position_date'] == row['equity_calendar_end']
        if 'reconciliation' in r:
            rc = r['reconciliation']; row.update(holdings_at_end = r['holdings_at_end'], account_equity_change = rc['account_equity_change'], allocated_to_instruments = rc['allocated_to_instruments'],
                                                 identified_not_allocated = rc['identified_not_allocated'], residual_unexplained = rc['residual_unexplained'], reconciled = rc['reconciled'])
        else: row['insufficient_data'] = r.get('insufficient_data')
        out.append(row)
    return out


# 入口 ---------------------------------------------------------------------------------------------------
def reevaluate(root, paths, output = None, runs_root = None):
    """paths：{标签: 成对实验目录}，第一个是主实验。写到新的实验目录；源目录只读"""
    sources = {l: load_source(p) for l, p in paths.items()}; first = next(iter(sources.values()))
    doc = {'kind': 'reeval', 'sources': {l: _sha_inputs(s) for l, s in sources.items()}, 'environment': environment(), 'note': '只读复用原实验产物；评价代码为修正后的 evaluation.ranking / dataset.dev_labels'}
    h = hashlib.sha256(json.dumps(canonical(doc['sources']), sort_keys = True).encode()).hexdigest()
    out = create_run_dir(Path(runs_root) if runs_root else Path(root) / 'runs', output, f'{h[:6]}-reeval'); status = RunStatus(out, out.name, kind = 'reeval', evidence = 'exploratory', config_hash = h)
    write_json(out / 'config.json', doc); status.stage('config')
    block = max(20, 4 * first['cfg'].label_h); nb, seed = first['cfg'].n_boot, first['cfg'].seed; result = {'sources': {l: s['dir'].name for l, s in sources.items()}}
    try:
        series = {l: daily_series(s) for l, s in sources.items()}; new = {}
        for l, s in sources.items():
            new[l] = canonical(evaluate(s['cfg'], s['sub'], s['plan'])); write_json(out / f'paired_eval_new_{l}.json', new[l])
        status.stage('evaluate')
        result['old_vs_new'] = {l: compare_old_new(s['old'], new[l]) for l, s in sources.items()}
        result['pairing'] = {l: {m: v['pairing'] for m, v in new[l]['models'].items()} for l in sources}
        result['top_n_fixed_list'] = {l: {m: {'coverage_a': v[f'top{s["cfg"].models.top_n}_mean_label']['coverage_a'], 'coverage_b': v[f'top{s["cfg"].models.top_n}_mean_label']['coverage_b']} for m, v in new[l]['models'].items()} for l, s in sources.items()}
        rows = []
        for l in sources:
            for (arm, model), t in series[l][1].items(): rows.append(t.reset_index().assign(experiment = l, arm = arm, model_id = model, selected_names = lambda d: d.selected_names.map(json.dumps), invalid_reason_counts = lambda d: d.invalid_reason_counts.map(json.dumps)))
        pd.concat(rows, ignore_index = True).to_parquet(out / 'topn_daily.parquet', index = False)
        result['holdout_boundary'] = {l: {arm: boundary_effect(item['output'], s['plan']) for arm, item in s['sub']['arms'].items()} for l, s in sources.items()}
        result['selection_replay'] = {l: {arm: replay_selection(item['output']) for arm, item in s['sub']['arms'].items()} for l, s in sources.items()}; status.stage('diagnostics')
        cd, common = common_dates(sources, series, block, nb, seed); result['common_test_dates'] = cd; result['portfolio_common'] = portfolio_common(sources, common, block, nb, seed)
        result['attribution_check'] = {l: attribution_check(s) for l, s in sources.items()}
        result['portfolio_status'] = {l: {m: {'runs': {a: {k: v for k, v in r.items() if k in ('run_id', 'status', 'valid', 'blocked_from', 'blocked_kinds', 'blocked_instruments')} for a, r in slot['runs'].items()},
                                             'full_period_metrics': slot.get('full_period_metrics', 'available' if 'pair' in slot else 'n/a'), 'diagnostic_prefix': slot.get('diagnostic_prefix')} for m, slot in new[l]['portfolio'].items()} for l in sources}
    except Exception as exc:
        status.finish('failed', error = f'{type(exc).__name__}: {exc}'); raise
    write_json(out / 'reeval.json', canonical(result)); status.finish('success', summary = {'sources': result['sources'], 'common_days': result['common_test_dates']['common_days']})
    files = {p.relative_to(out).as_posix(): file_sha(p) for p in sorted(out.rglob('*')) if p.is_file() and p.name != 'manifest.json'}
    write_json(out / 'manifest.json', {'run_id': out.name, 'kind': 'reeval', 'status': 'success', 'environment': doc['environment'], 'files': files})
    return {'run_id': out.name, 'output': str(out), 'status': 'success', 'sources': result['sources']}
