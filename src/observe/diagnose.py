"""分钟特征成对实验的有限诊断（只读）：缺失模式、缺失与打分变化的分解、集中持仓贡献。结果写到新目录，引用原实验编号与哈希，不改动源目录。

1. 缺失模式：哪些候选证券从未有分钟数据、是否与后来退市 / 合并有关、缺失行的标签画像、它们在两侧前 N 名里出现的次数与标签；缺失标记本身的秩 IC（探针，只用来识别
   「数据可用性」带来的信号，回溯的文件缺失不能变成正式预测特征）。
2. 打分变化的分解（用保存的模型系数重算，不重新拟合）：B = 加分钟侧的分数；b = 把分钟因子全部置 0 的同一批系数（只剩日频因子）；A = 基础侧的分数。
   b − A = 拟合扰动（含分钟因子拟合后日频因子系数的变化），B − b = 分钟因子数值本身的贡献（缺失行在 B 与 b 中相同）。
3. 集中持仓贡献：两侧回测按证券的盈亏（成交现金流 + 共同截止日的持仓市值；截止日来自净值日历，空仓则市值为零），与账户净值勾稽：已分配、已识别未分配（分红入账、应收等）、残差；未完全勾稽时比例标为部分归因。"""
from pathlib import Path

import numpy as np
import pandas as pd

from .data.store import Store
from .dataset import cross_sectional_preprocess, dev_labels
from .evaluation.paired import block_ci, diff_stats
from .evaluation.ranking import rank_ic_table, topn_table
from .features import load_factor_set
from .paired import OK
from .reeval import _holdout, _sha_inputs, load_source
from .replay import _read
from .runs import RunStatus, canonical, create_run_dir, environment, file_sha, write_json

KEY = ['decision_date', 'instrument']


def _f(x): return None if x is None or not np.isfinite(x) else float(x)


# 1. 缺失模式 ----------------------------------------------------------------------------------------------
def missingness(root, s):
    ext, base = (Path(s['sub']['arms'][k]['output']) for k in ('extended', 'base')); cfg = s['cfg']; H = _holdout(s['plan'])
    uni = pd.read_parquet(ext / 'universe.parquet'); elig = uni[uni.eligible][KEY].copy(); intr = pd.read_parquet(ext / 'intraday.parquet')[['date', 'instrument', 'n_bars']]
    elig['date'] = pd.to_datetime(elig.decision_date).dt.date; intr['date'] = pd.to_datetime(intr.date).dt.date
    j = elig.merge(intr, on = ['date', 'instrument'], how = 'left'); j['has'] = j.n_bars.notna()
    per = j.groupby('instrument').has.mean(); none = sorted(per.index[per == 0]); state = Store(root).state(cfg.snapshot); ins = Store(root).load_state(state, 'instruments')
    ins = ins.drop_duplicates('instrument', keep = 'last').set_index('instrument').reindex(per.index); last = pd.to_datetime(j.decision_date).max()
    delist = pd.to_datetime(ins.delist_date); gone = delist.notna() & (delist <= last)
    cross = pd.DataFrame({'delisted_or_merged_in_span': gone, 'no_minute_data': per == 0}); tab = cross.groupby(['delisted_or_merged_in_span', 'no_minute_data']).size()
    by_year = pd.DataFrame({'year': delist[gone].dt.year, 'no_minute': (per == 0)[gone]}).groupby('year').no_minute.agg(['size', 'sum']).rename(columns = {'size': 'delisted', 'sum': 'no_minute_data'})
    labd = dev_labels(pd.read_parquet(ext / 'labels.parquet'), H); jl = j.merge(labd[KEY + ['value', 'valid', 'invalid_reason']], on = KEY, how = 'left'); jl['group'] = np.where(jl.instrument.isin(none), 'no_minute', 'has_minute')
    jd = jl[pd.to_datetime(jl.decision_date) < pd.Timestamp(H)] if H is not None else jl        # 画像只看开发区间：留出期的标签一律不可用，会稀释有效比例
    prof = {g: {'rows': int(len(x)), 'instruments': int(x.instrument.nunique()), 'valid_label_share': _f(x.valid.fillna(False).astype(bool).mean()), 'mean_label': _f(x.value.mean()), 'median_label': _f(x.value.median()),
                'share_label_below_minus_10pct': _f((x.value < -0.10).sum() / max(x.value.notna().sum(), 1)), 'invalid_reasons': {k: int(v) for k, v in x[~x.valid.fillna(False).astype(bool)].invalid_reason.fillna('no_label').value_counts().items()}}
            for g, x in jd.groupby('group')}
    slots = {}
    for arm, d in (('base', base), ('extended', ext)):
        pred = pd.read_parquet(d / 'predictions.parquet')
        for model in ('ridge', 'equal_blend'):
            t = topn_table(pred, labd, cfg.models.top_n, model); names = [n for lst in t.selected_names for n in lst]; hit = [n in set(none) for n in names]
            lab_map = labd.set_index(KEY).value; vals = [lab_map.get((dt, n), np.nan) for dt, lst in t.selected_names.items() for n in lst]
            v = pd.Series(vals)[pd.Series(hit)]
            slots.setdefault(model, {})[arm] = {'list_slots': len(names), 'no_minute_slots': int(sum(hit)), 'no_minute_share': _f(sum(hit) / max(len(names), 1)), 'distinct_no_minute_names': int(len({n for n, h in zip(names, hit) if h})),
                                                'mean_valid_label_in_those_slots': _f(v.mean()), 'valid_labels_in_those_slots': int(v.notna().sum())}
    probe = {}
    dev = jd
    t = rank_ic_table(dev.assign(score = dev.has.astype(float)), label = 'value', min_n = cfg.models.min_names); ic = t.ic; grid = pd.Index(sorted(set(dev.decision_date)))
    probe = {'definition': '分数 = 当日是否有分钟数据（有 = 1）；秩 IC 为正表示有分钟数据的证券随后收益更高。只是数据可用性探针，不是特征', 'days_defined': int(ic.notna().sum()), 'days': int(len(grid)),
             'rank_ic_mean': _f(ic.mean()), 'rank_ic_ci95': block_ci(ic.reindex(grid).to_numpy(), max(20, 4 * cfg.label_h), cfg.n_boot, cfg.seed), 'rows_missing_share': _f(1 - dev.has.mean())}
    return {'candidates': int(len(per)), 'no_minute_instruments': none, 'crosstab_delisted_vs_no_minute': {f'delisted={a},no_minute={b}': int(n) for (a, b), n in tab.items()},
            'delisted_by_year': {int(y): {k: int(v) for k, v in r.items()} for y, r in by_year.iterrows()}, 'label_profile': prof, 'top_n_slots': slots, 'missing_marker_probe': probe,
            'note': '缺失只出现在区间内退市 / 合并的证券上：外部文件保留了哪些退市证券由其保留机制决定，回溯而非随机'}


# 2. 打分变化分解 ------------------------------------------------------------------------------------------
def score_decomposition(s):
    ext, base = (Path(s['sub']['arms'][k]['output']) for k in ('extended', 'base')); cfg = s['cfg']; H = _holdout(s['plan'])
    fs_e, fs_b = load_factor_set(ext / 'factor_set.yaml'), load_factor_set(base / 'factor_set.yaml'); names = [f['name'] for f in fs_e['factors']]; minute = [n for n in names if n not in {f['name'] for f in fs_b['factors']}]
    fac = pd.read_parquet(ext / 'factors.parquet'); X = cross_sectional_preprocess(fac.assign(eligible = True), names).drop(columns = 'eligible'); X['date'] = pd.to_datetime(X.date).dt.date
    labd = dev_labels(pd.read_parquet(ext / 'labels.parquet'), H); y = labd[KEY + ['value']]; pa, pb = pd.read_parquet(base / 'predictions.parquet'), pd.read_parquet(ext / 'predictions.parquet'); plan = s['plan']
    none = set(missingness_names(ext)); out = {}; block = max(20, 4 * cfg.label_h); grid = pd.Index(sorted(set(pb.decision_date)))
    for model in ('ridge', 'equal_blend'):
        parts = []
        for sp in plan.itertuples():
            m = _read(ext / 'models' / f'split{sp.split_id:02d}_{model}.json'); feats = m['features']
            test = X[(X.date >= pd.Timestamp(sp.test_start).date()) & (X.date <= pd.Timestamp(sp.test_end).date())][['date', 'instrument'] + feats].copy(); test = test.rename(columns = {'date': 'decision_date'})
            def score(F):
                if model == 'ridge': return F[feats].to_numpy(float) @ np.array([m['coef'][f] for f in feats]) + m['intercept']
                return F[feats].to_numpy(float) @ np.array([m['directions'][f] for f in feats], float) / len(feats)
            blank = test.copy(); blank[[f for f in feats if f in minute]] = 0.0
            parts.append(pd.DataFrame({'decision_date': test.decision_date.to_numpy(), 'instrument': test.instrument.to_numpy(), 'B_rebuilt': score(test), 'b': score(blank)}))
        R = pd.concat(parts, ignore_index = True); A = pa[pa.model_id == model][KEY + ['score']].rename(columns = {'score': 'A'}); B = pb[pb.model_id == model][KEY + ['score']].rename(columns = {'score': 'B'})
        J = R.merge(A, on = KEY, how = 'outer').merge(B, on = KEY, how = 'outer').merge(y, on = KEY, how = 'left')
        ic = {c: rank_ic_table(J.assign(score = J[c]), min_n = cfg.models.min_names).ic for c in ('A', 'B', 'b')}
        keep = ~J.instrument.isin(none); ics = {c: rank_ic_table(J[keep].assign(score = J.loc[keep, c]), min_n = cfg.models.min_names).ic for c in ('A', 'B', 'b')}
        pct = J.assign(**{c: J.groupby('decision_date')[c].rank(pct = True) for c in ('A', 'B', 'b')}); g = pct[pct.instrument.isin(none)]
        out[model] = {'rebuild_max_abs_diff': _f((J.B_rebuilt - J.B).abs().max()), 'rows': int(len(J)), 'rows_missing_any_score': int(J[['A', 'B', 'b']].isna().any(axis = 1).sum()),
                      'rank_ic_mean': {c: _f(ic[c].mean()) for c in ic},
                      'total_B_minus_A': diff_stats(ic['A'], ic['B'], block, cfg.n_boot, cfg.seed, grid), 'fit_perturbation_b_minus_A': diff_stats(ic['A'], ic['b'], block, cfg.n_boot, cfg.seed, grid),
                      'minute_values_B_minus_b': diff_stats(ic['b'], ic['B'], block, cfg.n_boot, cfg.seed, grid),
                      'complete_coverage_subsample': {'note': '剔除 25 只从未有分钟数据的证券后的受限诊断；剔除依据（后来退市或被合并）用到了事后信息，不能作为主结果', 'rank_ic_mean': {c: _f(ics[c].mean()) for c in ics},
                                                      'total_B_minus_A': diff_stats(ics['A'], ics['B'], block, cfg.n_boot, cfg.seed, grid), 'fit_perturbation_b_minus_A': diff_stats(ics['A'], ics['b'], block, cfg.n_boot, cfg.seed, grid),
                                                      'minute_values_B_minus_b': diff_stats(ics['b'], ics['B'], block, cfg.n_boot, cfg.seed, grid)},
                      'no_minute_names_mean_score_percentile': {c: _f(g[c].mean()) for c in ('A', 'b', 'B')}, 'no_minute_rows': int(len(g)),
                      'top_n_slots_no_minute': {c: int(J[J.instrument.isin(none)].merge(J.sort_values(['decision_date', c, 'instrument'], ascending = [True, False, True]).groupby('decision_date').head(cfg.models.top_n)[KEY], on = KEY).shape[0]) for c in ('A', 'b', 'B')}}
    return {'minute_factors': minute, 'definition': 'A = 基础侧分数；B = 加分钟侧分数；b = 加分钟侧同一批系数、分钟因子置 0。b − A 是拟合扰动，B − b 是分钟因子数值的贡献', 'models': out}


def missingness_names(ext):
    uni = pd.read_parquet(Path(ext) / 'universe.parquet'); e = uni[uni.eligible]; intr = pd.read_parquet(Path(ext) / 'intraday.parquet')[['date', 'instrument', 'n_bars']]
    e = e.assign(date = pd.to_datetime(e.decision_date).dt.date); intr = intr.assign(date = pd.to_datetime(intr.date).dt.date); j = e.merge(intr, on = ['date', 'instrument'], how = 'left')
    per = j.groupby('instrument').n_bars.apply(lambda x: x.notna().mean()); return sorted(per.index[per == 0])


# 3. 集中持仓贡献 ------------------------------------------------------------------------------------------
TOL = 1.0      # 勾稽容差（元）：现金流水四舍五入到分，逐笔累计后的差不应超过它


def _table(rows, cols):
    df = pd.DataFrame(rows, columns = None if len(rows) else cols)
    for c in cols:
        if c not in df: df[c] = np.nan       # 旧产物的现金流水没有 instrument 列等：补空列，避免空表或缺列时报 AttributeError / KeyError
    return df


def attribute_run(run, end = None, until = None, initial = None):
    """一个回放运行按证券的盈亏归因，并与账户净值勾稽。
    截止日 end 来自净值日历（最后一个早于 until 的净值日，或最后一个净值日），不是「最后一条有持仓的记录」；所有成交、现金、持仓、应收都按 end 对齐。
    截止日的持仓取 positions_daily 中恰好等于 end 的行：没有行且净值市值为 0 才算空仓；没有行而净值市值大于 0 时无法区分空仓与状态缺失，报数据不足，不猜。
    已分配 = 每只证券的（买入 + 卖出 + 费用）现金流水 + 截止日持仓市值；已识别未分配 = 分红入账（旧产物的现金流水不带证券）、期末应收、账户市值与持仓明细之差、其他现金项；
    残差 = 净值变化 − 已分配 − 已识别未分配，应约等于 0"""
    d = Path(run); eq = pd.DataFrame(_read(d / 'equity.json'))
    if not len(eq): return {'insufficient_data': '没有净值记录'}
    days = sorted(eq.date) if until is None else sorted(x for x in eq.date if x < str(until))
    if not days: return {'insufficient_data': f'早于 {until} 没有净值记录'}
    end = str(end) if end is not None else days[-1]
    if end not in set(eq.date): return {'insufficient_data': f'截止日 {end} 不在净值日历里'}
    row = eq[eq.date == end].iloc[0]; init = float(eq.iloc[0].equity) if initial is None else float(initial)
    ce = _table(_read(d / 'cash_events.json'), ['date', 'kind', 'amount', 'instrument']); ce = ce[ce.date <= end]
    pos = _table(_read(d / 'positions_daily.json'), ['date', 'instrument', 'qty', 'last_price']); at_end = pos[pos.date == end]
    if not len(at_end) and float(row.market_value) > TOL: return {'insufficient_data': f'{end} 没有持仓明细，但净值市值为 {row.market_value:.2f}：无法区分空仓与状态缺失', 'end': end}
    trade = ce[ce.kind.isin(['buy', 'sell', 'fee'])]; flow = trade.groupby('instrument').amount.sum() if len(trade) else pd.Series(dtype = float)
    mv = (at_end.qty * at_end.last_price).groupby(at_end.instrument).sum() if len(at_end) else pd.Series(dtype = float)
    pnl = flow.add(mv, fill_value = 0.0); other = ce[~ce.kind.isin(['buy', 'sell', 'fee', 'dividend_paid'])]
    identified = {'dividends_paid_unallocated': float(ce[ce.kind == 'dividend_paid'].amount.sum()), 'receivable_at_end': float(row.receivable),
                  'ledger_market_value_minus_position_detail': float(row.market_value - mv.sum()), 'other_cash_events': float(other.amount.sum())}
    change = float(row.equity) - init; allocated = float(pnl.sum()); residual = change - allocated - sum(identified.values())
    return {'end': end, 'holdings_at_end': 'listed' if len(at_end) else 'flat', 'pnl': pnl, 'reconciliation': {'account_equity_change': change, 'allocated_to_instruments': allocated, 'identified_not_allocated': identified,
            'residual_unexplained': residual, 'reconciled': abs(residual) <= TOL, 'note': '旧产物的分红现金流水不带证券，只能整笔计入「未分配」，不按权重摊平'}}


def _arm_end(run, until):
    days = sorted(x['date'] for x in _read(Path(run) / 'equity.json') if until is None or x['date'] < str(until)); return days[-1] if days else None


def concentration(s, none):
    cfg = s['cfg']; out = {}; none = set(none); init = cfg.initial_cash
    for model in cfg.backtest_models:
        runs = {b['arm']: b for b in s['sub']['backtests'] if b['model'] == model}
        if set(runs) != {'base', 'extended'}: continue
        bad = []
        for b in runs.values():
            if b['status'] not in OK: bad += [i['date'] for i in _read(Path(b['output']) / 'status.json').get('issues') or []]
        until = min(bad) if bad else None; ends = {a: _arm_end(b['output'], until) for a, b in runs.items()}
        if None in ends.values(): out[model] = {'insufficient_data': '一侧在阻断日之前没有净值记录'}; continue
        end = min(ends.values()); att = {a: attribute_run(b['output'], end, None, init) for a, b in runs.items()}       # 双方共同截止日：所有项目按它对齐
        if any('insufficient_data' in x for x in att.values()): out[model] = {'evaluated_through': end, 'insufficient_data': {a: x.get('insufficient_data') for a, x in att.items() if 'insufficient_data' in x}}; continue
        pnl = {a: x['pnl'] for a, x in att.items()}
        common = pnl['base'].index.intersection(pnl['extended'].index); only_b, only_e = pnl['base'].index.difference(pnl['extended'].index), pnl['extended'].index.difference(pnl['base'].index)
        top = lambda p: {'top_gainers': {k: _f(v / init) for k, v in p.nlargest(5).items()}, 'top_losers': {k: _f(v / init) for k, v in p.nsmallest(5).items()},
                         'top5_abs_share': _f(p.abs().nlargest(5).sum() / p.abs().sum()) if len(p) and p.abs().sum() else None, 'names': int(len(p)), 'allocated_pnl_over_initial': _f(p.sum() / init)}
        arms = {a: {**top(p), 'holdings_at_end': att[a]['holdings_at_end'], 'reconciliation_over_initial': {k: (_f(v / init) if not isinstance(v, (dict, str, bool)) else {kk: _f(vv / init) for kk, vv in v.items()} if isinstance(v, dict) else v)
                                                                                                             for k, v in att[a]['reconciliation'].items()}} for a, p in pnl.items()}
        idx = pnl['extended'].index.union(pnl['base'].index); effects = pnl['extended'].reindex(idx).fillna(0) - pnl['base'].reindex(idx).fillna(0)
        unalloc = {a: sum(att[a]['reconciliation']['identified_not_allocated'].values()) for a in att}; complete = all(x['reconciliation']['reconciled'] for x in att.values())
        status = 'unreconciled' if not complete else 'partial' if any(abs(v) > TOL for v in unalloc.values()) else 'complete'
        out[model] = {'evaluated_through': end, 'planned_arm_ends': ends, 'valid_portfolio_comparison': until is None, 'attribution_status': status,
                      'attribution_note': {'complete': '已分配项与账户净值勾稽，无未分配项', 'partial': '部分归因：分红 / 应收等无法分配到证券的项目单独列出，比例按已分配的盈亏计算，不是精确账户贡献', 'unreconciled': '勾稽差额超出容差，归因不可信'}[status],
                      'arms': arms, 'allocated_difference_extended_minus_base_over_initial': _f(effects.sum() / init), 'unallocated_difference_over_initial': _f((unalloc['extended'] - unalloc['base']) / init),
                      'account_equity_difference_over_initial': _f((att['extended']['reconciliation']['account_equity_change'] - att['base']['reconciliation']['account_equity_change']) / init),
                      'decomposition_over_initial': {'common_names': _f((pnl['extended'][common].sum() - pnl['base'][common].sum()) / init), 'extended_only_names': _f(pnl['extended'][only_e].sum() / init),
                                                     'base_only_names': _f(-pnl['base'][only_b].sum() / init)}, 'names_common': int(len(common)), 'names_extended_only': int(len(only_e)), 'names_base_only': int(len(only_b)),
                      'no_minute_names_effect_over_initial': {'total': _f(effects[effects.index.isin(none)].sum() / init), 'share_of_allocated_difference': _f(effects[effects.index.isin(none)].sum() / effects.sum()) if effects.sum() else None,
                                                             'names': {k: _f(v / init) for k, v in effects[effects.index.isin(none) & (effects.abs() > 0.001 * init)].sort_values().items()}},
                      'largest_single_name_effects': {k: _f(v / init) for k, v in effects.reindex(effects.abs().nlargest(5).index).items()},
                      'trading': {a: {k: _read(Path(b['output']) / 'trading.json').get(k) for k in ('turnover_two_sided_daily_mean', 'fill_rate', 'reject_reasons', 'top5_weight_mean', 'fee_ratio_to_initial')} for a, b in runs.items()}}
    return out


# 入口 ---------------------------------------------------------------------------------------------------
def diagnose(root, path, source_check = None, output = None, runs_root = None):
    s = load_source(path); doc = {'kind': 'diagnostic', 'source': _sha_inputs(s), 'environment': environment(), 'note': '只读：缺失模式、打分变化分解、集中持仓贡献'}
    if source_check: doc['source_check'] = {'file_sha256': file_sha(source_check), **_read(source_check)}
    out = create_run_dir(Path(runs_root) if runs_root else Path(root) / 'runs', output, f'{s["dir"].name[-4:]}-diagnose'); status = RunStatus(out, out.name, kind = 'diagnostic', evidence = 'exploratory'); write_json(out / 'config.json', doc)
    try:
        res = {'source': s['dir'].name, 'missingness': missingness(root, s)}; status.stage('missingness')
        res['score_decomposition'] = score_decomposition(s); status.stage('decomposition'); res['concentration'] = concentration(s, res['missingness']['no_minute_instruments']); status.stage('concentration')
        if source_check: res['source_check'] = doc['source_check']
    except Exception as exc:
        status.finish('failed', error = f'{type(exc).__name__}: {exc}'); raise
    write_json(out / 'diagnostics.json', canonical(res)); status.finish('success')
    write_json(out / 'manifest.json', {'run_id': out.name, 'kind': 'diagnostic', 'status': 'success', 'environment': doc['environment'], 'files': {p.name: file_sha(p) for p in sorted(out.iterdir()) if p.is_file() and p.name != 'manifest.json'}})
    return {'run_id': out.name, 'output': str(out), 'status': 'success'}
