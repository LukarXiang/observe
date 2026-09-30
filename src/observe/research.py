"""研究流水线（模块 11–14、17）：固定快照 → 股票池 → 因子 → 标签 → 切分 → 基线与 Ridge 的样本外预测 → 因子与模型评价。

预测只覆盖开发区间的测试窗；最终留出区间（默认最近 252 个交易日）不生成预测、不参与任何选择。
产物写入新建实验目录（与 run 相同的独占创建、running 状态与 manifest），预测表交给 run 的唯一执行入口做组合与账本回测。"""
import hashlib, json, shutil
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
from pydantic import Field, ValidationError

from .data.prices import with_adjusted
from .data.store import Store
from .dataset import cross_sectional_preprocess, dev_labels, plan_splits, samples
from .execution import InputBlocked, sessions as open_sessions
from .factors.intraday import INTRADAY_FIELDS, daily_features, gap_reasons
from .features import factor_frame, load_factor_set, panel
from .labels import build_labels
from .evaluation.ranking import rank_ic, rank_ic_table, topn_summary, topn_table, undefined_reasons
from .models import EqualBlend, RidgeModel, SingleFactor, day_weights
from .replay import _Strict
from .runs import RunStatus, canonical, compare_frames, compare_tables, create_run_dir, drift, ensure_outside, environment, file_sha, write_json
from .universe import build_universe, version as universe_version

FACTOR_SET = 'configs/factor_sets/daily_basic_v1.yaml'
TABLES = ('calendar', 'bars_1d', 'instruments', 'adj_factors', 'adj_coverage')
CORE = {'universe': ('decision_date', 'instrument'), 'factors': ('date', 'instrument'), 'labels': ('decision_date', 'instrument'),
        'split_plan': ('split_id',), 'predictions': ('model_id', 'decision_date', 'instrument'), 'intraday': ('date', 'instrument')}
CORE_JSON = ('model_eval', 'factor_eval', 'limitations', 'intraday_coverage')
RESTRICTED = '受限样本：缺退市证券的分钟数据'


class UniverseConfig(_Strict):
    boards: list[str] = Field(default_factory = lambda: ['main'])
    min_listed_sessions: int = Field(120, ge = 1)
    exclude_st: bool = True
    suspend_window: int = Field(20, ge = 1)
    max_suspended: int = Field(10, ge = 0)
    liquidity_window: int = Field(20, ge = 1)
    min_avg_amount: float = Field(2e7, ge = 0)


class SplitConfig(_Strict):
    train: int = Field(504, ge = 1)
    valid: int = Field(126, ge = 1)
    test: int = Field(63, ge = 1)
    holdout: int = Field(252, ge = 0)


class ModelConfig(_Strict):
    baseline_factor: str = 'rev_5'
    ridge_alphas: list[float] = Field(default_factory = lambda: [1.0, 10.0, 100.0, 1000.0, 10000.0], min_length = 1, max_length = 12)
    correlation_threshold: float = Field(0.9, gt = 0, le = 1)
    min_names: int = Field(30, ge = 2)
    top_n: int = Field(20, ge = 1)


class ResearchConfig(_Strict):
    snapshot: str = Field(min_length = 1)
    start: date | None = None
    end: date | None = None
    factor_set: str = FACTOR_SET
    label_h: int = Field(5, ge = 1)
    universe: UniverseConfig = Field(default_factory = UniverseConfig)
    split: SplitConfig = Field(default_factory = SplitConfig)
    models: ModelConfig = Field(default_factory = ModelConfig)
    minute_pool: bool = False      # 研究候选再限制在分钟股票池内，区间限制在有名单的年份、且不晚于分钟线最后一天；分钟聚合特征因子必须开启


def research_params(file_cfg = None, **cli):
    raw = {**(file_cfg or {}), **{k: v for k, v in cli.items() if v is not None}}; output = raw.pop('output', None)
    try: return {'output': output, 'config': ResearchConfig.model_validate(raw)}
    except ValidationError as exc: raise ValueError(f'research 配置不合法：{exc}') from None


def run_research(root, output = None, runs_root = None, **params):
    p = research_params(params); return _research(root, p['config'], output or p['output'], runs_root)


def _pq(out, name, df): df.to_parquet(Path(out) / f'{name}.parquet', index = False)


def evidence_level(cfg): return 'exploratory' if cfg.minute_pool else 'development_oos'      # 分钟样本缺退市证券：数据受限，证据级别为探索


def _minute_inputs(store, state, cfg, need_bars):
    """分钟股票池研究的输入：名单表与分钟线分区必须在快照里；需要分钟特征时读全部分区（含预热月份），否则只读最后一个分区确定分钟线的最后一天"""
    parts, pool = state['tables'].get('bars_5m', {}), state['tables'].get('minute_universe', {})
    if not parts or not pool: raise InputBlocked([{'kind': 'minute_data_missing', 'detail': '快照没有 bars_5m 或 minute_universe，不能做分钟股票池研究'}])
    if cfg.end: parts = {k: v for k, v in parts.items() if k <= f'{cfg.end:%Y%m}'}
    if not parts: raise InputBlocked([{'kind': 'minute_data_missing', 'detail': f'{cfg.end} 之前没有分钟线'}])
    last = max(parts); day = pd.read_parquet(store.root / parts[last]['file'], columns = ['bar_end']).bar_end.max().date()
    return {'parts': parts if need_bars else {last: parts[last]}, 'last_day': day}


def _research(root, cfg, output = None, runs_root = None, factor_file = None, tag = 'research', reproduce_of = None, compare = None):
    store = Store(root); state = store.state(cfg.snapshot)
    factor_file = Path(factor_file or cfg.factor_set); fset = load_factor_set(factor_file)
    names = [f['name'] for f in fset['factors']]
    if cfg.models.baseline_factor not in names: raise ValueError(f'单因子基线 {cfg.models.baseline_factor} 不在因子集里')
    intraday = sorted({x for f in fset['factors'] for x in f['fields']} & set(INTRADAY_FIELDS))
    if intraday and not cfg.minute_pool: raise ValueError(f'因子集用了分钟聚合特征 {intraday}，必须同时设置 minute_pool: true（只有分钟股票池内的证券有值）')
    doc = {'kind': 'research', 'config': cfg.model_dump(mode = 'json'), 'snapshot_id': state.get('snapshot_id') or cfg.snapshot, 'batch_id': state.get('batch_id'),
           'factor_set': {'source_path': str(cfg.factor_set), 'sha256': fset['sha256'], 'version': fset['version']}, 'label': {'name': 'adj_open_to_open_h', 'h': cfg.label_h},
           'environment': environment(), 'reproduce_of': reproduce_of}
    h = hashlib.sha256(json.dumps(canonical({k: doc[k] for k in ('config', 'snapshot_id', 'factor_set', 'label')}), sort_keys = True).encode()).hexdigest()
    out = create_run_dir(Path(runs_root) if runs_root else Path(root) / 'runs', output, '-'.join(x for x in (h[:6], tag) if x))
    status = RunStatus(out, out.name, kind = 'research', evidence = evidence_level(cfg), config_hash = h, reproduce_of = reproduce_of)
    shutil.copyfile(factor_file, out / 'factor_set.yaml'); write_json(out / 'config.json', doc); status.stage('config')
    limitations = []
    try:
        used = {n: state['tables'].get(n, {}) for n in TABLES + (('minute_universe',) if cfg.minute_pool else ())}
        minute = _minute_inputs(store, state, cfg, bool(intraday)) if cfg.minute_pool else None
        if minute: used['bars_5m'] = minute['parts']
        t = {name: store.load_state(state, name) for name in TABLES}
        write_json(out / 'data_manifest.json', {'snapshot_id': doc['snapshot_id'], 'batch_id': doc['batch_id'], 'offline': True, 'tables': state.get('tables', {}),
                                                'used': {n: {p: {**v, 'file_sha256': file_sha(store.root / v['file'])} for p, v in parts.items()} for n, parts in used.items()}})
        if minute:
            minute['pool'] = store.load_state(state, 'minute_universe')
            minute['daily'] = pd.concat([daily_features(pd.read_parquet(store.root / v['file'])) for v in minute['parts'].values()], ignore_index = True) if intraday else None
        from .data.audit import audit_status
        from .data.update import default_rules
        try: rules = default_rules()
        except Exception: rules = None
        audit = {k: v for k, v in audit_status(root, doc['batch_id'], rules).items() if k != 'rows'}   # 运行前检查：与回放、数据中心同一判定
        if audit['status'] != 'passed':
            limitations.append({'kind': 'data_audit', 'detail': f"快照批次审计状态为 {audit['status']}（范围 {audit['scope']}），不是全快照审计通过", 'audit': audit})
        status.stage('load', audit = audit['status'])
        if cfg.minute_pool: limitations.append({'kind': 'minute_sample_restricted', 'detail': f'{RESTRICTED}（外部分钟线不含退市证券；这些证券的分钟特征缺失，按 0 处理）'})
        result = _pipeline(cfg, fset, t, out, status, limitations, minute)
        final = 'success_limited' if limitations else 'success'; info = {'summary': result}
    except InputBlocked as exc:
        final, info = 'blocked', {'blocked': exc.issues}
    except Exception as exc:
        status.finish('failed', error = f'{type(exc).__name__}: {exc}'); raise
    write_json(out / 'limitations.json', canonical(limitations))
    extra = {}
    if compare is not None:
        extra = compare(out, {'status': final, **info})
        if extra['reproduction']['result'] == 'mismatch': extra['execution_status'], final = final, 'mismatch'
    status.finish(final, limitations = limitations, **info, **extra)
    files = {p.relative_to(out).as_posix(): file_sha(p) for p in sorted(out.rglob('*')) if p.is_file() and p.name != 'manifest.json'}
    write_json(out / 'manifest.json', {'run_id': out.name, 'kind': 'research', 'status': final, 'environment': doc['environment'], 'files': files})
    return {'run_id': out.name, 'output': str(out), 'status': final, 'limitations': limitations, **info, **extra}


def _pipeline(cfg, fset, t, out, status, limitations, minute = None):
    names = [f['name'] for f in fset['factors']]; directions = {f['name']: f['direction'] for f in fset['factors']}
    bars = t['bars_1d'].copy(); bars['date'] = pd.to_datetime(bars.date).dt.date
    if not len(bars): raise InputBlocked([{'kind': 'bars_missing', 'detail': '快照没有日线'}])
    have = set(bars.date); first, last = min(have), max(have)
    cal = [d for d in open_sessions(t['calendar']) if first <= d <= last]
    missing = [d for d in cal if d not in have]
    if missing: raise InputBlocked([{'kind': 'missing_session', 'detail': f'{len(missing)} 个交易日整日没有日线', 'dates': [str(d) for d in missing[:50]]}])
    warmup = max(cfg.universe.min_listed_sessions, max(f['lookback'] for f in fset['factors']) + 1, cfg.universe.liquidity_window, cfg.universe.suspend_window)
    days = [d for d in cal[warmup:] if (cfg.start is None or d >= cfg.start) and (cfg.end is None or d <= cfg.end)]
    if minute:
        years = sorted({int(y) for y in minute['pool'].year})
        if not years or years != list(range(years[0], years[-1] + 1)): raise InputBlocked([{'kind': 'minute_pool_years', 'detail': f'分钟股票池的年份 {years} 不连续或为空'}])
        days = [d for d in days if years[0] <= d.year <= years[-1] and d <= minute['last_day']]
    if not days: raise InputBlocked([{'kind': 'insufficient_history', 'detail': f'交易日历 {len(cal)} 天，不足预热 {warmup} 天，或区间内没有分钟股票池'}])

    uni = build_universe(bars, t['instruments'], cal, cfg.universe.model_dump(), days[0], days[-1])
    if minute: uni = _restrict_to_pool(uni, minute['pool'], cfg.universe.model_dump())
    _pq(out, 'universe', uni)
    elig = uni[uni.eligible]; status.stage('universe', rows = len(uni), eligible = len(elig), days = len(days), warmup = warmup)
    ever = sorted(set(elig.instrument))
    view = with_adjusted(bars[bars.instrument.isin(set(ever))], t['adj_factors'], t['adj_coverage'])
    ok = view.merge(elig[['decision_date', 'instrument']], left_on = ['date', 'instrument'], right_on = ['decision_date', 'instrument'])
    unusable = float((ok.adjustment_status != 'usable').mean()) if len(ok) else 1.0
    if unusable > 0: limitations.append({'kind': 'adjusted_price_unavailable', 'detail': f'{unusable:.2%} 的研究候选行没有可用复权价（因子、标签缺失）', 'share': unusable})
    if minute and minute['daily'] is not None:
        view['date'] = pd.to_datetime(view.date).dt.date; view = view.merge(minute['daily'].drop(columns = 'n_bars'), on = ['date', 'instrument'], how = 'left')
    wide = panel(view, cal, ever)
    mask = elig.assign(v = True).pivot(index = 'decision_date', columns = 'instrument', values = 'v').reindex(index = cal, columns = ever).fillna(False).astype(bool)
    fac = factor_frame(fset, wide, mask); _pq(out, 'factors', fac); status.stage('factors', rows = len(fac), factors = names)
    if minute and minute['daily'] is not None:
        table, coverage = _intraday_report(elig, minute['daily']); _pq(out, 'intraday', table); write_json(out / 'intraday_coverage.json', coverage)
        status.stage('intraday', rows = len(table), with_bars = coverage['total']['with_minute_bars'], candidate_rows = coverage['total']['candidate_rows'])
    lab = build_labels(view.rename(columns = {'open_adj': 'adj_open'})[['date', 'instrument', 'adj_open', 'is_trading']], cal, h = cfg.label_h)
    lab = lab.merge(elig[['decision_date', 'instrument']], on = ['decision_date', 'instrument']); _pq(out, 'labels', lab)
    status.stage('labels', rows = len(lab), valid = int(lab.valid.sum()))

    s = cfg.split; holdout = days[-s.holdout] if s.holdout and len(days) > s.holdout else None
    plan = plan_splits(days, s.train, s.valid, s.test, holdout)
    if not len(plan): raise InputBlocked([{'kind': 'insufficient_history_for_split', 'detail': f'开发区间 {len(days) - (s.holdout if holdout else 0)} 个决策日，不足一个训练 + 验证 + 测试窗（{s.train + s.valid + s.test}）'}])
    plan['holdout_start'] = holdout; _pq(out, 'split_plan', plan); status.stage('splits', windows = len(plan), holdout_start = str(holdout) if holdout else None)

    X = cross_sectional_preprocess(fac.assign(eligible = True), names).drop(columns = 'eligible')
    preds, evals, models_dir = [], [], out / 'models'; models_dir.mkdir()
    for sp in plan.itertuples(index = False):
        kept, dropped = _prune(X, sp, names, cfg.models.correlation_threshold)
        def rows(stage):
            y = samples(lab, sp, stage)[['decision_date', 'instrument', 'value']]
            return X.merge(y, left_on = ['date', 'instrument'], right_on = ['decision_date', 'instrument']).sort_values(['date', 'instrument'])
        sel, val, fit = rows('select'), rows('valid'), rows('fit')
        test = X[(X.date >= sp.test_start) & (X.date <= sp.test_end)].sort_values(['date', 'instrument'])
        candidates = [('single_factor', SingleFactor([cfg.models.baseline_factor], directions, factor = cfg.models.baseline_factor)),
                      ('equal_blend', EqualBlend(kept, directions))] + [(f'ridge@{a:g}', RidgeModel(kept, directions, alpha = a)) for a in cfg.models.ridge_alphas]
        tried = []
        for key, m in candidates:
            m.fit(sel[m.features], sel.value, day_weights(sel.date))
            ic = rank_ic(val.assign(score = m.predict(val[m.features])), min_n = cfg.models.min_names).mean()
            tried.append({'split_id': sp.split_id, 'candidate': key, 'valid_rank_ic': None if np.isnan(ic) else float(ic)})
        ridge = [x for x in tried if x['candidate'].startswith('ridge@')]
        best = max(ridge, key = lambda x: -np.inf if x['valid_rank_ic'] is None else x['valid_rank_ic'])['candidate']   # 并列取先出现（更小的 alpha）
        for model_id, key in (('single_factor', 'single_factor'), ('equal_blend', 'equal_blend'), ('ridge', best)):
            m = dict(candidates)[key]
            m.fit(fit[m.features], fit.value, day_weights(fit.date), split_id = sp.split_id, train_start = sp.train_start, fit_end = sp.valid_end, fit_asof = sp.fit_asof,
                  refit = 'train+valid', label = f'adj_open_to_open_{cfg.label_h}', dropped_features = dropped, selected = key)
            m.save(models_dir / f'split{sp.split_id:02d}_{model_id}.json')
            preds.append(pd.DataFrame({'model_id': model_id, 'decision_date': test.date.to_numpy(), 'instrument': test.instrument.to_numpy(), 'score': m.predict(test[m.features]),
                                       'split_id': sp.split_id, 'fit_asof': sp.fit_asof, 'evidence_level': evidence_level(cfg)}))
        evals += tried
    pred = pd.concat(preds, ignore_index = True); _pq(out, 'predictions', pred); status.stage('models', predictions = len(pred))

    model_eval = _model_eval(pred, lab, evals, cfg.models, holdout); write_json(out / 'model_eval.json', canonical(model_eval))
    dev = [d for d in days if holdout is None or d < holdout]
    factor_eval = _factor_eval(fac, lab, names, directions, dev, cfg.models.min_names, holdout); write_json(out / 'factor_eval.json', canonical(factor_eval))
    status.stage('evaluation')
    return {'days': len(days), 'first_day': str(days[0]), 'last_day': str(days[-1]), 'windows': len(plan), 'holdout_start': str(holdout) if holdout else None,
            'test_start': str(plan.test_start.min()), 'test_end': str(plan.test_end.max()), 'models': {k: v['test_rank_ic_mean'] for k, v in model_eval['summary'].items()}}


def _restrict_to_pool(uni, pool, universe_cfg):
    """研究候选再限制在当年的分钟股票池内（名单按上一自然年数据生成，当年的决策日都可用）"""
    members = pd.MultiIndex.from_arrays([pool.year.astype(int).to_numpy(), pool.instrument.to_numpy()])
    key = pd.MultiIndex.from_arrays([pd.to_datetime(uni.decision_date).dt.year.to_numpy(), uni.instrument.to_numpy()])
    drop = uni.eligible.to_numpy() & ~key.isin(members)
    uni = uni.copy(); uni.loc[drop, 'eligible'] = False; uni.loc[drop, 'reason'] = 'not_in_minute_pool'
    uni['universe_version'] = universe_version({**universe_cfg, 'minute_pool': True}); return uni


def _intraday_report(elig, daily):
    """研究候选行 × 分钟特征（无分钟线的行各特征为缺失）与覆盖报告：按年、按证券；缺失原因见 factors.intraday.gap_reasons"""
    e = elig[['decision_date', 'instrument']].rename(columns = {'decision_date': 'date'}); e['date'] = pd.to_datetime(e.date).dt.date
    j = e.merge(daily, on = ['date', 'instrument'], how = 'left'); year = pd.to_datetime(j.date).dt.year
    def block(g): return {'candidate_rows': int(len(g)), 'with_minute_bars': int(g.n_bars.notna().sum()), 'bar_coverage': _f(g.n_bars.notna().mean()),
                          'feature_coverage': {f: _f(g[f].notna().mean()) for f in INTRADAY_FIELDS}, 'gaps': gap_reasons(g[g.n_bars.notna()])}
    per = j.groupby('instrument').n_bars.apply(lambda x: x.notna().mean()).sort_values(kind = 'stable')
    coverage = {'total': block(j), 'by_year': {int(y): block(g) for y, g in j.groupby(year)},
                'instruments': {'candidates': int(len(per)), 'no_minute_data': sorted(per.index[per == 0]), 'below_90pct': int((per < 0.9).sum()),
                                'lowest': {i: _f(v) for i, v in per.head(20).items()}},
                'note': f'{RESTRICTED}；无分钟线的研究候选行特征缺失，横截面预处理后按 0 处理'}
    return j, canonical(coverage)


def _prune(X, sp, names, threshold):
    """训练窗内每日秩相关的均值；绝对值大于阈值的一对只保留因子集里靠前者"""
    w = X[(X.date >= sp.train_start) & (X.date <= sp.train_end)]
    corr = w.groupby('date')[names].apply(lambda g: g.rank().corr()).groupby(level = 1).mean().reindex(index = names, columns = names)
    kept, dropped = [], []
    for n in names:
        hit = [k for k in kept if abs(corr.loc[n, k]) > threshold]
        if hit: dropped.append({'factor': n, 'correlated_with': hit[0], 'mean_rank_corr': float(corr.loc[n, hit[0]])})
        else: kept.append(n)
    return kept, dropped


def _model_eval(pred, lab, tried, mc, holdout = None):
    """测试窗上的模型评价。标签只用开发区间可用的（dataset.dev_labels）；秩 IC 与前 N 名都走 evaluation.ranking 的唯一实现，
    前 N 名名单由预测先固定，标签无效的证券留在名单里、不替补"""
    labd = dev_labels(lab, holdout); j = pred.merge(labd[['decision_date', 'instrument', 'value']], on = ['decision_date', 'instrument'], how = 'left')
    per, summary, key = [], {}, f'top{mc.top_n}_mean_label'
    for (model, split), g in j.groupby(['model_id', 'split_id']):
        ic = rank_ic(g, min_n = mc.min_names); top = topn_table(g, labd, mc.top_n)
        per.append({'model_id': model, 'split_id': int(split), 'test_rank_ic_mean': _f(ic.mean()), 'test_days': int(ic.notna().sum()), key: _f(top.mean_label.mean()),
                    f'top{mc.top_n}_valid_label_share': _f(top.valid_label_count.sum() / top.selected_count.sum()) if len(top) else None, f'top{mc.top_n}_undefined_days': int((~top.defined).sum())})
    by = pd.DataFrame(per)
    for model, g in j.groupby('model_id'):
        t = rank_ic_table(g, min_n = mc.min_names); ic = t.ic
        summary[model] = {'test_rank_ic_mean': _f(ic.mean()), 'test_rank_ic_std': _f(ic.std()), 'positive_share': _f((ic > 0).mean()), 'test_days': int(ic.notna().sum()),
                          'rank_ic_undefined': undefined_reasons(t), f'top{mc.top_n}': topn_summary(topn_table(g, labd, mc.top_n))}
    wins = {}
    if len(by):
        wide = by.pivot(index = 'split_id', columns = 'model_id', values = 'test_rank_ic_mean')
        for base in ('single_factor', 'equal_blend'):
            if 'ridge' in wide and base in wide: wins[f'ridge_vs_{base}'] = {'wins': int((wide.ridge > wide[base]).sum()), 'windows': int(wide[['ridge', base]].notna().all(axis = 1).sum())}
    return {'eval_version': 2, 'per_window': per, 'summary': summary, 'window_wins': wins, 'selection': tried,
            'label_rule': dev_label_rule(holdout),
            'note': '标签是复权 open-to-open 收益，只作排序目标，不是可实现收益；组合收益只来自账本回测'}


def dev_label_rule(holdout):
    return {'holdout_start': None if holdout is None else str(holdout), 'rule': '标签有效，且决策日与成熟时点（退出日）都严格早于最终留出起点' if holdout is not None else '没有最终留出，标签有效即可'}


def _factor_eval(fac, lab, names, directions, dev, min_names, holdout = None):
    """开发区间因子评价：因子行取开发区间决策日，标签只用开发区间可用的（dataset.dev_labels，成熟时点也在留出起点之前）"""
    labd = dev_labels(lab, holdout); j = fac[fac.date.isin(set(dev))].merge(labd[['decision_date', 'instrument', 'value']], left_on = ['date', 'instrument'], right_on = ['decision_date', 'instrument'], how = 'left')
    out = {}
    for n in names:
        t = rank_ic_table(j.assign(score = j[n]), min_n = min_names); ic = t.ic
        cover = fac[fac.date.isin(set(dev))][n].notna().mean()
        years = ic.groupby(pd.to_datetime(pd.Series(ic.index, index = ic.index)).dt.year).mean()
        out[n] = {'direction': directions[n], 'rank_ic_mean': _f(ic.mean()), 'rank_ic_std': _f(ic.std()), 'positive_share': _f((ic > 0).mean()),
                  'direction_adjusted_ic': _f(ic.mean() * directions[n]), 'coverage': _f(cover), 'days': int(ic.notna().sum()), 'by_year': {int(k): _f(v) for k, v in years.items()},
                  'undefined': undefined_reasons(t)}
    return {'eval_version': 2, 'period': {'start': str(min(dev)) if dev else None, 'end': str(max(dev)) if dev else None, 'note': '开发区间，不含最终留出'}, 'label_rule': dev_label_rule(holdout), 'factors': out}


def _f(x): return None if x is None or not np.isfinite(x) else float(x)


def reproduce_research(root, run, output = None, abs_tol = 1e-9, rel_tol = 0.0):
    """用冻结的配置、因子集与快照在新目录重跑研究流水线，逐表比较股票池、因子、标签、切分、预测与评价"""
    from .replay import ReproduceRefused, _read, check_snapshot
    source = Path(run); ensure_outside(source, output)
    manifest, doc, st = _read(source / 'manifest.json'), _read(source / 'config.json'), _read(source / 'status.json')
    if st.get('status') not in ('success', 'success_limited', 'blocked', 'mismatch'): raise ReproduceRefused(f"源实验状态为 {st.get('status')}，不是已完成的实验")
    integrity = sorted(n for n, sha in manifest.get('files', {}).items() if not (source / n).exists() or file_sha(source / n) != sha)
    try: cfg = ResearchConfig.model_validate(doc['config'])
    except (KeyError, ValidationError) as exc: raise ReproduceRefused(f'源实验配置不合法：{exc}') from None
    frozen = source / 'factor_set.yaml'
    if not frozen.exists() or file_sha(frozen) != doc['factor_set']['sha256']: raise ReproduceRefused('冻结的 factor_set.yaml 与记录的哈希不一致')
    check_snapshot(Store(root), cfg.snapshot, _read(source / 'data_manifest.json'))
    expected = {n: pd.read_parquet(source / f'{n}.parquet') if (source / f'{n}.parquet').exists() else None for n in CORE}
    expected_json = {n: _read(source / f'{n}.json') if (source / f'{n}.json').exists() else None for n in CORE_JSON}
    code = drift(doc.get('environment') or {}, environment())

    def compare(out, now):
        out, diffs, tables = Path(out), [], {}
        for n, keys in CORE.items():
            e = expected[n]; a = pd.read_parquet(out / f'{n}.parquet') if (out / f'{n}.parquet').exists() else None
            if e is None or a is None: tables[n] = {'differences': int((e is None) != (a is None))}; continue
            d, tables[n] = compare_frames(e, a, keys, abs_tol, rel_tol); diffs += [{'table': n, **x} for x in d]
        d, t = compare_tables(expected_json, {n: _read(out / f'{n}.json') if (out / f'{n}.json').exists() else None for n in CORE_JSON}, abs_tol, rel_tol)
        diffs += d; tables.update(t)
        result = 'match' if not any(v.get('differences') for v in tables.values()) and not integrity else 'mismatch'
        write_json(out / 'comparison.json', {'source': str(source), 'reproduced': str(out), 'result': result, 'tolerance': {'abs': abs_tol, 'rel': rel_tol},
                                            'tables': tables, 'differences': diffs[:200], 'source_integrity': {'modified_files': integrity}, 'code_drift': code})
        return {'reproduction': {'of': str(source), 'result': result, 'differences': sum(v.get('differences', 0) for v in tables.values()),
                                 'modified_source_files': integrity, 'code_drift': sorted(code)}}

    return _research(root, cfg, output, source.parent, factor_file = frozen, tag = 'repro', reproduce_of = str(source.resolve()), compare = compare)
