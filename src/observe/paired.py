"""成对对照实验（阶段 4）：同一个分钟股票池、同一区间、同一切分，「日频基础因子」对「日频 + 分钟特征」。

两侧各是一次普通的研究实验（research.py，minute_pool = true），只有因子集不同：扩展因子集必须以基础因子集为前缀，其余全部相同。
运行后核对两侧的股票池、标签、切分逐值相同（否则不是成对），再比较样本外预测：逐日秩 IC、前 N 名平均标签、分窗口、分年份，
差值均值给块抽样 95% 区间；两侧的预测再各自走同一个回放入口做账本回测，比较成本后的组合结果。没有增量也照实报告。
证据级别是「探索」：分钟样本缺退市证券（决策 20），且区间只有 2022 年以后。"""
import hashlib, json, shutil
from pathlib import Path

import pandas as pd
from pydantic import Field, ValidationError, model_validator

from .data.store import Store
from .evaluation.paired import MODEL_IDS, curve_pairs, factor_pairs, model_pairs, pairing_gate
from .features import load_factor_set
from .replay import PortfolioConfig, ReproduceRefused, _read, run_offline
from .research import RESTRICTED, ResearchConfig, _research, dev_label_rule
from .runs import RunStatus, canonical, compare_frames, create_run_dir, drift, ensure_outside, environment, file_sha, write_json

EXTENDED = 'configs/factor_sets/daily_intraday_v1.yaml'
OK = ('success', 'success_limited')


class PairedConfig(ResearchConfig):
    extended_factor_set: str = EXTENDED
    minute_pool: bool = True
    initial_cash: float = Field(1_000_000.0, gt = 0)
    portfolio: PortfolioConfig = Field(default_factory = lambda: PortfolioConfig(n = 20, max_weight = 0.1, rebalance_every = 5, buffer = 10, max_sell = 5))
    backtest_models: list[str] = Field(default_factory = lambda: ['ridge', 'equal_blend'])
    n_boot: int = Field(1000, ge = 100, le = 20000)
    seed: int = 20260930

    @model_validator(mode = 'after')
    def _pool(self):
        if not self.minute_pool: raise ValueError('成对实验必须限制在分钟股票池内（minute_pool: true）')
        return self


def paired_params(file_cfg = None, **cli):
    raw = {**(file_cfg or {}), **{k: v for k, v in cli.items() if v is not None}}; output = raw.pop('output', None)
    try: return {'output': output, 'config': PairedConfig.model_validate(raw)}
    except ValidationError as exc: raise ValueError(f'paired 配置不合法：{exc}') from None


def run_paired(root, output = None, runs_root = None, **params):
    p = paired_params(params); return _paired(root, p['config'], output or p['output'], runs_root)


def arm_config(cfg, factor_set):
    """一侧的研究配置：与成对配置的研究字段完全相同，只换因子集"""
    return ResearchConfig(**{k: getattr(cfg, k) for k in ResearchConfig.model_fields if k != 'factor_set'}, factor_set = factor_set)


def check_extension(base, ext):
    """扩展因子集必须以基础因子集为前缀（名字、公式、方向逐个相同）且参数相同，否则两侧差的不只是分钟特征"""
    key = lambda f: (f['name'], f['expr'], f['direction'])
    if base['min_obs_ratio'] != ext['min_obs_ratio']: raise ValueError('两个因子集的 min_obs_ratio 不同')
    if [key(f) for f in ext['factors'][:len(base['factors'])]] != [key(f) for f in base['factors']]:
        raise ValueError('扩展因子集的前面部分必须与基础因子集逐个相同（名字、公式、方向、顺序）')
    if len(ext['factors']) == len(base['factors']): raise ValueError('扩展因子集没有新增因子')
    return [f['name'] for f in ext['factors'][len(base['factors']):]]


def _same_frames(a, b, keys): return compare_frames(a, b, keys, 0.0, 0.0)[1]['differences'] == 0


def _paired(root, cfg, output = None, runs_root = None, tag = 'paired'):
    store = Store(root); state = store.state(cfg.snapshot); base, ext = load_factor_set(cfg.factor_set), load_factor_set(cfg.extended_factor_set); new = check_extension(base, ext)
    doc = {'kind': 'paired', 'config': cfg.model_dump(mode = 'json'), 'snapshot_id': state.get('snapshot_id') or cfg.snapshot, 'batch_id': state.get('batch_id'),
           'factor_sets': {'base': {'source_path': cfg.factor_set, 'sha256': base['sha256']}, 'extended': {'source_path': cfg.extended_factor_set, 'sha256': ext['sha256']}},
           'new_factors': new, 'environment': environment()}
    h = hashlib.sha256(json.dumps(canonical({k: doc[k] for k in ('config', 'snapshot_id', 'factor_sets')}), sort_keys = True).encode()).hexdigest()
    runs_root = Path(runs_root) if runs_root else Path(root) / 'runs'
    out = create_run_dir(runs_root, output, '-'.join((h[:6], tag))); status = RunStatus(out, out.name, kind = 'paired', registry = Path(root) / 'runs', evidence = 'exploratory', config_hash = h)
    shutil.copyfile(cfg.factor_set, out / 'base_factor_set.yaml'); shutil.copyfile(cfg.extended_factor_set, out / 'extended_factor_set.yaml'); write_json(out / 'config.json', doc); status.stage('config')
    subruns = {'arms': {}, 'backtests': []}; limitations = []; blocked = None
    try:
        for key, path in (('base', cfg.factor_set), ('extended', cfg.extended_factor_set)):
            r = _research(root, arm_config(cfg, path), runs_root = runs_root, tag = f'pair-{key}'); subruns['arms'][key] = {'run_id': r['run_id'], 'output': r['output'], 'status': r['status']}
            status.stage(f'arm_{key}', run_id = r['run_id'], status = r['status'])
            if r['status'] not in OK: blocked = {'arm': key, 'run_id': r['run_id'], 'status': r['status'], 'blocked': r.get('blocked')}; break
        if blocked is None:
            a, b = (Path(subruns['arms'][k]['output']) for k in ('base', 'extended'))
            for n, keys in (('universe', ('decision_date', 'instrument')), ('labels', ('decision_date', 'instrument')), ('split_plan', ('split_id',))):
                if not _same_frames(pd.read_parquet(a / f'{n}.parquet'), pd.read_parquet(b / f'{n}.parquet'), keys): raise RuntimeError(f'两侧的 {n} 不相同，不是成对实验')
            status.stage('pairing_check')
            plan = pd.read_parquet(a / 'split_plan.parquet')
            for model in cfg.backtest_models:
                for key in ('base', 'extended'):
                    r = run_offline(root, runs_root = runs_root, snapshot = cfg.snapshot, initial_cash = cfg.initial_cash, portfolio = cfg.portfolio.model_dump(),
                                    scores = {'source': 'predictions', 'run': subruns['arms'][key]['output'], 'model': model})
                    subruns['backtests'].append({'model': model, 'arm': key, 'run_id': r['run_id'], 'output': r['output'], 'status': r['status']})
                    status.stage(f'backtest_{model}_{key}', run_id = r['run_id'], status = r['status'])
            write_json(out / 'subruns.json', subruns)
            report = evaluate(cfg, subruns, plan); write_json(out / 'paired_eval.json', canonical(report)); status.stage('evaluation')
            limitations = report['limitations']
    except Exception as exc:
        status.finish('failed', error = f'{type(exc).__name__}: {exc}'); raise
    if blocked is not None: final, info = 'blocked', {'blocked': [blocked]}
    else: final, info = ('success_limited' if limitations else 'success'), {'summary': report['headline']}
    write_json(out / 'subruns.json', subruns); write_json(out / 'limitations.json', canonical(limitations))
    status.finish(final, limitations = limitations, subruns = subruns, **info)
    files = {p.relative_to(out).as_posix(): file_sha(p) for p in sorted(out.rglob('*')) if p.is_file() and p.name != 'manifest.json'}
    write_json(out / 'manifest.json', {'run_id': out.name, 'kind': 'paired', 'status': final, 'environment': doc['environment'], 'files': files})
    return {'run_id': out.name, 'output': str(out), 'status': final, 'limitations': limitations, 'subruns': subruns, **info}


# 评价 -------------------------------------------------------------------------------------------------
def _json(path): return json.loads(Path(path).read_text(encoding = 'utf-8'))


def evaluate(cfg, subruns, plan):
    """只读两侧实验目录与回测目录里已保存的产物，不重新训练、不重新成交；同样输入得到同样输出（块抽样种子固定）"""
    a, b = (Path(subruns['arms'][k]['output']) for k in ('base', 'extended')); block = max(20, 4 * cfg.label_h); nb, seed = cfg.n_boot, cfg.seed
    pa, pb, lab = pd.read_parquet(a / 'predictions.parquet'), pd.read_parquet(b / 'predictions.parquet'), pd.read_parquet(a / 'labels.parquet')
    holdout = plan.holdout_start.iloc[0]; hold = None if pd.isna(holdout) else pd.Timestamp(holdout).date()
    lab_b = pd.read_parquet(b / 'labels.parquet'); gate = pairing_gate(pa, pb, lab, lab_b, MODEL_IDS)
    models = model_pairs(pa, pb, lab, block, cfg.models.min_names, cfg.models.top_n, nb, seed, hold, 2 * block, plan, 'exploratory')
    invalid = gate + [f'{m}:{r}' for m, v in models.items() for r in v['invalid_reasons']]; valid = not invalid      # 正式成对条件：不满足则主摘要数值不可用，受限诊断另列
    base_names = [f['name'] for f in load_factor_set(a / 'factor_set.yaml')['factors']]; ext = load_factor_set(b / 'factor_set.yaml')
    new = [f['name'] for f in ext['factors'] if f['name'] not in base_names]; directions = {f['name']: f['direction'] for f in ext['factors']}
    fac = pd.read_parquet(b / 'factors.parquet')
    dev = [d for d in sorted(set(fac.date)) if hold is None or d < hold]
    ridge = {}
    for sp in plan.itertuples():
        m = _json(b / 'models' / f'split{sp.split_id:02d}_ridge.json')
        mb_ = _json(a / 'models' / f'split{sp.split_id:02d}_ridge.json')
        ridge[int(sp.split_id)] = {'alpha': (m.get('penalty') or {}).get('alpha_used', m['params'].get('alpha')), 'base_alpha': (mb_.get('penalty') or {}).get('alpha_used', mb_['params'].get('alpha')), 'lambda': (m.get('penalty') or {}).get('lambda'), 'base_lambda': (mb_.get('penalty') or {}).get('lambda'), 'selected': m['info'].get('selected'), 'base_selected': mb_['info'].get('selected'), 'kept_new': [f for f in m['features'] if f in new], 'dropped': m['info'].get('dropped_features', []),
                                   'coef_new': {f: m['coef'][f] for f in m['features'] if f in new}}
    windows = [(sp.split_id, pd.Timestamp(sp.test_start).date(), pd.Timestamp(sp.test_end).date()) for sp in plan.itertuples()]
    portfolio, outs = {}, {(x['model'], x['arm']): Path(x['output']) for x in subruns['backtests']}
    for x in subruns['backtests']:
        d = outs[(x['model'], x['arm'])]; slot = portfolio.setdefault(x['model'], {'runs': {}}); st = _json(d / 'status.json'); entry = {'run_id': x['run_id'], 'status': x['status']}
        if x['status'] in OK:
            m, t = _json(d / 'metrics.json'), _json(d / 'trading.json')
            entry.update({k: m.get(k) for k in ('total_return', 'annual_return', 'annual_vol', 'sharpe', 'max_drawdown')}, turnover_two_sided_daily_mean = t['turnover_two_sided_daily_mean'],
                         fee_ratio_to_initial = t['fee_ratio_to_initial'], fill_rate = t['fill_rate'])
        else:       # 账本阻断（如持仓证券退市）：该运行的净值不可信，不列绩效指标，只记阻断原因与起始日
            issues = st.get('issues') or []
            entry.update(valid = False, blocked_from = min((i['date'] for i in issues), default = None), blocked_kinds = sorted({i['kind'] for i in issues}),
                         blocked_instruments = sorted({i['instrument'] for i in issues if i.get('instrument')}))
        slot['runs'][x['arm']] = entry
    for model, slot in portfolio.items():
        r = slot['runs']
        if set(r) == {'base', 'extended'} and all((outs[(model, k)] / 'equity.json').exists() for k in r):
            bad = [v['blocked_from'] for v in r.values() if v.get('blocked_from')]
            eq = {k: _json(outs[(model, k)] / 'equity.json') for k in r}
            pair = curve_pairs(eq['base'], eq['extended'], windows, block, nb, seed, until = min(bad) if bad else None)
            if not bad: slot['pair'] = pair
            else:       # 完整计划区间的组合指标不可用；阻断前的共同区间只作诊断，不能当作有效的组合比较
                slot['full_period_metrics'] = 'unavailable'
                slot['diagnostic_prefix'] = {**pair, 'valid_portfolio_comparison': False, 'truncation_reason': '账本阻断：' + '、'.join(sorted({f"{k}:{'/'.join(v['blocked_kinds'])}({','.join(v['blocked_instruments'])})@{v['blocked_from']}" for k, v in r.items() if v.get('blocked_from')}))}
    cover = _json(b / 'intraday_coverage.json'); sa, sb = _json(a / 'status.json'), _json(b / 'status.json')
    kinds = sorted({x['kind'] for r in (sa, sb) for x in r.get('limitations', [])})
    limitations = [{'kind': 'minute_sample_restricted', 'detail': RESTRICTED}]
    if len(plan) < 6: limitations.append({'kind': 'few_test_windows', 'detail': f'只有 {len(plan)} 个测试窗（约 {len(plan) * cfg.split.test} 个交易日），差值的区间很宽，结论只作探索'})
    if not valid: limitations.append({'kind': 'pairing_invalid', 'detail': '两侧预测不满足正式成对条件，主摘要不可用；受限诊断见 models.*.diagnostic', 'invalid_reasons': invalid})
    limitations.append({'kind': 'arm_limitations', 'detail': f'两侧实验自身的限制类型：{kinds}'})
    ridge_ic = (models.get('ridge') or {}).get('rank_ic') or {}
    headline = {'valid_primary_comparison': valid, 'invalid_reasons': invalid, 'train_start': str(plan.train_start.min()), 'test_start': str(plan.test_start.min()), 'test_end': str(plan.test_end.max()), 'windows': int(len(plan)),
                'ridge_rank_ic_base': ridge_ic.get('a_mean'), 'ridge_rank_ic_extended': ridge_ic.get('b_mean'), 'ridge_rank_ic_diff': ridge_ic.get('diff_mean'), 'ridge_rank_ic_diff_ci95': ridge_ic.get('diff_ci95')}
    return {'design': {'question': '在同一分钟股票池、同一区间、同一切分上，日频基础因子加入分钟聚合因子后，样本外预测是否有增量',
                       'primary': 'Ridge 在测试窗上的逐日秩 IC，扩展侧减基础侧的均值及块抽样 95% 区间', 'secondary': ['等权合成的秩 IC', '前 N 名平均标签', '分窗口与分年份的差值方向', '成本后账本回测的组合指标'],
                       'block_days': block, 'n_boot': nb, 'seed': seed, 'sides': {'base': subruns['arms']['base']['run_id'], 'extended': subruns['arms']['extended']['run_id']},
                       'new_factors': new, 'directions_prespecified': True, 'evidence': 'exploratory', 'alt_block_days': 2 * block,
                       'uncertainty_scope': '固定预测条件下按交易日位置的块抽样区间；预测、模型选择与拟合当作已知，区间不含重新选模、重新拟合带来的变化', 'label_rule': dev_label_rule(hold)},
            'valid_primary_comparison': valid, 'invalid_reasons': invalid, 'headline': headline, 'models': models, 'new_factors': factor_pairs(fac, lab, new, directions, dev, block, cfg.models.min_names, nb, seed, hold), 'ridge_new_factor_use': ridge,
            'coverage': {'total': cover['total'], 'by_year': cover['by_year'], 'instruments': cover['instruments']}, 'portfolio': portfolio, 'limitations': limitations}


# 复现 -------------------------------------------------------------------------------------------------
def _flat(x, path = ''):
    """嵌套的 JSON 对象 → {路径: 叶子值}，用于逐叶比较"""
    if isinstance(x, dict): return {q: v for k, val in x.items() for q, v in _flat(val, f'{path}/{k}').items()}
    if isinstance(x, list): return {q: v for k, val in enumerate(x) for q, v in _flat(val, f'{path}/{k}').items()}
    return {path: x}


def reproduce_paired(root, run, output = None, abs_tol = 1e-9, rel_tol = 0.0):
    """逐个复现两侧实验与全部回测（各自逐表比较），再用源目录里已保存的产物重算 paired_eval.json 与源文件比较；源目录只读"""
    from .replay import reproduce
    source = Path(run); ensure_outside(source, output)
    manifest, doc, st = _read(source / 'manifest.json'), _read(source / 'config.json'), _read(source / 'status.json')
    if st.get('status') not in ('success', 'success_limited', 'blocked', 'mismatch'): raise ReproduceRefused(f"源实验状态为 {st.get('status')}，不是已完成的实验")
    integrity = sorted(n for n, sha in manifest.get('files', {}).items() if not (source / n).exists() or file_sha(source / n) != sha)
    try: cfg = PairedConfig.model_validate(doc['config'])
    except (KeyError, ValidationError) as exc: raise ReproduceRefused(f'源实验配置不合法：{exc}') from None
    for name, key in (('base_factor_set.yaml', 'base'), ('extended_factor_set.yaml', 'extended')):
        if not (source / name).exists() or file_sha(source / name) != doc['factor_sets'][key]['sha256']: raise ReproduceRefused(f'冻结的 {name} 与记录的哈希不一致')
    subruns = _read(source / 'subruns.json'); results = {}
    for label, item in [(f'arm_{k}', v) for k, v in subruns['arms'].items()] + [(f"backtest_{x['model']}_{x['arm']}", x) for x in subruns['backtests']]:
        r = reproduce(root, item['output'], None, abs_tol, rel_tol); results[label] = {'source': item['run_id'], 'reproduced': r['run_id'], 'result': r['reproduction']['result'], 'differences': r['reproduction']['differences']}
    plan = pd.read_parquet(Path(subruns['arms']['base']['output']) / 'split_plan.parquet')
    now = canonical(evaluate(cfg, subruns, plan)); expected = _read(source / 'paired_eval.json')
    fa, fb = _flat(expected), _flat(now)
    bad = [k for k in sorted(set(fa) | set(fb)) if fa.get(k) != fb.get(k) and not (isinstance(fa.get(k), float) and isinstance(fb.get(k), float) and abs(fa[k] - fb[k]) <= abs_tol + rel_tol * abs(fb[k]))]
    result = 'match' if not bad and not integrity and all(v['result'] == 'match' for v in results.values()) else 'mismatch'
    out = create_run_dir(source.parent, output, f'{st.get("config_hash", "")[:6]}-paired-repro'); RunStatus(out, out.name, kind = 'paired', reproduce_of = str(source.resolve())).finish(
        'success' if result == 'match' else 'mismatch', reproduction = {'of': str(source), 'result': result})
    write_json(out / 'comparison.json', {'source': str(source), 'result': result, 'tolerance': {'abs': abs_tol, 'rel': rel_tol}, 'subruns': results, 'paired_eval_differences': bad[:200],
                                         'source_integrity': {'modified_files': integrity}, 'code_drift': drift(doc.get('environment') or {}, environment())})
    write_json(out / 'manifest.json', {'run_id': out.name, 'kind': 'paired', 'status': 'success' if result == 'match' else 'mismatch', 'files': {}})
    return {'run_id': out.name, 'output': str(out), 'status': 'success' if result == 'match' else 'mismatch',
            'reproduction': {'of': str(source), 'result': result, 'differences': len(bad) + sum(v['differences'] for v in results.values()), 'modified_source_files': integrity}}
