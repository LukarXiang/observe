"""成对对照实验（阶段 4）：两侧只差因子集、样本逐值相同；分钟股票池与分钟线区间的限制；覆盖报告；受限样本与证据级别；复现。离线，合成数据。"""
import json, shutil
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from observe.cli import main
from observe.jobs import Jobs
from observe.paired import PairedConfig, check_extension, evaluate, run_paired
from observe.replay import reproduce
from observe.research import run_research
from observe.features import load_factor_set
from tests.integration.helpers import instruments, snapshot
from tests.integration.test_research import FACTORS, market

EXTENDED = FACTORS + """  - name: rv_5
    expr: ts_mean(rv_5m, 5)
    direction: -1
    desc: 5 日日内已实现波动均值
  - name: vwap_5
    expr: ts_mean(vwap_dev, 5)
    direction: -1
    desc: 5 日收盘价相对成交均价偏离均值
"""
POOL, LAST = 30, 80             # 分钟股票池只有前 30 只；分钟线只到第 80 个交易日（共 90 天）


def minute_bars(rows, days, seed = 1, drop = ()):
    """每只证券每天 48 根 5 分钟线：路径从开盘价走到收盘价，波动大小因证券而异"""
    rng = np.random.default_rng(seed); ends = [pd.Timedelta(minutes = 575 + 5 * k) for k in range(24)] + [pd.Timedelta(minutes = 785 + 5 * k) for k in range(24)]; out = []
    for r in rows:
        if r['date'] > days[LAST] or not r['is_trading'] or int(r['instrument'][1:6]) > POOL or r['instrument'] in drop: continue
        sigma = 0.0005 * (1 + int(r['instrument'][1:6]) % 5); steps = rng.normal(0, sigma, 48)
        path = r['open'] * np.exp(np.cumsum(steps) - (np.arange(1, 49) / 48) * (steps.sum() - np.log(r['close'] / r['open'])))
        close = np.round(path, 2); open_ = np.r_[r['open'], close[:-1]]; vol = 100 * rng.integers(1, 50, 48)
        out.append(pd.DataFrame({'bar_end': [pd.Timestamp(r['date']) + e for e in ends], 'instrument': r['instrument'], 'open': open_, 'high': np.maximum(open_, close) + 0.01,
                                 'low': np.minimum(open_, close) - 0.01, 'close': close, 'volume': vol, 'amount': np.round(vol * close, 2), 'source': 'external_1m'}))
    return pd.concat(out, ignore_index = True)


def make(root, n_days = 90, shock = None, drop = ()):
    days, insts, rows = market(n_days = n_days); bars = minute_bars(rows, days, drop = drop)
    if shock is not None:                                                    # 只改 shock 日及以后的分钟线：尾盘 30 分钟收盘价与最高价抬高 2%
        late = (bars.bar_end.dt.date >= shock) & (bars.bar_end.dt.hour * 60 + bars.bar_end.dt.minute >= 875); bars.loc[late, ['close', 'high']] *= 1.02
    cov = pd.DataFrame([{'instrument': i, 'status': 'no_events', 'verified_from': days[0].replace(year = 1990), 'verified_through': days[-1], 'has_start_basis': True,
                         'has_gap': False, 'confirmed_no_events': True} for i in insts])
    pool = pd.DataFrame({'year': 2023, 'instrument': insts[:POOL], 'rank': range(1, POOL + 1), 'avg_amount': 1e8, 'traded_days': 240, 'generated_asof': days[0].replace(year = 2022, month = 12, day = 30)})
    extra = {'bars_5m': {f'{y}{m:02d}': g for (y, m), g in bars.groupby([bars.bar_end.dt.year, bars.bar_end.dt.month])}, 'minute_universe': {'all': pool}}
    sid = snapshot(root, rows, inst = instruments(*insts), sessions = days, coverage = cov, extra = extra)
    (Path(root) / 'base.yaml').write_text(FACTORS, encoding = 'utf-8'); (Path(root) / 'ext.yaml').write_text(EXTENDED, encoding = 'utf-8')
    return sid, days, str(Path(root) / 'base.yaml'), str(Path(root) / 'ext.yaml')


def cfg(sid, base, ext, **kw):
    return {'snapshot': sid, 'factor_set': base, 'extended_factor_set': ext, 'label_h': 2, 'n_boot': 200,
            'universe': {'min_listed_sessions': 1, 'suspend_window': 5, 'max_suspended': 3, 'liquidity_window': 5, 'min_avg_amount': 1e6},
            'split': {'train': 20, 'valid': 10, 'test': 5, 'holdout': 5}, 'models': {'baseline_factor': 'rev_3', 'ridge_alphas': [1.0, 100.0], 'min_names': 10, 'top_n': 5},
            'initial_cash': 1_000_000, 'portfolio': {'n': 5, 'max_weight': 0.2, 'rebalance_every': 2, 'buffer': 2, 'max_sell': 5}, **kw}


def read(path, name): return json.loads((Path(path) / name).read_text(encoding = 'utf-8'))


@pytest.fixture(scope = 'module')
def paired(tmp_path_factory):
    root = tmp_path_factory.mktemp('paired'); sid, days, base, ext = make(root)
    r = run_paired(root, **cfg(sid, base, ext)); return root, sid, days, base, ext, r


def test_both_sides_share_the_same_restricted_sample(paired):
    root, sid, days, base, ext, r = paired; assert r['status'] == 'success_limited'
    a, b = Path(r['subruns']['arms']['base']['output']), Path(r['subruns']['arms']['extended']['output'])
    for n in ('universe', 'labels', 'split_plan'): pd.testing.assert_frame_equal(pd.read_parquet(a / f'{n}.parquet'), pd.read_parquet(b / f'{n}.parquet'))
    uni = pd.read_parquet(a / 'universe.parquet'); u = uni[uni.eligible]
    assert set(u.instrument) <= {f'6{k:05d}.SH' for k in range(1, POOL + 1)} and (uni.reason == 'not_in_minute_pool').any()      # 池外证券被排除并注明原因
    assert pd.Timestamp(uni.decision_date.max()).date() <= days[LAST]                                                   # 区间不晚于分钟线最后一天
    fa, fb = pd.read_parquet(a / 'factors.parquet'), pd.read_parquet(b / 'factors.parquet')
    assert list(fb.columns[:len(fa.columns)]) == list(fa.columns) and {'rv_5', 'vwap_5'} <= set(fb.columns) and fb.rv_5.notna().mean() > 0.5
    pa, pb = pd.read_parquet(a / 'predictions.parquet'), pd.read_parquet(b / 'predictions.parquet')
    assert (pa.evidence_level == 'exploratory').all() and read(a, 'status.json')['evidence'] == 'exploratory'
    key = ['model_id', 'decision_date', 'instrument']; assert len(pa) == len(pb) and pa[key].equals(pb[key])
    assert np.array_equal(pa[pa.model_id == 'single_factor'].score, pb[pb.model_id == 'single_factor'].score)              # 基线用同一个因子：两侧分数相同
    assert not np.allclose(pa[pa.model_id == 'ridge'].score, pb[pb.model_id == 'ridge'].score)


def test_restricted_sample_and_coverage_are_disclosed(paired):
    root, sid, days, base, ext, r = paired; b = Path(r['subruns']['arms']['extended']['output'])
    lim = read(b, 'limitations.json'); assert any(x['kind'] == 'minute_sample_restricted' and '缺退市证券的分钟数据' in x['detail'] for x in lim)
    cov = read(b, 'intraday_coverage.json'); total = cov['total']
    assert total['candidate_rows'] == total['with_minute_bars'] > 0 and total['bar_coverage'] == 1.0 and set(cov['by_year']) == {'2023'}
    assert total['feature_coverage']['rv_5m'] == 1.0 and total['gaps']['too_few_bars'] == 0
    intraday = pd.read_parquet(b / 'intraday.parquet'); assert {'n_bars', 'rv_5m', 'vwap_dev'} <= set(intraday.columns) and (intraday.n_bars.dropna() == 48).all()
    data = read(b, 'data_manifest.json')['used']; assert 'bars_5m' in data and 'minute_universe' in data                       # 用到的分钟线分区按哈希冻结
    only_base = read(r['subruns']['arms']['base']['output'], 'data_manifest.json')['used']; assert len(only_base['bars_5m']) == 1   # 基础侧只读最后一个分区确定分钟线的最后一天
    assert not (Path(r['subruns']['arms']['base']['output']) / 'intraday.parquet').exists()


def test_paired_report_compares_prediction_and_portfolio_layers(paired):
    root, sid, days, base, ext, r = paired; out = Path(r['output']); ev = read(out, 'paired_eval.json'); st = read(out, 'status.json')
    assert st['kind'] == 'paired' and st['evidence'] == 'exploratory' and ev['design']['evidence'] == 'exploratory' and ev['design']['directions_prespecified']
    assert set(ev['models']) == {'single_factor', 'equal_blend', 'ridge'} and ev['models']['single_factor']['identical_scores']
    ridge = ev['models']['ridge']; ic = ridge['rank_ic']
    assert ic['days'] > 10 and ic['diff_mean'] == pytest.approx(ic['b_mean'] - ic['a_mean']) and ic['diff_ci95'][0] <= ic['diff_ci95'][1]
    assert ridge['windows'] == len(ridge['by_window']) >= 3 and set(ridge['by_year']) == {'2023'} and 'top5_mean_label' in ridge
    assert ev['models']['single_factor']['rank_ic']['diff_mean'] == pytest.approx(0)
    assert set(ev['new_factors']) == {'rv_5', 'vwap_5'} and ev['new_factors']['rv_5']['coverage'] > 0.5 and ev['new_factors']['rv_5']['rank_ic_ci95'] is not None
    assert set(ev['ridge_new_factor_use']) == set(map(str, range(ridge['windows']))) and ev['headline']['ridge_rank_ic_diff'] == ic['diff_mean']
    pf = ev['portfolio']; assert set(pf) == {'ridge', 'equal_blend'}
    for slot in pf.values():
        assert set(slot['runs']) == {'base', 'extended'} and all(v['status'] in ('success', 'success_limited') for v in slot['runs'].values())
        assert slot['pair']['daily_return']['days'] > 10 and len(slot['pair']['by_window']) == ridge['windows']
    assert any(x['kind'] == 'few_test_windows' for x in ev['limitations']) or ridge['windows'] >= 6
    assert (out / 'base_factor_set.yaml').read_text(encoding = 'utf-8') == FACTORS


def test_backtests_use_each_sides_predictions(paired):
    root, sid, days, base, ext, r = paired
    for x in r['subruns']['backtests']:
        doc = read(x['output'], 'config.json'); assert doc['scores']['research_run_id'] == r['subruns']['arms'][x['arm']]['run_id'] and doc['scores']['evidence'] == 'exploratory_prediction'
        assert doc['config']['scores']['model'] == x['model']


def test_paired_run_reproduces_and_detects_tampering(paired):
    root, sid, days, base, ext, r = paired; rep = reproduce(root, r['output'])
    assert rep['status'] == 'success' and rep['reproduction']['result'] == 'match'
    c = read(rep['output'], 'comparison.json'); assert set(c['subruns']) >= {'arm_base', 'arm_extended', 'backtest_ridge_base'} and all(v['result'] == 'match' for v in c['subruns'].values())
    p = Path(r['output']) / 'paired_eval.json'; backup = p.read_bytes(); ev = json.loads(backup)
    try:
        ev['models']['ridge']['rank_ic']['diff_mean'] += 0.5; p.write_text(json.dumps(ev), encoding = 'utf-8')
        bad = reproduce(root, r['output']); assert bad['status'] == 'mismatch' and 'paired_eval.json' in read(bad['output'], 'comparison.json')['source_integrity']['modified_files']
    finally: p.write_bytes(backup)


def test_intraday_factors_require_the_minute_pool(paired, tmp_path):
    root, sid, days, base, ext, r = paired
    with pytest.raises(ValueError, match = 'minute_pool'): run_research(root, snapshot = sid, factor_set = ext, label_h = 2, models = {'baseline_factor': 'rev_3'})
    with pytest.raises(ValueError, match = 'minute_pool'): run_paired(root, **cfg(sid, base, ext, minute_pool = False))
    short = tmp_path / 'short.yaml'; short.write_text(EXTENDED.replace('name: rev_3', 'name: rev_4'), encoding = 'utf-8')
    with pytest.raises(ValueError, match = '逐个相同'): check_extension(load_factor_set(base), load_factor_set(short))
    with pytest.raises(ValueError, match = '没有新增'): check_extension(load_factor_set(base), load_factor_set(base))


def test_missing_minute_data_blocks_instead_of_running_on_a_daily_universe(tmp_path):
    days, insts, rows = market(n_days = 60)
    cov = pd.DataFrame([{'instrument': i, 'status': 'no_events', 'verified_from': days[0].replace(year = 1990), 'verified_through': days[-1], 'has_start_basis': True,
                         'has_gap': False, 'confirmed_no_events': True} for i in insts])
    sid = snapshot(tmp_path, rows, inst = instruments(*insts), sessions = days, coverage = cov)
    (tmp_path / 'b.yaml').write_text(FACTORS, encoding = 'utf-8'); (tmp_path / 'e.yaml').write_text(EXTENDED, encoding = 'utf-8')
    r = run_paired(tmp_path, **cfg(sid, str(tmp_path / 'b.yaml'), str(tmp_path / 'e.yaml')))
    assert r['status'] == 'blocked' and r['blocked'][0]['blocked'][0]['kind'] == 'minute_data_missing'


def test_paired_through_cli_and_queue(paired, tmp_path):
    root, sid, days, base, ext, r = paired
    import yaml
    y = tmp_path / 'p.yaml'; y.write_text(yaml.safe_dump(cfg(sid, base, ext)), encoding = 'utf-8')
    try: code = main(['--root', str(root), 'paired', '--config', str(y)]) or 0
    except SystemExit as e: code = e.code
    assert code == 0
    q = Jobs(root); jid = q.submit('paired', cfg(sid, base, ext)); q.run_next(); assert q.get(jid)['status'] == 'success_limited'


def test_future_minute_bars_do_not_change_earlier_predictions(paired, tmp_path):
    root, sid, days, base, ext, r = paired; cut = days[LAST - 12]
    sid2, _, _, ext2 = make(tmp_path, shock = cut)
    def study(root_, sid_, ext_): return run_research(root_, **{k: v for k, v in cfg(sid_, base, ext_).items() if k in ('snapshot', 'label_h', 'universe', 'split', 'models')}, factor_set = ext_, minute_pool = True)
    pa, pb = (pd.read_parquet(Path(x['output']) / 'predictions.parquet') for x in (study(root, sid, ext), study(tmp_path, sid2, ext2)))
    part = lambda p, early: p[(p.decision_date < cut) == early].sort_values(['model_id', 'decision_date', 'instrument']).reset_index(drop = True)
    assert len(part(pa, True)) > 0 and len(part(pa, False)) > 0
    pd.testing.assert_frame_equal(part(pa, True), part(pb, True))                                                   # 分钟线改动之前的预测逐值不变
    now = lambda p: p[p.model_id == 'equal_blend'].score.to_numpy()
    assert not np.allclose(now(part(pa, False)), now(part(pb, False)))                                             # 改动确实影响了当天及以后


def test_blocked_backtest_is_reported_and_compared_only_on_the_clean_period(paired, tmp_path):
    """基础侧 Ridge 回测因持仓退市被账本阻断：不列绩效指标，成对比较只取阻断日之前双方都干净的区间"""
    root, sid, days, base, ext, r = paired; sub = json.loads(json.dumps(r['subruns'])); item = next(x for x in sub['backtests'] if x['model'] == 'ridge' and x['arm'] == 'base')
    copy = tmp_path / 'blocked_run'; shutil.copytree(item['output'], copy); item.update(output = str(copy), status = 'blocked')
    eq = json.loads((copy / 'equity.json').read_text(encoding = 'utf-8')); first = eq[len(eq) // 2]['date']
    st = json.loads((copy / 'status.json').read_text(encoding = 'utf-8')); st.update(status = 'blocked', issues = [{'date': first, 'instrument': '600001.SH', 'kind': 'delisted_holding', 'qty': 100}])
    (copy / 'status.json').write_text(json.dumps(st), encoding = 'utf-8')
    plan = pd.read_parquet(Path(r['subruns']['arms']['base']['output']) / 'split_plan.parquet'); c = PairedConfig.model_validate(json.loads((Path(r['output']) / 'config.json').read_text(encoding = 'utf-8'))['config'])
    ev = evaluate(c, sub, plan)['portfolio']['ridge']; run = ev['runs']['base']
    assert run['valid'] is False and run['blocked_from'] == first and run['blocked_kinds'] == ['delisted_holding'] and run['blocked_instruments'] == ['600001.SH'] and 'sharpe' not in run
    assert 'sharpe' in ev['runs']['extended'] and 'pair' not in ev and ev['full_period_metrics'] == 'unavailable'
    d = ev['diagnostic_prefix']; assert d['valid_portfolio_comparison'] is False and d['period']['truncated_before'] == first and d['period']['last'] < first and 'delisted_holding' in d['truncation_reason']
    assert d['period']['days'] < len(eq) - 1 and d['planned_period']['days'] == len(eq) - 1 and d['excluded_days'] == d['planned_period']['days'] - d['period']['days'] > 0
    full = evaluate(c, r['subruns'], plan)['portfolio']['ridge']; assert full['pair']['period']['truncated_before'] is None and full['pair']['period']['days'] == len(eq) - 1


def test_reevaluation_is_read_only_deterministic_and_pinned_to_source_hashes(paired, tmp_path):
    from observe.reeval import reevaluate
    from observe.replay import ReproduceRefused
    from tests.integration.helpers import tree_hash
    root, sid, days, base, ext, r = paired; src = Path(r['output']); runs = src.parent
    other = run_paired(root, **cfg(sid, base, ext, split = {'train': 15, 'valid': 10, 'test': 5, 'holdout': 5}))          # 训练窗更短的第二个实验：测试日期与主实验部分重叠
    before = tree_hash(runs); a = reevaluate(root, {'main': src, 'short': other['output']}); after = tree_hash(runs)
    assert {k: v for k, v in after.items() if not k.startswith(Path(a['output']).name)} == before                                  # 源实验、子实验目录一个字节都没变
    out = Path(a['output']); assert out.parent == runs and '-reeval-' in out.name and read(out, 'status.json')['kind'] == 'reeval'
    doc = read(out, 'config.json'); assert doc['sources']['main']['paired_dir'] == src.name and len(doc['sources']['main']['manifest_sha256']) == 64
    assert all(len(v) == 64 for arm in doc['sources']['main']['arms'].values() for k, v in arm.items() if k.endswith('.parquet'))
    res = read(out, 'reeval.json')
    assert res['old_vs_new']['main']['summary']['models/rank_ic']['changed'] == 0 and res['old_vs_new']['main']['summary']['models/rank_ic']['metrics'] > 0             # 预测与标签都是有限值：秩 IC 不变
    assert all(v['identical'] for d in res['selection_replay'].values() for v in d.values())                                          # 选参重放与原保存值逐项一致 ⇒ 预测不受影响
    assert res['holdout_boundary']['main']['extended']['boundary_labels_used_by_saved_diagnostics']['label_rows'] > 0
    assert res['common_test_dates']['common_days'] > 0 and set(res['common_test_dates']['effect_difference']) >= {'ridge', 'equal_blend'} and 'topn_daily.parquet' in read(out, 'manifest.json')['files']
    assert set(res['portfolio_common']) <= {'ridge', 'equal_blend'} and read(out, 'paired_eval_new_main.json')['design']['label_rule']['holdout_start'] is not None
    again = reevaluate(root, {'main': src, 'short': other['output']}); assert read(again['output'], 'reeval.json') == res and again['output'] != a['output']
    p = src / 'paired_eval.json'; backup = p.read_bytes()
    try:
        p.write_text(backup.decode('utf-8').replace('exploratory', 'x', 1), encoding = 'utf-8')
        with pytest.raises(ReproduceRefused, match = 'manifest'): reevaluate(root, {'main': src})
    finally: p.write_bytes(backup)


@pytest.fixture(scope = 'module')
def paired_missing(tmp_path_factory):
    """分钟股票池里有两只证券完全没有分钟线（模拟退市证券缺文件）"""
    root = tmp_path_factory.mktemp('paired_missing'); sid, days, base, ext = make(root, drop = ('600003.SH', '600004.SH'))
    return root, run_paired(root, **cfg(sid, base, ext))


def test_diagnostics_decompose_score_changes_and_pnl_and_are_read_only(paired_missing, tmp_path):
    from observe.diagnose import diagnose
    from tests.integration.helpers import tree_hash
    root, r = paired_missing; runs = Path(r['output']).parent; src = Path(r['output']); before = tree_hash(runs)
    d = diagnose(root, src); after = tree_hash(runs); res = read(d['output'], 'diagnostics.json')
    assert {k: v for k, v in after.items() if not k.startswith(Path(d['output']).name)} == before                                      # 源目录只读
    m = res['missingness']; assert m['no_minute_instruments'] == ['600003.SH', '600004.SH'] and m['candidates'] > 10 and 'no_minute' in m['label_profile'] and 'has_minute' in m['label_profile']
    assert set(m['top_n_slots']) == {'ridge', 'equal_blend'} and m['missing_marker_probe']['rows_missing_share'] > 0
    sd = res['score_decomposition']['models']
    for model in ('ridge', 'equal_blend'):
        v = sd[model]; assert v['rebuild_max_abs_diff'] < 1e-9 and v['rows_missing_any_score'] == 0                                 # 用保存的系数重建的加分钟侧分数与保存的预测逐值一致
        assert v['total_B_minus_A']['diff_mean'] == pytest.approx(v['fit_perturbation_b_minus_A']['diff_mean'] + v['minute_values_B_minus_b']['diff_mean'], abs = 1e-12)      # 总差 = 拟合扰动 + 分钟因子数值
        cc = v['complete_coverage_subsample']; assert set(cc['rank_ic_mean']) == {'A', 'B', 'b'} and '事后' in cc['note']
    assert sd['equal_blend']['fit_perturbation_b_minus_A']['diff_mean'] == 0                                                          # 等权合成没有拟合环节
    c = res['concentration']; assert set(c) == {'ridge', 'equal_blend'}
    for v in c.values():
        assert abs(v['arms']['base']['unreconciled_over_initial']) < 0.01 and v['names_common'] + v['names_extended_only'] > 0                 # 按证券的盈亏加总与净值变化对得上（合成数据无分红）
        assert v['decomposition_over_initial']['common_names'] + v['decomposition_over_initial']['extended_only_names'] + v['decomposition_over_initial']['base_only_names'] == pytest.approx(v['difference_extended_minus_base_over_initial'])
        assert 'no_minute_names_effect_over_initial' in v and v['valid_portfolio_comparison'] is True
    assert read(d['output'], 'status.json')['kind'] == 'diagnostic'
