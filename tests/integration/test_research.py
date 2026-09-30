"""研究流水线公开入口：临时快照 → run_research → 预测表 → run_offline 用同一执行入口回测；复现与防泄漏。"""
import json
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from observe.cli import main
from observe.jobs import Jobs
from observe.models import FeatureMismatch, _Model
from observe.replay import ReproduceRefused, reproduce, run_offline
from observe.research import run_research
from tests.integration.helpers import instruments, snapshot

FACTORS = """version: 1
min_obs_ratio: 1.0
factors:
  - name: rev_3
    expr: close_adj / ts_delay(close_adj, 3) - 1
    direction: -1
    desc: 3 日收益，短期反转
  - name: vol_5
    expr: ts_std(ret, 5)
    direction: -1
    desc: 5 日波动，低波动异象
  - name: amount_surge
    expr: amount / ts_mean(amount, 5)
    direction: -1
    desc: 放量后回落
"""


def market(n_inst = 40, n_days = 90, seed = 0, shock = None):
    """每只股票日收益 = -0.5 × 前一日收益 + 噪声：短期反转信号真实存在"""
    rng = np.random.default_rng(seed); days = [d.date() for d in pd.bdate_range('2023-01-02', periods = n_days)]
    insts = [f'6{k:05d}.SH' for k in range(1, n_inst + 1)]; rows = []
    for i in insts:
        r, px = 0.0, 10.0 + rng.uniform(0, 10)
        for k, d in enumerate(days):
            pre = px; r = -0.5 * r + rng.normal(0, 0.015); px = round(max(pre * (1 + r), 1.0), 2)
            if shock and d >= shock: px = round(px * 1.05, 2) if px * 1.05 < pre * 1.09 else px
            amt = float(rng.uniform(5e7, 2e8))
            rows.append({'date': d, 'instrument': i, 'open': pre, 'high': max(pre, px), 'low': min(pre, px), 'close': px, 'preclose': pre, 'volume': amt / px,
                         'amount': amt, 'turnover': 0.01, 'is_trading': True, 'is_st': False, 'board': 'main'})
    return days, insts, rows


def make(root, **kw):
    days, insts, rows = market(**kw)
    cov = pd.DataFrame([{'instrument': i, 'status': 'no_events', 'verified_from': date(1990, 1, 1), 'verified_through': days[-1], 'has_start_basis': True,
                         'has_gap': False, 'confirmed_no_events': True} for i in insts])
    sid = snapshot(root, rows, inst = instruments(*insts), sessions = days, coverage = cov)
    fs = Path(root) / 'factors.yaml'; fs.write_text(FACTORS, encoding = 'utf-8')
    return sid, days, str(fs)


def cfg(sid, fs):
    return {'snapshot': sid, 'factor_set': fs, 'label_h': 2,
            'universe': {'min_listed_sessions': 1, 'suspend_window': 5, 'max_suspended': 3, 'liquidity_window': 5, 'min_avg_amount': 1e6},
            'split': {'train': 20, 'valid': 10, 'test': 5, 'holdout': 5}, 'models': {'baseline_factor': 'rev_3', 'ridge_alphas': [1.0, 100.0], 'min_names': 10, 'top_n': 5}}


def read(path, name): return json.loads((Path(path) / name).read_text(encoding = 'utf-8'))


@pytest.fixture(scope = 'module')
def base(tmp_path_factory):
    root = tmp_path_factory.mktemp('research'); sid, days, fs = make(root)
    r = run_research(root, **cfg(sid, fs)); return root, sid, days, fs, r


def test_predictions_cover_only_test_windows_out_of_sample(base):
    root, sid, days, fs, r = base; out = Path(r['output'])
    assert r['status'] == 'success' and read(out, 'status.json')['kind'] == 'research'
    plan, pred = pd.read_parquet(out / 'split_plan.parquet'), pd.read_parquet(out / 'predictions.parquet')
    assert len(plan) >= 5 and set(pred.model_id) == {'single_factor', 'equal_blend', 'ridge'}
    assert not pred.duplicated(['model_id', 'decision_date', 'instrument']).any()
    for sp in plan.itertuples():
        p = pred[pred.split_id == sp.split_id]
        assert p.decision_date.min() >= sp.test_start and p.decision_date.max() <= sp.test_end and (p.fit_asof < sp.test_start).all()
    assert pred.decision_date.max() < plan.holdout_start.iloc[0]                                  # 最终留出区间没有预测
    ev = read(out, 'model_eval.json')
    assert ev['summary']['single_factor']['test_rank_ic_mean'] > 0.05                              # 预设的反转信号被基线捕捉到
    assert ev['summary']['ridge']['test_rank_ic_mean'] > 0.05 and 'ridge_vs_equal_blend' in ev['window_wins']
    assert read(out, 'factor_eval.json')['factors']['rev_3']['direction_adjusted_ic'] > 0.05


def test_saved_model_reproduces_predictions_and_rejects_feature_changes(base):
    root, sid, days, fs, r = base; out = Path(r['output'])
    m = _Model.load(out / 'models' / 'split00_ridge.json'); pred = pd.read_parquet(out / 'predictions.parquet')
    from observe.dataset import cross_sectional_preprocess
    fac = pd.read_parquet(out / 'factors.parquet'); X = cross_sectional_preprocess(fac.assign(eligible = True), ['rev_3', 'vol_5', 'amount_surge'])
    p0 = pred[(pred.model_id == 'ridge') & (pred.split_id == 0)].sort_values(['decision_date', 'instrument'])
    test = X[X.date.isin(set(p0.decision_date))].sort_values(['date', 'instrument'])
    assert np.array_equal(m.predict(test[m.features]), p0.score.to_numpy())
    with pytest.raises(FeatureMismatch): m.predict(test[m.features[::-1]])
    with pytest.raises(FeatureMismatch): m.predict(test[m.features[:-1]])


def test_future_prices_do_not_change_earlier_predictions(base, tmp_path):
    root, sid, days, fs, r = base
    sid2, _, fs2 = make(tmp_path, shock = days[-8])                                              # 只改最后几天的价格
    r2 = run_research(tmp_path, **cfg(sid2, fs2))
    a, b = pd.read_parquet(Path(r['output']) / 'predictions.parquet'), pd.read_parquet(Path(r2['output']) / 'predictions.parquet')
    early = lambda p: p[p.decision_date < days[-8] - pd.Timedelta(days = 10).to_pytimedelta()].sort_values(['model_id', 'decision_date', 'instrument']).reset_index(drop = True)
    pd.testing.assert_frame_equal(early(a), early(b))


def test_research_reproduces_and_detects_tampering(base):
    root, sid, days, fs, r = base
    rep = reproduce(root, r['output'])
    assert rep['status'] == 'success' and rep['reproduction']['result'] == 'match'
    src = Path(r['output']); pred = pd.read_parquet(src / 'predictions.parquet'); backup = (src / 'predictions.parquet').read_bytes()
    try:
        pred.loc[0, 'score'] += 1.0; pred.to_parquet(src / 'predictions.parquet', index = False)
        bad = reproduce(root, r['output']); c = read(bad['output'], 'comparison.json')
        assert bad['status'] == 'mismatch' and c['tables']['predictions']['columns'] == {'score': 1} and 'predictions.parquet' in c['source_integrity']['modified_files']
    finally: (src / 'predictions.parquet').write_bytes(backup)


def test_predictions_drive_the_single_execution_entry(base):
    root, sid, days, fs, r = base
    params = {'snapshot': sid, 'initial_cash': 1_000_000, 'scores': {'source': 'predictions', 'run': r['output'], 'model': 'ridge'},
              'portfolio': {'n': 5, 'max_weight': 0.2, 'rebalance_every': 2, 'buffer': 2, 'max_sell': 5}, 'execution': {'liquidity_window': 5}}
    run = run_offline(root, **params); out = Path(run['output'])
    pred = pd.read_parquet(Path(r['output']) / 'predictions.parquet'); ridge = pred[pred.model_id == 'ridge']
    assert run['status'] == 'success' and run['start'] == str(ridge.decision_date.min()) and run['end'] == str(ridge.decision_date.max())
    scores = read(out, 'scores.json'); doc = read(out, 'config.json')
    assert len(scores) == len(ridge) and doc['scores']['source'] == 'predictions' and doc['scores']['model'] == 'ridge'
    fills = read(out, 'fills.json'); assert fills and len({f['instrument'] for f in fills}) >= 5
    first = min(ridge.decision_date); top = ridge[ridge.decision_date == first].nlargest(5, 'score').instrument
    assert {f['instrument'] for f in fills if f['side'] == 'buy' and f['decision_date'] == str(first)} == set(top)
    assert reproduce(root, run['output'])['reproduction']['result'] == 'match'
    p = Path(r['output']) / 'predictions.parquet'; backup = p.read_bytes()
    try:
        ridge2 = pred.copy(); ridge2.loc[0, 'score'] += 1; ridge2.to_parquet(p, index = False)
        with pytest.raises(ReproduceRefused, match = 'predictions.parquet'): reproduce(root, run['output'])
    finally: p.write_bytes(backup)


def test_research_status_through_cli_and_queue(base, tmp_path):
    root, sid, days, fs, r = base
    import yaml
    y = tmp_path / 'r.yaml'; y.write_text(yaml.safe_dump(cfg(sid, fs)), encoding = 'utf-8')
    try: code = main(['--root', str(root), 'research', '--config', str(y)]) or 0
    except SystemExit as e: code = e.code
    assert code == 0
    q = Jobs(root); jid = q.submit('research', cfg(sid, fs)); q.run_next(); assert q.get(jid)['status'] == 'success'
    short = {**cfg(sid, fs), 'split': {'train': 200, 'valid': 10, 'test': 5, 'holdout': 5}}
    jid = q.submit('research', short); q.run_next(); j = q.get(jid)
    assert j['status'] == 'blocked' and 'insufficient_history_for_split' in j['result']


def test_development_analysis_ignores_prices_from_the_holdout_start(base, tmp_path):
    """只改最终留出起点及以后的价格：开发区间的因子评价与测试窗评价逐值不变；不用统一规则（只按决策日过滤）时因子评价会变"""
    root, sid, days, fs, r = base; out = Path(r['output']); hold = pd.Timestamp(pd.read_parquet(out / 'split_plan.parquet').holdout_start.iloc[0]).date()
    sid2, _, fs2 = make(tmp_path, shock = hold); r2 = run_research(tmp_path, **cfg(sid2, fs2)); out2 = Path(r2['output'])
    for name in ('factor_eval.json', 'model_eval.json'): assert read(out, name) == read(out2, name)
    rule = read(out, 'factor_eval.json')['label_rule']; assert rule['holdout_start'] == str(hold)
    from observe.research import _factor_eval
    names = list(read(out, 'factor_eval.json')['factors']); dirs = {n: -1 for n in names}
    fac, fac2 = pd.read_parquet(out / 'factors.parquet'), pd.read_parquet(out2 / 'factors.parquet'); lab, lab2 = pd.read_parquet(out / 'labels.parquet'), pd.read_parquet(out2 / 'labels.parquet')
    dev = sorted(d for d in set(fac.date) if pd.Timestamp(d).date() < hold)
    assert _factor_eval(fac, lab, names, dirs, dev, 10, hold) == _factor_eval(fac2, lab2, names, dirs, dev, 10, hold)
    assert _factor_eval(fac, lab, names, dirs, dev, 10, None) != _factor_eval(fac2, lab2, names, dirs, dev, 10, None)      # 旧规则会把成熟时点在留出期内的标签算进来


def _replay_params(sid, run, **scores): return {'snapshot': sid, 'initial_cash': 1_000_000, 'scores': {'source': 'predictions', 'run': str(run), 'model': 'ridge', **scores},
                                             'portfolio': {'n': 5, 'max_weight': 0.2, 'rebalance_every': 2, 'buffer': 2, 'max_sell': 5}, 'execution': {'liquidity_window': 5}}


def test_predictions_are_checked_against_snapshot_candidates_fit_time_and_range(base, tmp_path):
    import shutil
    from observe.data.store import Store
    root, sid, days, fs, r = base; src = Path(r['output'])
    other = Store(root).snapshot('same content, different snapshot')                                          # 内容相同、编号不同的快照
    with pytest.raises(ValueError, match = '快照.*不一致'): run_offline(root, **_replay_params(other, src))
    ok = run_offline(root, **_replay_params(other, src, allow_cross_snapshot = True)); doc = read(ok['output'], 'config.json')
    assert ok['status'] == 'success_limited' and doc['scores']['cross_snapshot_scenario'] == {'research_snapshot': sid, 'replay_snapshot': other}
    assert any(x['kind'] == 'cross_snapshot_scenario' for x in ok['limitations'])                                # 对照情景写进限制，双方版本都记录
    with pytest.raises(ValueError, match = '没有模型'): run_offline(root, **_replay_params(sid, src, model = 'lightgbm'))
    pred = pd.read_parquet(src / 'predictions.parquet'); first, last = pred[pred.model_id == 'ridge'].decision_date.min(), pred[pred.model_id == 'ridge'].decision_date.max()
    with pytest.raises(ValueError, match = '超出'): run_offline(root, **{**_replay_params(sid, src), 'start': str(pd.Timestamp(first).date() - pd.Timedelta(days = 4))})
    with pytest.raises(ValueError, match = '超出'): run_offline(root, **{**_replay_params(sid, src), 'end': str(pd.Timestamp(last).date() + pd.Timedelta(days = 4))})
    bad = tmp_path / 'bad_fit'; shutil.copytree(src, bad); p = pd.read_parquet(bad / 'predictions.parquet'); p['fit_asof'] = p.decision_date; p.to_parquet(bad / 'predictions.parquet', index = False)
    with pytest.raises(ValueError, match = 'fit_asof'): run_offline(root, **_replay_params(sid, bad))
    off = tmp_path / 'off_universe'; shutil.copytree(src, off); u = pd.read_parquet(off / 'universe.parquet'); u.loc[u.decision_date == first, 'eligible'] = False; u.to_parquet(off / 'universe.parquet', index = False)
    with pytest.raises(ValueError, match = '不在该实验的研究候选'): run_offline(root, **_replay_params(sid, off))


def test_holdout_is_never_silently_dropped_and_undefined_selection_blocks(tmp_path):
    root = tmp_path / 'short'; root.mkdir(); sid, days, fs = make(root, n_days = 46)                                     # 预热 6 天后剩 40 个可用决策日
    small = {'train': 10, 'valid': 5, 'test': 5}
    r = run_research(root, **{**cfg(sid, fs), 'split': {**small, 'holdout': 60}})
    assert r['status'] == 'blocked' and r['blocked'][0]['kind'] == 'insufficient_history' and '60' in r['blocked'][0]['detail'] and not (Path(r['output']) / 'predictions.parquet').exists()   # 没有生成五个没有留出的测试窗
    ok = run_research(root, **{**cfg(sid, fs), 'split': {**small, 'holdout': 0}})                                       # 显式 holdout = 0：不设最终留出，窗口正常
    assert ok['status'] == 'success' and ok['summary']['holdout_start'] is None and ok['summary']['windows'] == 5
    tight = run_research(root, **{**cfg(sid, fs), 'split': {**small, 'holdout': 30}})                                    # 留出后只剩 10 个开发日，不足一个窗口
    assert tight['status'] == 'blocked' and tight['blocked'][0]['kind'] == 'insufficient_history_for_split' and '已扣除 30' in tight['blocked'][0]['detail']
    undefined = run_research(root, **{**cfg(sid, fs), 'split': {**small, 'holdout': 5}, 'models': {**cfg(sid, fs)['models'], 'min_names': 10000}})
    assert undefined['status'] == 'blocked' and undefined['blocked'][0]['kind'] == 'selection_undefined' and any(t['undefined'] for t in undefined['blocked'][0]['candidates'])
