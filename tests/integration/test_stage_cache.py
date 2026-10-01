from pathlib import Path

import pandas as pd

from observe.data.store import Store
from observe.experiments import run_experiment
from observe.replay import read_core, reproduce
from observe.research import run_research
from tests.integration.test_research import cfg, make, read


def events(result): return read(result['output'], 'cache.json')['events']


def test_changing_one_formula_only_computes_that_factor_and_preserves_unrelated_stages(tmp_path, monkeypatch):
    from observe import research
    sid, days, fs = make(tmp_path, n_days = 55); config = cfg(sid, fs); a = run_research(tmp_path, **config)
    calls, original = [], research.factor_frame
    def track(spec, *args, **kwargs): calls.extend(f['name'] for f in spec['factors']); return original(spec, *args, **kwargs)
    monkeypatch.setattr(research, 'factor_frame', track)
    Path(fs).write_text(Path(fs).read_text().replace('expr: ts_std(ret, 5)', 'expr: -ts_std(ret, 5)'))
    b = run_research(tmp_path, **config); rows = events(b)
    assert calls == ['vol_5']
    assert [e['result'] for e in rows if e['stage'] == 'factor'] == ['hit', 'miss', 'hit']
    assert all(e['result'] == 'hit' for e in rows if e['stage'] in ('universe', 'labels', 'splits'))
    assert next(e for e in rows if e['stage'] == 'models')['result'] == 'miss'
    fa, fb = (pd.read_parquet(Path(r['output']) / 'factors.parquet') for r in (a, b))
    pd.testing.assert_frame_equal(fa[['date', 'instrument', 'rev_3', 'amount_surge']], fb[['date', 'instrument', 'rev_3', 'amount_surge']])


def test_label_h_and_snapshot_invalidate_the_correct_upstream_stages(tmp_path):
    sid, days, fs = make(tmp_path, n_days = 55); config = cfg(sid, fs); run_research(tmp_path, **config)
    changed = run_research(tmp_path, **{**config, 'label_h': 3}); rows = events(changed)
    assert all(e['result'] == 'hit' for e in rows if e['stage'] in ('factor', 'universe', 'splits'))
    assert all(e['result'] == 'miss' for e in rows if e['stage'] in ('labels', 'models'))
    same_data_new_snapshot = Store(tmp_path).snapshot('快照身份变化仍全部重算')
    fresh = run_research(tmp_path, **{**config, 'snapshot': same_data_new_snapshot})
    assert all(e['result'] == 'miss' for e in events(fresh))


def test_cost_changes_reuse_training_and_diagnostics_then_identical_runs_reuse_ledgers(tmp_path, monkeypatch):
    from observe import replay, research
    sid, days, fs = make(tmp_path, n_days = 55); config = cfg(sid, fs)
    config.update(n_boot = 100, cost_scenarios = ['base'], portfolio = {'n': 5, 'max_weight': .2, 'rebalance_every': 2}, execution = {'slippage': .001, 'liquidity_window': 5})
    a = run_experiment(tmp_path, **config); config['execution']['slippage'] = .003
    def forbidden(*a, **k): raise AssertionError('缓存命中不能重训')
    monkeypatch.setattr(research, 'selection_stage', forbidden)
    b = run_experiment(tmp_path, **config)
    assert all(e['result'] == 'hit' for e in events(b['subruns']['research'])) and events(b)[0]['result'] == 'hit'
    assert all(events(r)[0]['result'] == 'miss' for r in b['subruns']['backtests'] + b['subruns']['benchmarks'])
    assert read(a['output'], 'report.json')['benchmark']['comparisons'][0]['benchmarks']['000300.SH']['available'] is False
    assert read(b['output'], 'report.json')['benchmark']['comparisons'][0]['benchmarks']['universe_equal']['available']
    def cannot_simulate(*a, **k): raise AssertionError('完全相同的账本阶段应命中')
    monkeypatch.setattr(replay, 'run_loop', cannot_simulate)
    c = run_experiment(tmp_path, **config)
    assert all(events(r)[0]['result'] == 'hit' for r in c['subruns']['backtests'] + c['subruns']['benchmarks'])
    for rb, rc in zip(b['subruns']['backtests'], c['subruns']['backtests']): assert read_core(rb['output']) == read_core(rc['output'])
    # 清除故障注入后，复现必须真跑并逐值一致；所有阶段标为 bypass。
    monkeypatch.undo(); rep = reproduce(tmp_path, b['run_id'])
    assert rep['reproduction']['result'] == 'match' and rep['reproduction']['differences'] == 0
    children = [rep['subruns']['research'], *rep['subruns']['backtests'], *rep['subruns']['benchmarks']]
    assert all(e['result'] == 'bypass' for r in children for e in events(r))
