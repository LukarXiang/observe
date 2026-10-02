"""完整后端流程、预测复用、成本情景、API / CLI / 队列与离线复现。"""
import json
from pathlib import Path

import pandas as pd
import pytest
from fastapi.testclient import TestClient

from observe.api.app import create_app
from observe.cli import main, run_kind
from observe.experiments import run_experiment, run_variant
from observe.jobs import Jobs
from observe.models import LGBMModel, RidgeModel
from observe.replay import reproduce, run_offline
from observe.runs import RunRegistry, file_sha
from observe.data.store import Store
from observe.data.indices import update_indices
from tests.integration.test_research import cfg, make, read
from tests.unit.test_indices import Source, response

pytest.importorskip('lightgbm')


@pytest.fixture(scope = 'module')
def full(tmp_path_factory):
    root = tmp_path_factory.mktemp('complete'); sid, days, fs = make(root, n_days = 55)
    store = Store(root); idx = response(days = days)
    for name in ('open', 'close', 'preclose'): idx[name] = [100 + k for k in range(len(days))]
    idx['high'] = idx.close + 1; idx['low'] = idx.close - 1
    dates = pd.date_range(days[0], days[-1]); cal = pd.DataFrame({'date': dates.date, 'is_open': dates.isin(pd.to_datetime(days))})
    inst = pd.concat([store.load('instruments'), pd.DataFrame([{'instrument': '000300.SH', 'kind': 'index', 'board': 'index'}])], ignore_index = True)
    store.publish(store.write_batch({'calendar': {'all': store.write_partition('calendar', 'all', cal)},
                                     'instruments': {'all': store.write_partition('instruments', 'all', inst)}}))
    assert update_indices(root, days[0], days[-1], source = Source({'sh.000300': idx}))['status'] == 'published'
    run_kind(root, 'data_audit', {}); sid = store.snapshot('经公开指数入库入口审计发布的完整实验')
    config = cfg(sid, fs); config['models'].update(lgbm = [{'num_boost_round': 12, 'early_stopping_rounds': 3, 'num_leaves': 4, 'min_data_in_leaf': 10}], num_threads = 1)
    config.update(n_boot = 100, portfolio = {'n': 5, 'max_weight': .2, 'rebalance_every': 2, 'buffer': 2, 'max_sell': 5}, execution = {'slippage': .001, 'liquidity_window': 5})
    return root, config, run_experiment(root, **config)


def test_four_models_and_every_cost_scenario_share_the_frozen_predictions(full):
    root, config, result = full; out = Path(result['output']); report = read(out, 'report.json')
    assert result['status'] == 'success' and len(result['subruns']['backtests']) == 12
    research = Path(result['subruns']['research']['output']); pred = pd.read_parquet(research / 'predictions.parquet'); plan = pd.read_parquet(research / 'split_plan.parquet')
    assert set(pred.model_id) == {'single_factor', 'equal_blend', 'ridge', 'lgbm'} and pred.decision_date.max() < plan.holdout_start.iloc[0]
    assert len(report['model']['comparisons']) == 4 and all(x['valid_primary_comparison'] for x in report['model']['comparisons'].values())
    for model in set(pred.model_id):
        rows = [r for r in result['subruns']['backtests'] if r['model'] == model]; docs = {r['scenario']: read(r['output'], 'config.json') for r in rows}
        assert len({d['scores']['predictions_sha256'] for d in docs.values()}) == 1
        assert docs['fees_x2']['config']['execution']['fee_multiplier'] == 2 and docs['slippage_x2']['config']['execution']['slippage'] == .002
    assert all(r['valid_performance'] for r in report['portfolio']) and len(report['model']['evaluation']['lgbm_fit_records']) == len(plan)
    assert RunRegistry(root / 'runs').get(out.name)['status'] == 'success'
    assert len(result['subruns']['benchmarks']) == 3 and report['benchmark']['available']
    assert all(x['benchmarks']['000300.SH']['benchmark_missing_days'] == 0 and x['benchmarks']['universe_equal']['available'] for x in report['benchmark']['comparisons'])
    first = result['subruns']['benchmarks'][0]; equal_scores = read(first['output'], 'scores.json')
    uni = pd.read_parquet(research / 'universe.parquet'); scores = pd.DataFrame(equal_scores); scores['decision_date'] = pd.to_datetime(scores.decision_date).dt.date
    expected = uni[uni.eligible & uni.decision_date.between(plan.test_start.min(), plan.test_end.max())]
    assert set(zip(scores.decision_date, scores.instrument)) == set(zip(expected.decision_date, expected.instrument)) and set(scores.score) == {0}


def test_variant_cannot_train_and_preserves_all_existing_parent_files(full, monkeypatch):
    root, config, result = full; source = Path(result['output']); before = {n: file_sha(source / n) for n in read(source, 'manifest.json')['files']}
    def forbidden(*args, **kwargs): raise AssertionError('组合变体不得重新拟合')
    monkeypatch.setattr(RidgeModel, 'fit', forbidden); monkeypatch.setattr(LGBMModel, 'fit', forbidden)
    variant = run_variant(root, result['run_id'], {'model': 'lgbm', 'portfolio': {'n': 3}})
    assert variant['status'] == 'success' and variant['training_reused'] and source.resolve() in Path(variant['output']).resolve().parents
    doc = read(variant['output'], 'config.json')['config']
    assert doc['portfolio']['n'] == 3 and doc['portfolio']['rebalance_every'] == 2 and doc['execution']['slippage'] == .001
    assert {n: file_sha(source / n) for n in before} == before
    with pytest.raises(ValueError): run_variant(root, result['run_id'], {'snapshot': 'new-snapshot'})


def test_complete_experiment_reproduces(full):
    root, config, result = full; rep = reproduce(root, result['run_id'])
    assert rep['status'] == 'success' and rep['reproduction']['result'] == 'match' and rep['reproduction']['differences'] == 0
    children = [rep['subruns']['research'], *rep['subruns']['backtests'], *rep['subruns']['benchmarks']]
    assert all(not read(c['output'], 'cache.json')['enabled'] and read(c['output'], 'cache.json')['hits'] == 0 for c in children)


def test_api_cli_and_queue_query_the_same_frozen_results(full, capsys):
    root, config, result = full; c = TestClient(create_app(root)); rid = result['run_id']
    assert c.get('/api/runs', params = {'kind': 'experiment'}).json()[0]['kind'] == 'experiment'
    assert c.get(f'/api/runs/{rid}').json()['reports']['report']['header']['snapshot_id'] == config['snapshot']
    page = c.get(f'/api/runs/{rid}/predictions', params = {'limit': 2, 'offset': 1}).json()
    assert len(page['rows']) == 2 and page['total'] > 2
    selected = c.get(f'/api/runs/{rid}/predictions', params = {'model': 'lgbm', 'limit': 2}).json()
    assert selected['total'] * 4 == page['total'] and {r['model_id'] for r in selected['rows']} == {'lgbm'}
    day = selected['rows'][0]['decision_date'][:10]
    dated = c.get(f'/api/runs/{rid}/predictions', params = {'model': 'lgbm', 'date': day, 'instrument': selected['rows'][0]['instrument']}).json()
    assert dated['total'] == 1 and dated['rows'][0]['model_id'] == 'lgbm'
    assert len(c.get(f'/api/runs/{rid}/equity', params = {'model': 'lgbm', 'limit': 2}).json()['rows']) == 2
    assert c.get(f'/api/runs/{rid}/equity', params = {'model': 'universe_equal'}).status_code == 200
    active = c.get(f'/api/runs/{rid}/benchmark_daily', params = {'model': 'lgbm', 'scenario': 'fees_x2', 'benchmark': '000300.SH', 'limit': 2}).json()
    assert len(active['rows']) == 2 and {r['benchmark_id'] for r in active['rows']} == {'000300.SH'} and {r['scenario'] for r in active['rows']} == {'fees_x2'}
    assert c.get(f'/api/runs/{rid}/positions', params = {'instrument': page['rows'][0]['instrument'], 'limit': 2}).status_code == 200
    assert c.get('/api/factors/rev_3/evaluation', params = {'run': rid}).status_code == 200
    assert c.post('/api/factors/validate', json = {'expr': 'ts_mean(close_adj, 5)'}).json()['lookback'] == 4
    assert c.post('/api/factors/validate', json = {'expr': "__import__('os')"}).status_code == 400
    assert c.get(f'/api/runs/{rid}/predictions', params = {'limit': -1}).status_code == 422
    assert c.get(f'/api/runs/{rid}/unknown').status_code == 400 and c.get('/api/runs/missing').status_code == 404
    assert c.post('/api/jobs', json = {'kind': 'experiment', 'params': {'snapshot': config['snapshot'], 'unknown': 1}}).status_code == 400
    assert main(['--root', str(root), 'runs', 'list', '--kind', 'experiment']) is None and rid in capsys.readouterr().out
    jid = c.post('/api/jobs', json = {'kind': 'factor_eval', 'params': {'run': rid, 'n_boot': 100}}).json()['job_id']; Jobs(root).run_next()
    job = Jobs(root).get(jid); assert job['status'] == 'success' and json.loads(job['result'])['factors'] == 3
    assert c.get('/api/runs/compare', params = {'ids': rid + ',' + result['subruns']['research']['run_id']}).status_code == 200


def test_windows_prediction_paths_are_relocated_by_identity(full):
    root, config, result = full; research = Path(result['subruns']['research']['output']); ref = rf'D:\projects\observe\data\runs\{research.name}'
    r = run_offline(root, snapshot = config['snapshot'], scores = {'source': 'predictions', 'run': ref, 'model': 'ridge'}, portfolio = {'n': 5, 'max_weight': .2})
    assert r['status'] == 'success' and read(r['output'], 'config.json')['scores']['run'] == str(research.resolve())
    assert reproduce(root, r['run_id'])['reproduction']['result'] == 'match'


def test_research_blocking_does_not_produce_portfolio_results(full):
    root, config, result = full
    blocked = run_experiment(root, **{**config, 'split': {'train': 200, 'valid': 10, 'test': 5, 'holdout': 5}})
    assert blocked['status'] == 'blocked' and not blocked['subruns']['backtests'] and not (Path(blocked['output']) / 'report.json').exists()


@pytest.mark.parametrize('case', ['duplicate', 'nonfinite', 'wrong_split'])
def test_invalid_frozen_predictions_never_enter_the_ledger(full, tmp_path, case):
    import shutil
    root, config, result = full; source = Path(result['subruns']['research']['output']); bad = tmp_path / case; shutil.copytree(source, bad)
    p = pd.read_parquet(bad / 'predictions.parquet')
    if case == 'duplicate': p = pd.concat([p, p[p.model_id == 'ridge'].iloc[:1]], ignore_index = True)
    elif case == 'nonfinite': p.loc[p.model_id == 'ridge', 'score'] = float('nan')
    else: p.loc[p.model_id == 'ridge', 'split_id'] = 999
    p.to_parquet(bad / 'predictions.parquet', index = False)
    with pytest.raises(ValueError, match = {'duplicate': '主键重复', 'nonfinite': '非有限', 'wrong_split': '测试窗不一致'}[case]):
        run_offline(root, snapshot = config['snapshot'], scores = {'source': 'predictions', 'run': str(bad), 'model': 'ridge'})


def test_empty_positions_and_fills_accept_date_and_instrument_filters(tmp_path):
    from observe.artifacts import run_table
    out = tmp_path / 'runs' / 'cash'; out.mkdir(parents = True)
    for name in ('positions_daily', 'fills'): (out / f'{name}.json').write_text('[]')
    for table in ('positions', 'fills'):
        assert run_table(tmp_path, 'cash', table, start = '2024-01-02', instrument = '600000.SH')['rows'] == []


def test_complete_experiment_integrity_graph_and_api(full):
    from observe.integrity import verify_run
    root, config, result = full
    report = verify_run(root, result['run_id'])
    assert report['status'] == 'ok' and report['summary']['runs'] == 17 and report['summary']['references'] == 16
    assert TestClient(create_app(root)).get(f"/api/runs/{result['run_id']}/verify").json() == report
