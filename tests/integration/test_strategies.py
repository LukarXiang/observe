import json
from pathlib import Path
import shutil

import numpy as np
import pandas as pd
import pytest

from observe.artifacts import run_table
from observe.api.app import create_app
from observe.jobs import Jobs
from observe.integrity import verify_run
from observe.replay import ReproduceRefused, reproduce
from observe.strategies import StrategyConfig, _generate_signals, run_strategy
from observe.strategy_catalog import BP_SOURCE, catalog_strategies, strategy_id
from observe.runs import RunRegistry, canonical
from observe.runs import file_sha
from observe.data.store import Store
from observe.data.constituents import POLICY, VERSION
from tests.integration.helpers import bar, instruments, snapshot, tree_hash
from fastapi.testclient import TestClient


def _fixture(root):
    days = list(pd.bdate_range('2024-01-02', periods = 95).date); insts = [f'60000{k}.SH' for k in range(1, 5)] + ['000333.SZ']
    rows = []
    for k, inst in enumerate(insts):
        previous = 10.0
        for d in days:
            close = round(previous * 1.005, 2)
            rows.append({**bar(d, inst, previous, pre = previous, close = close), 'pe_ttm': [10, 20, 0, -5, 8][k], 'pb_mrq': [1, 2, 0, -1, 3][k]})
            previous = close
    coverage = pd.DataFrame([{'instrument': i, 'status': 'no_events', 'verified_from': days[0], 'verified_through': days[-1], 'has_start_basis': True,
                              'has_gap': False, 'confirmed_no_events': True} for i in insts])
    sid = snapshot(root, rows, instruments(*insts), sessions = days, coverage = coverage)
    source = Path(root) / 'original.txt'; source.write_text('# synthetic source\ndef initialize(context):\n    pass\n', encoding = 'utf-8')
    config = {'snapshot': sid, 'source_path': str(source), 'start': days[25], 'end': days[-1], 'initial_cash': 100000, 'parameters': {'top_fraction': .5},
              'universe': {'boards': ['main'], 'min_listed_sessions': 1, 'exclude_st': False, 'suspend_window': 1, 'max_suspended': 0, 'liquidity_window': 1, 'min_avg_amount': 0},
              'portfolio': {'construction': 'target_weights', 'n': 1, 'max_weight': 1, 'buffer': 0, 'max_sell': None, 'rebalance_every': 1,
                            'rebalance_frequency': 'daily', 'refill_between_rebalance': True}, 'execution': {'slippage': .001, 'liquidity_window': 2}}
    return sid, days, config


@pytest.mark.parametrize('implementation', ['bp_component_v1', 'ep_component_v1', 'ma10_ma20_v1'])
def test_rule_strategy_unique_ledger_frozen_sources_and_reproduce(tmp_path, monkeypatch, implementation):
    _, _, config = _fixture(tmp_path)
    def forbidden(*a, **kw): raise AssertionError('规则策略不得拟合模型或联网')
    monkeypatch.setattr('observe.research._fit_predict', forbidden)
    monkeypatch.setattr('observe.data.sources.baostock.BaoStock.session', forbidden)
    result = run_strategy(tmp_path, implementation = implementation, **config); out = Path(result['output'])
    assert result['status'] == 'success_limited' and len(result['subruns']['backtests']) == 3
    targets = pd.read_parquet(out / 'targets.parquet')
    assert targets.groupby('decision_date').weight.sum().eq(1).all()
    if implementation == 'bp_component_v1':
        f = pd.read_parquet(out / 'factors.parquet'); first = f[f.date == f.date.min()].set_index('instrument')
        assert first.loc['600001.SH', 'value'] == 1 and first.loc['600002.SH', 'value'] == .5 and np.isnan(first.loc['600003.SH', 'value'])
        assert first.loc['600004.SH', 'value'] == -1
        stock_target = targets[(targets.decision_date == targets.decision_date.min()) & targets.instrument.ne('CASH')]
        assert stock_target.instrument.tolist() == ['600001.SH'] and stock_target.weight.tolist() == [1]
    assert verify_run(tmp_path, out)['status'] == 'ok'
    before = tree_hash(out); again = reproduce(tmp_path, out)
    assert again['reproduction']['result'] == 'match' and again['reproduction']['differences'] == 0
    assert before == tree_hash(out)
    cache = json.loads((Path(again['output']) / 'cache.json').read_text())
    assert not cache['enabled']
    assert run_table(tmp_path, out, 'targets')['total'] == len(targets)
    assert run_table(tmp_path, out, 'equity')['total'] > 0
    (out / 'source.original').write_text('modified', encoding = 'utf-8')
    with pytest.raises(ReproduceRefused): reproduce(tmp_path, out)


def test_catalog_preserves_gb18030_sources_duplicate_mapping_and_ids(tmp_path):
    source = tmp_path / 'source'; source.mkdir()
    text = '# 标题：测试\nfrom jqdata import *\ndef f():\n    get_index_stocks("000300.XSHG")\n    get_ticks("600000.XSHG")\n    order_value("600000.XSHG", 1000)\n'
    (source / 'first.txt').write_bytes(text.encode('gb18030')); (source / 'second.txt').write_text(text, encoding = 'utf-8')
    (source / 'research.py').write_text('import numpy\nx = 1\n')
    r = catalog_strategies(tmp_path, source); records = json.loads((Path(r['output']) / 'catalog.json').read_text())
    assert r['files'] == 3 and r['distinct_contents'] == 2
    assert records[0]['encoding'] == 'gb18030' and records[2]['duplicate_of'] == records[0]['strategy_id']
    assert records[2]['status'] == '重复版本' and records[0]['status'] == '暂不可复现'
    assert catalog_strategies(tmp_path, source)['catalog_id'] == r['catalog_id']


def test_strategy_reproduces_after_moving_data_to_another_root(tmp_path):
    original = tmp_path / 'original'; moved = tmp_path / 'moved'
    _, _, config = _fixture(original)
    result = run_strategy(original, implementation = 'bp_component_v1', **config)
    shutil.copytree(original, moved, ignore = shutil.ignore_patterns('registry.sqlite', 'registry.sqlite-*'))
    original.rename(tmp_path / 'archived-original')  # 原绝对路径不存在；冻结 JSON 仍逐字保留。
    source = moved / 'runs' / result['run_id']; before = tree_hash(source)
    RunRegistry(moved / 'runs').index()
    assert verify_run(moved, result['run_id'])['status'] == 'ok'
    assert run_table(moved, result['run_id'], 'equity')['total'] > 0
    again = reproduce(moved, result['run_id'])
    assert again['reproduction']['result'] == 'match' and again['reproduction']['differences'] == 0
    assert tree_hash(source) == before


def test_catalog_dated_variant_evidence_does_not_claim_original_complete(tmp_path):
    source = tmp_path / 'source'; file = source / BP_SOURCE; file.parent.mkdir(parents = True)
    file.write_text('def f():\n    order_value("600000.XSHG", 1000)\n')
    proof = {'strategy_id': strategy_id(BP_SOURCE), 'source_sha256': file_sha(file), 'implementation': 'bp_csi800_weekly_v1',
             'scope': '中证800周频近似版本', 'historical_constituents': {'policy': POLICY, 'strict_evidence': False}}
    dest = tmp_path / 'catalog/strategies/implementation-evidence.json'; dest.parent.mkdir(parents = True)
    dest.write_text(json.dumps([proof]))
    result = catalog_strategies(tmp_path, source); item = json.loads((Path(result['output']) / 'catalog.json').read_text())[0]
    assert item['status'] == '已验证' and not item['original_strategy_complete']
    assert 'bp_csi800_weekly_v1' in item['implementations'] and '缺中证800历史成分' not in item['gaps']
    assert any('日精度未证明' in g for g in item['gaps'])
    file.write_text('modified source')
    changed = catalog_strategies(tmp_path, source)
    assert json.loads((Path(changed['output']) / 'catalog.json').read_text())[0]['status'] == '实现中'


def test_strategy_api_queue_and_financial_query_share_services(tmp_path):
    sid, _, config = _fixture(tmp_path); client = TestClient(create_app(tmp_path))
    config['implementation'] = 'bp_component_v1'; config = canonical(config)
    response = client.post('/api/jobs', json = {'kind': 'rule_strategy', 'params': config})
    assert response.status_code == 200
    jid = response.json()['job_id']; Jobs(tmp_path).run_next()
    job = Jobs(tmp_path).get(jid)
    assert job['status'] == 'success_limited'
    query = client.get('/api/data/financial-history', params = {'snapshot': sid, 'fields': 'net_profit_ytd', 'instruments': '000333.SZ', 'decision_time': '2024-06-28 16:00'})
    assert query.status_code == 200 and query.json()['coverage']['visible_rows'] == 0
    assert client.get('/api/data/financial-history', params = {'snapshot': sid, 'fields': 'bad', 'instruments': '000333.SZ', 'decision_time': '2024-06-28'}).status_code == 400
    assert client.post('/api/jobs', json = {'kind': 'financial_import', 'params': {'annual': 'x'}}).status_code == 400
    partial = {**config, 'universe': {'boards': ['main']}}
    assert client.post('/api/jobs', json = {'kind': 'rule_strategy', 'params': partial}).status_code == 400


def test_future_prices_and_valuation_never_change_previous_rule_targets(tmp_path):
    sid, days, config = _fixture(tmp_path); store = Store(tmp_path); config['implementation'] = 'bp_component_v1'; cfg = StrategyConfig.model_validate(config)
    data = {t: store.load(t, sid) for t in ('calendar', 'instruments', 'bars_1d', 'adj_factors', 'adj_coverage')}
    before, after = tmp_path / 'before', tmp_path / 'after'; before.mkdir(); after.mkdir(); _generate_signals(cfg, data, before)
    bars = data['bars_1d'].copy(); future = bars.date >= days[50]
    bars.loc[future, 'pb_mrq'] *= 99; bars.loc[future, 'close'] *= 1.5
    _generate_signals(cfg, {**data, 'bars_1d': bars}, after)
    for name, column in [('factors', 'date'), ('targets', 'decision_date')]:
        a, b = pd.read_parquet(before / f'{name}.parquet'), pd.read_parquet(after / f'{name}.parquet')
        pd.testing.assert_frame_equal(a[a[column] < days[50]], b[b[column] < days[50]])


def _index_fixture(root):
    sid, days, config = _fixture(root); st = Store(root)
    changes = {}
    for table in ('instruments', 'bars_1d', 'adj_coverage'):
        frame_ = st.load(table, sid); frame_['instrument'] = frame_.instrument.replace({'000333.SZ': '688001.SH'})
        if table == 'instruments': frame_.loc[frame_.instrument.eq('688001.SH'), 'board'] = 'star'
        changes[table] = {'all' if table != 'bars_1d' else '2024': st.write_partition(table, 'all' if table != 'bars_1d' else '2024', frame_)}
    st.publish(st.write_batch(changes)); sid = st.snapshot('star fixture')
    extras = [f'601{k:03d}.SH' for k in range(797)]
    master = pd.concat([st.load('instruments', sid), instruments(*extras)], ignore_index = True)
    members = ['600002.SH', '600004.SH', '688001.SH', *extras]
    frame = pd.DataFrame([{'date': day, 'index': '000906.SH', 'instrument': i, 'source_date': day, 'component_index': '000300.SH' if k < 300 else '000905.SH',
                           'source_name': i, 'source_sha256': 'synthetic', 'source_lag_days': 0, 'policy': POLICY, 'processing_version': VERSION, 'strict_usable': False}
                          for day in days[25:] for k, i in enumerate(members)])
    st.publish(st.write_batch({'instruments': {'all': st.write_partition('instruments', 'all', master)}, 'index_constituents': {'2024': st.write_partition('index_constituents', '2024', frame)}}))
    config.update(snapshot = st.snapshot('historical index'), index_universe = {'index': '000906.SH', 'policy': POLICY}, rules = 'configs/rule_profiles/csi800_daily_v1.yaml')
    config['universe'].update(boards = ['main', 'star'], min_listed_sessions = 6)
    return days, config, frame


@pytest.mark.parametrize('implementation', ['bp_csi800_weekly_v1', 'ep_csi800_weekly_v1'])
def test_historical_index_strategy_freezes_memberships_and_reproduces(tmp_path, monkeypatch, implementation):
    _, config, _ = _index_fixture(tmp_path)
    def forbidden(*args, **kwargs): raise AssertionError('策略运行和复现必须离线')
    monkeypatch.setattr('observe.data.sources.baostock.BaoStock.session', forbidden)
    result = run_strategy(tmp_path, implementation = implementation, **config); out = Path(result['output'])
    assert result['status'] == 'success_limited'
    targets = pd.read_parquet(out / 'targets.parquet'); first = targets[targets.decision_date == targets.decision_date.min()]
    expected = '600002.SH' if implementation.startswith('bp_') else '688001.SH'
    assert first[first.instrument != 'CASH'].instrument.tolist() == [expected]  # BP:1/2>1/3；EP:1/8>1/20；600001不属于指数
    coverage = pd.read_parquet(out / 'signal_coverage.parquet')
    assert coverage.index_members.eq(800).all() and coverage.index_members_with_bars.eq(3).all()
    manifest = json.loads((out / 'data_manifest.json').read_text()); assert manifest['used']['index_constituents']
    assert not manifest['index_universe']['strict_evidence'] and verify_run(tmp_path, out)['status'] == 'ok'
    before = tree_hash(out); copied = reproduce(tmp_path, out)
    assert copied['reproduction']['result'] == 'match' and copied['reproduction']['differences'] == 0 and before == tree_hash(out)


def test_future_constituents_do_not_change_previous_signals_and_missing_dates_block(tmp_path):
    days, config, frame = _index_fixture(tmp_path); st = Store(tmp_path)
    config['implementation'] = 'bp_csi800_weekly_v1'; cfg = StrategyConfig.model_validate(config)
    data = {t: st.load(t, config['snapshot']) for t in ('calendar', 'instruments', 'bars_1d', 'adj_factors', 'adj_coverage', 'index_constituents')}
    before, after = tmp_path / 'before', tmp_path / 'after'; before.mkdir(); after.mkdir(); _generate_signals(cfg, data, before)
    future = frame.date >= days[50]; changed = frame.copy(); changed.loc[future & changed.instrument.eq('600002.SH'), 'instrument'] = '600001.SH'
    _generate_signals(cfg, {**data, 'index_constituents': changed}, after)
    for name, column in [('universe', 'decision_date'), ('factors', 'date'), ('targets', 'decision_date')]:
        a, b = pd.read_parquet(before / f'{name}.parquet'), pd.read_parquet(after / f'{name}.parquet')
        pd.testing.assert_frame_equal(a[a[column] < days[50]], b[b[column] < days[50]])
    from observe.execution import InputBlocked
    with pytest.raises(InputBlocked): _generate_signals(cfg, {**data, 'index_constituents': frame[frame.date != days[30]]}, tmp_path / 'missing')
    with pytest.raises(ValueError, match = '显式声明'): StrategyConfig.model_validate({k: v for k, v in config.items() if k != 'index_universe'})
    with pytest.raises(ValueError, match = '上市前5个交易日'):
        StrategyConfig.model_validate({**config, 'universe': {**config['universe'], 'min_listed_sessions': 1}})
