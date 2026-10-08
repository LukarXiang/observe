import json
from contextlib import nullcontext
from pathlib import Path

import pandas as pd
import pytest

from observe.data.store import Store
from observe.execution import InputBlocked, build, load
from observe.replay import RunConfig, read_core, reproduce
from observe.runs import compare_tables, file_sha, write_json
from observe.strategies import StrategyConfig, _generate_signals, rule_source, run_strategy
from scripts.archive_strategy_progress import archive
from scripts.run_strategy_batch4 import protect, supplement_calendar
from tests.integration.helpers import tree_hash
from tests.integration.test_strategies import _fixture


def test_filtered_inputs_preserve_quotes_actions_and_missing_security(tmp_path):
    sid, days, _ = _fixture(tmp_path); store = Store(tmp_path); state = store.state(sid)
    actions = pd.DataFrame([{'instrument': i, 'ex_date': days[33], 'cash_per_share': .1, 'bonus_ratio': 0,
                            'rights_ratio': 0, 'rights_price': 0, 'record_date': days[32], 'pay_date': days[34],
                            'bonus_list_date': None, 'source': 'synthetic'} for i in ('000333.SZ', '600001.SH')])
    store.publish(store.write_batch({'corp_actions': {'all': store.write_partition('corp_actions', 'all', actions)}}))
    state = store.state(store.snapshot('scope actions fixture'))
    full = load(store, state, days[25], days[-1], 2)
    scoped = load(store, state, days[25], days[-1], 2, ['000333.SZ'])
    for name in ('bars_1d', 'instruments', 'corp_actions'):
        expected = full[name][full[name].instrument.eq('000333.SZ')].reset_index(drop = True)
        pd.testing.assert_frame_equal(scoped[name], expected)
    a = build(full, days[25], days[-1], liquidity_window = 2)
    b = build(scoped, days[25], days[-1], liquidity_window = 2)
    assert b.market == {d: {'000333.SZ': quotes['000333.SZ']} for d, quotes in a.market.items()}
    assert b.actions == {d: [action for action in actions if action['instrument'] == '000333.SZ'] for d, actions in a.actions.items()}
    assert scoped['partitions'] == full['partitions']
    with pytest.raises(InputBlocked, match = 'instrument_missing'):
        load(store, state, instruments = ['999999.SH'])
    with pytest.raises(ValueError, match = '非空'):
        load(store, state, instruments = [])


@pytest.mark.parametrize('implementation,parameters', [
    ('ma10_ma20_v1', {}), ('bollinger_breakout_corrected_v1', {'boll_window': 20, 'boll_std_multiplier': 2}),
    ('ma5_ma10_price_v1', {'short': 5, 'long': 10, 'buy_multiplier': 1}),
])
def test_scope_economic_equivalence_and_offline_reproduce(tmp_path, monkeypatch, implementation, parameters):
    _, _, config = _fixture(tmp_path)
    config.update(implementation = implementation, parameters = parameters, instrument = '000333.SZ')
    full = run_strategy(tmp_path, **config)
    filtered = run_strategy(tmp_path, **config, execution_instruments = ['000333.SZ'])
    assert full['status'] == filtered['status'] == 'success_limited'
    for name in ('universe', 'factors', 'scores', 'targets', 'signal_coverage'):
        pd.testing.assert_frame_equal(pd.read_parquet(Path(full['output']) / f'{name}.parquet'),
                                      pd.read_parquet(Path(filtered['output']) / f'{name}.parquet'))
    for a, b in zip(full['subruns']['backtests'], filtered['subruns']['backtests'], strict = True):
        difference, _ = compare_tables(read_core(a['output']), read_core(b['output']))
        assert not difference
    for name in ('report.json', 'benchmark_eval.json'):
        assert json.loads((Path(full['output']) / name).read_text()) == json.loads((Path(filtered['output']) / name).read_text())
    def forbidden(*a, **kw): raise AssertionError('Offline reproduction attempted network')
    monkeypatch.setattr('observe.data.sources.baostock.BaoStock.session', forbidden)
    before = tree_hash(Path(filtered['output'])); again = reproduce(tmp_path, filtered['run_id'])
    assert again['reproduction']['result'] == 'match' and again['reproduction']['differences'] == 0
    assert before == tree_hash(Path(filtered['output']))


def test_scope_compatibility_validation_and_target_escape(tmp_path):
    _, _, config = _fixture(tmp_path)
    cfg = StrategyConfig.model_validate({**config, 'implementation': 'ma10_ma20_v1'})
    assert 'execution_instruments' not in cfg.model_dump(mode = 'json')
    assert 'execution_instruments' not in RunConfig(snapshot = cfg.snapshot).model_dump(mode = 'json')
    for scope in ([], ['000333.SZ', '000333.SZ'], ['000333'], ['600001.SH']):
        with pytest.raises(ValueError): StrategyConfig.model_validate({**cfg.model_dump(), 'execution_instruments': scope})
    with pytest.raises(ValueError, match = '全市场'):
        StrategyConfig.model_validate({**config, 'implementation': 'bp_component_v1', 'execution_instruments': ['000333.SZ']})
    with pytest.raises(ValueError, match = '规则策略'):
        RunConfig(snapshot = cfg.snapshot, execution_instruments = ['000333.SZ'])
    result = run_strategy(tmp_path, **cfg.model_dump(), execution_instruments = ['000333.SZ']); output = Path(result['output'])
    bc = RunConfig(snapshot = cfg.snapshot, scores = {'source': 'rules', 'run': str(output), 'model': cfg.implementation},
                   portfolio = cfg.portfolio, execution_instruments = ['000333.SZ'])
    with pytest.raises(ValueError, match = '范围不匹配'):
        rule_source(tmp_path, bc.model_copy(update = {'execution_instruments': None}), cfg.snapshot)
    frame = pd.read_parquet(output / 'targets.parquet'); frame.loc[0, 'instrument'] = '600001.SH'
    frame.to_parquet(output / 'targets.parquet', index = False)
    manifest = json.loads((output / 'signals_manifest.json').read_text()); manifest['files']['targets.parquet'] = file_sha(output / 'targets.parquet')
    write_json(output / 'signals_manifest.json', manifest)
    with pytest.raises(ValueError, match = '目标超出'):
        rule_source(tmp_path, bc, cfg.snapshot)


def test_ma5_price_filter_buy_exit_and_neutral(tmp_path):
    sid, days, config = _fixture(tmp_path); store = Store(tmp_path)
    config.update(implementation = 'ma5_ma10_price_v1', instrument = '000333.SZ', parameters = {'short': 5, 'long': 10, 'buy_multiplier': 1})
    data = {t: store.load(t, sid) for t in ('calendar', 'instruments', 'bars_1d', 'adj_factors', 'adj_coverage')}
    bars = data['bars_1d']; one = bars.instrument.eq('000333.SZ'); bars.loc[one, 'close'] = 10
    for k, price in ((26, 12), (27, 11), (28, 10.5), (29, 13), (30, 11.3)):
        bars.loc[one & bars.date.eq(days[k]), 'close'] = price
    output = tmp_path / 'signals'; output.mkdir(); _generate_signals(StrategyConfig.model_validate(config), data, output)
    factors = pd.read_parquet(output / 'factors.parquet').set_index('date')
    assert factors.loc[days[25], 'state'] == 0  # Equality must not buy.
    assert factors.loc[days[26], 'state'] == 1
    exit_ = factors.loc[days[28]]
    assert exit_.close_adj > exit_.ma_long and exit_.close_adj < exit_.ma_short and exit_.state == 0
    assert factors.loc[days[29], 'state'] == 1
    assert factors.loc[days[31], 'state'] == 0
    expected = 0
    prices = bars[one].sort_values('date').set_index('date').close
    for day, row in factors.iterrows():
        k = prices.index.get_loc(day); a = sum(prices.iloc[k - 4:k + 1]) / 5; b = sum(prices.iloc[k - 9:k + 1]) / 10
        if a > b and prices.loc[day] > a: expected = 1
        if prices.loc[day] < a: expected = 0
        assert row.ma_short == pytest.approx(a) and row.ma_long == pytest.approx(b) and row.state == expected


def test_progress_archives_legacy_and_declarative_without_false_mapping(tmp_path):
    root = tmp_path; catalog = root / 'catalog/strategies/c1'; catalog.mkdir(parents = True)
    record = {'strategy_id': 'S-test', 'path': 'original.txt', 'bytes_sha256': 'sha', 'status': '待审查',
              'review_status': '静态扫描，尚未人工审查', 'fidelity': '未审查', 'gaps': [], 'next_step': 'review', 'duplicate_of': None, 'implementations': []}
    write_json(catalog / 'catalog.json', [record]); write_json(catalog.parent / 'latest.json', {'catalog_id': 'c1'})
    for name, source in [('legacy', {}), ('declarative', {'path': 'original.txt', 'bytes_sha256': 'sha'})]:
        output = root / 'runs' / name; output.mkdir(parents = True)
        write_json(output / 'config.json', {'kind': 'strategy', 'config': {'name': 'spec', 'start': '2024-01-01', 'end': '2024-02-01'}, 'source': source})
        write_json(output / 'status.json', {'status': 'success_limited'})
    first = archive(root, 'first'); before = tree_hash(Path(first['output']))
    second = archive(root, 'second')
    assert first['output'] != second['output'] and tree_hash(Path(first['output'])) == before
    assert second['locally_backtested_sources'] == 1 and second['manually_reviewed'] == 0
    assert second['locally_reproduced_sources'] == 0
    runs = json.loads((Path(second['output']) / 'runs.json').read_text())
    assert next(r for r in runs if r['run_id'] == 'legacy')['strategy_id'] is None


@pytest.mark.parametrize('response_days', [['2021-01-09', '2021-01-10'], ['2021-01-09']])
def test_calendar_supplement_preserves_old_rows_and_rejects_incomplete(tmp_path, response_days):
    store = Store(tmp_path); dates = pd.date_range('2021-01-04', '2026-09-29').date
    old = pd.DataFrame({'date': [d for d in dates if str(d) not in ('2021-01-09', '2021-01-10')], 'is_open': False})
    store.publish(store.write_batch({'calendar': {'all': store.write_partition('calendar', 'all', old)}}))
    sid = store.snapshot('before supplement'); before = tree_hash(tmp_path / 'std'); published = store.published_path.read_bytes()
    class Source:
        def session(self): return nullcontext(self)
        def calendar(self, start, end):
            assert str(start) == '2021-01-09' and str(end) == '2021-01-10'
            return pd.DataFrame({'calendar_date': response_days, 'is_trading_day': '0'})
    directory = tmp_path / 'stage'; directory.mkdir()
    if len(response_days) == 1:
        with pytest.raises(ValueError, match = 'Incomplete'): supplement_calendar(tmp_path, directory, Source())
        assert store.published_path.read_bytes() == published and tree_hash(tmp_path / 'std') == before
        assert json.loads((directory / 'calendar-supplement.json').read_text())['status'] == 'failed'
    else:
        result = supplement_calendar(tmp_path, directory, Source())
        assert result['old_rows_retained'] == len(old) and len(store.load('calendar')) == len(old) + 2
        pd.testing.assert_frame_equal(store.load('calendar', sid), old)
        assert all(file_sha(tmp_path / 'std' / name) == sha for name, sha in before.items())
    with pytest.raises(FileExistsError): supplement_calendar(tmp_path, directory, Source())


def test_protection_excludes_mutable_registry_from_older_baseline(tmp_path):
    store = Store(tmp_path); day = pd.Timestamp('2024-01-02').date()
    frames = {'calendar': pd.DataFrame({'date': [day], 'is_open': [True]}),
              'index_1d': pd.DataFrame({'date': [day], 'index': ['000300.SH'], 'close': [100.]})}
    store.publish(store.write_batch({t: {'all': store.write_partition(t, 'all', f)} for t, f in frames.items()}))
    sid = store.snapshot('protected'); registry = tmp_path / 'runs/registry.sqlite'; registry.parent.mkdir(); registry.write_bytes(b'old')
    directory = tmp_path / 'stage'; directory.mkdir()
    write_json(directory / 'baseline.json', {'published': store.published(), 'controls': {
        f'snapshots/{sid}.json': file_sha(tmp_path / 'snapshots' / f'{sid}.json'), 'runs/registry.sqlite': file_sha(registry)},
        'partitions': {}})
    registry.write_bytes(b'new')
    result = protect(tmp_path, directory)
    assert result['old_controls_unchanged'] == 1 and result['mutable_registry_files_excluded'] == 1
