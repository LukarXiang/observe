import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

from observe.cli import main
from observe.data.store import Store
from observe.data.valuations import import_valuations
from observe.experiments import ExperimentConfig
from observe.factors.expr import compute, parse
from observe.replay import reproduce
from observe.research import ResearchConfig, run_research
from observe.strategy.run import run_strategy
from tests.integration.helpers import tree_hash
from tests.integration.test_research import cfg, make
from tests.integration.test_strategies import _fixture
from tests.integration.test_strategy_merge import _spec
from tests.unit.test_valuations import bind_raw


def recover(store, bars):
    frames = []
    for day, group in bars.groupby('date', sort=True):
        group = group.reset_index(drop=True)
        raw = pd.DataFrame({'date': str(day), 'code': group.instrument.map(lambda x: f'{x[-2:].lower()}.{x[:6]}'),
            'tradestatus': group.is_trading.astype(int).astype(str),
            'psTTM': (10 + (group.close / group.preclose - 1) * 100).astype(str),
            'pcfNcfTTM': (50 + np.arange(len(group))).astype(str)})
        frames.append((day, raw))
    manifest, sha = bind_raw(store, frames); result = import_valuations(store.root, manifest, sha)
    return store.snapshot(), result


def test_optional_policy_preserves_old_config_serialization():
    research = ResearchConfig(snapshot='frozen')
    assert 'valuation_policy' not in research.model_dump(mode='json')
    assert 'valuation_policy' not in ResearchConfig(**research.model_dump()).model_dump()
    experiment = ExperimentConfig(snapshot='frozen')
    assert 'valuation_policy' not in experiment.model_dump(mode='json')
    with pytest.raises(ValueError): ResearchConfig(snapshot='frozen', valuation_policy='strict_override')


def test_ps_research_requires_opt_in_and_reproduces_offline(tmp_path, monkeypatch):
    old, _, factor_file = make(tmp_path); store = Store(tmp_path)
    new, _ = recover(store, store.load('bars_1d', old))
    factors = Path(factor_file); factors.write_text(factors.read_text().replace('close_adj / ts_delay(close_adj, 3) - 1', 'cs_zscore(ps_ttm)'), encoding='utf-8')
    params = cfg(new, factor_file)
    blocked = run_research(tmp_path, **params)
    assert blocked['status'] == 'blocked' and blocked['blocked'][0]['kind'] == 'valuation_policy'
    def forbidden(*a, **kw): raise AssertionError('Offline valuation research attempted network')
    monkeypatch.setattr('observe.data.sources.baostock.BaoStock.session', forbidden)
    result = run_research(tmp_path, **params, valuation_policy='provider_final')
    assert result['status'] == 'success_limited'
    out = Path(result['output']); doc = json.loads((out / 'status.json').read_text())
    assert doc['evidence'] == 'exploratory'
    used = json.loads((out / 'data_manifest.json').read_text())['used']
    assert {'valuations_1d', 'valuation_coverage'} <= used.keys()
    assert all(v['file_sha256'] for t in ('valuations_1d', 'valuation_coverage') for v in used[t].values())
    fac = pd.read_parquet(out / 'factors.parquet'); values = store.load('valuations_1d', new)
    day = fac.date.min(); x = values[values.date.eq(day)].set_index('instrument').ps_ttm
    expected = (x - x.mean()) / x.std(ddof=1)
    actual = fac[fac.date.eq(day)].set_index('instrument').rev_3
    assert np.allclose(actual.sort_index(), expected.sort_index())
    assert pd.read_parquet(out / 'predictions.parquet').evidence_level.eq('exploratory').all()
    before = tree_hash(out); again = reproduce(tmp_path, out)
    assert again['reproduction']['result'] == 'match' and tree_hash(out) == before
    old_block = run_research(tmp_path, **{**params, 'snapshot': old}, valuation_policy='provider_final')
    assert old_block['status'] == 'blocked' and old_block['blocked'][0]['kind'] == 'valuation_missing'


def test_declarative_valuation_uses_shared_ledger_and_freezes(tmp_path, monkeypatch):
    old, days, _ = _fixture(tmp_path); store = Store(tmp_path); sid, result = recover(store, store.load('bars_1d', old))
    spec = _spec(tmp_path); raw = yaml.safe_load(spec.read_text()); raw['select']['pipelines'][0] = [{'sort': 'ps_ttm', 'take': 1}]
    spec.write_text(yaml.safe_dump(raw), encoding='utf-8')
    blocked = run_strategy(tmp_path, spec, snapshot=sid, start=days[25], end=days[-1])
    assert blocked['status'] == 'blocked' and blocked['blocked'][0]['kind'] == 'valuation_policy'
    raw['valuation_policy'] = 'provider_final'; spec.write_text(yaml.safe_dump(raw), encoding='utf-8')
    def forbidden(*a, **kw): raise AssertionError('Network attempted')
    monkeypatch.setattr('observe.data.sources.baostock.BaoStock.session', forbidden)
    run = run_strategy(tmp_path, spec, snapshot=sid, start=days[25], end=days[-1]); out = Path(run['output'])
    assert run['status'] == 'success_limited' and json.loads((out / 'fills.json').read_text())
    assert 'valuations_1d' in json.loads((out / 'data_manifest.json').read_text())['used']
    before = tree_hash(out); again = reproduce(tmp_path, out)
    assert again['reproduction']['result'] == 'match' and tree_hash(out) == before
    assert result['strict_usable'] is False


def test_cli_queries_strict_and_explicit_final(tmp_path, capsys):
    old, days, _ = _fixture(tmp_path); store = Store(tmp_path); sid, _ = recover(store, store.load('bars_1d', old))
    args = ['--root', str(tmp_path), 'data', 'valuation-history', '--snapshot', sid, '--start', str(days[0]), '--end', str(days[-1]), '--instruments', '600001.SH']
    main(args); strict = json.loads(capsys.readouterr().out)
    assert strict['data'] == [] and strict['coverage']['strict_excluded'] == len(days)
    main(args + ['--mode', 'provider_final']); final = json.loads(capsys.readouterr().out)
    assert len(final['data']) == len(days) and final['coverage']['strict_usable'] is False


def test_existing_sample_zscore_handles_missing_and_zero_spread():
    x = pd.DataFrame([[1., 2., 3.], [2., 2., 2.], [1., np.nan, 3.]])
    out = compute('cs_zscore(ps_ttm)', {'ps_ttm': x})
    assert out.iloc[0].tolist() == [-1, 0, 1] and out.iloc[1].isna().all()
    assert out.iloc[2, 0] == pytest.approx(-1 / np.sqrt(2)) and np.isnan(out.iloc[2, 1])
    assert parse('cs_zscore(pcf_ncf_ttm)').fields == {'pcf_ncf_ttm'}


def test_valuation_range_uses_expression_warmup_and_blocks_earlier_request(tmp_path):
    old, days, factor_file = make(tmp_path); store = Store(tmp_path)
    bars = store.load('bars_1d', old); new, _ = recover(store, bars[bars.date.ge(days[40])])
    factors = Path(factor_file); factors.write_text(factors.read_text().replace('close_adj / ts_delay(close_adj, 3) - 1', 'cs_zscore(ps_ttm)'), encoding='utf-8')
    params = cfg(new, factor_file); params['universe']['min_listed_sessions'] = 30
    result = run_research(tmp_path, **params, start=days[40], valuation_policy='provider_final')
    assert result['status'] == 'success_limited'
    earlier = run_research(tmp_path, **params, start=days[35], valuation_policy='provider_final')
    assert earlier['status'] == 'blocked' and earlier['blocked'][0]['kind'] == 'valuation_coverage'


def test_unused_sidecar_does_not_enter_legacy_research_manifest(tmp_path):
    old, _, factor_file = make(tmp_path); store = Store(tmp_path); new, _ = recover(store, store.load('bars_1d', old))
    result = run_research(tmp_path, **cfg(new, factor_file))
    assert result['status'] == 'success_limited'
    assert all(row['kind'] != 'valuation_provider_final' for row in result['limitations'])
    used = json.loads((Path(result['output']) / 'data_manifest.json').read_text())['used']
    assert 'valuations_1d' not in used and 'valuation_coverage' not in used
    assert reproduce(tmp_path, result['output'])['reproduction']['result'] == 'match'


def test_explicit_ps_end_outside_calendar_or_price_tail_is_blocked(tmp_path):
    old, days, factor_file = make(tmp_path); store = Store(tmp_path)
    bars = store.load('bars_1d', old); calendar = store.load('calendar', old)
    extra_day = (pd.Timestamp(days[-1]) + pd.offsets.BDay()).date()
    calendar.loc[len(calendar)] = [extra_day, True]
    store.publish(store.write_batch({'calendar': {'all': store.write_partition('calendar', 'all', calendar)}}))
    extra = bars[bars.date.eq(days[-1])].copy(); extra['date'] = extra_day
    sid, _ = recover(store, pd.concat([bars, extra], ignore_index=True))
    fs = Path(factor_file); fs.write_text(fs.read_text().replace('close_adj / ts_delay(close_adj, 3) - 1', 'cs_zscore(ps_ttm)'), encoding='utf-8')
    params = cfg(sid, factor_file)
    outside = run_research(tmp_path, **params, end=extra_day + pd.Timedelta(days=10).to_pytimedelta(), valuation_policy='provider_final')
    assert outside['status'] == 'blocked' and outside['blocked'][0]['kind'] == 'valuation_calendar_range'
    beyond_prices = run_research(tmp_path, **params, end=extra_day, valuation_policy='provider_final')
    assert beyond_prices['status'] == 'blocked' and beyond_prices['blocked'][0]['kind'] == 'valuation_price_range'


def test_explicit_known_weekend_ps_end_is_allowed(tmp_path):
    old, days, factor_file = make(tmp_path); store = Store(tmp_path)
    bars = store.load('bars_1d', old); calendar = store.load('calendar', old)
    weekend = (pd.Timestamp(days[-1]) + pd.Timedelta(days=1)).date()
    assert weekend.weekday() == 5
    calendar.loc[len(calendar)] = [weekend, False]
    store.publish(store.write_batch({'calendar': {'all': store.write_partition('calendar', 'all', calendar)}}))
    sid, _ = recover(store, bars)
    fs = Path(factor_file); fs.write_text(fs.read_text().replace('close_adj / ts_delay(close_adj, 3) - 1', 'cs_zscore(ps_ttm)'), encoding='utf-8')
    result = run_research(tmp_path, **cfg(sid, factor_file), end=weekend, valuation_policy='provider_final')
    assert result['status'] == 'success_limited'
