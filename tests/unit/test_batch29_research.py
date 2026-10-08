from copy import deepcopy
from pathlib import Path

import numpy as np
import pytest
import talib

from observe.runs import file_sha, write_json
from scripts import review_strategy_batch29 as batch


@pytest.fixture
def sources(tmp_path):
    rows = []
    for name in batch.SOURCES:
        path = Path('repo/量化策略源代码') / name
        if not path.is_file(): pytest.skip('Restore ignored original strategy sources')
        rows.append({'source_copy': str(path), 'source_sha256': file_sha(path)})
    write_json(tmp_path / 'source-reviews/review.json', {'sources': rows}); return tmp_path


def test_exact_diagnostics_preserve_source_rules_and_defects(sources):
    rows = {r['case']: r for r in batch.diagnostics(sources)['cases']}
    assert rows['empty_sort_preserves_lexical_pool']['pool'] == ['a', 'b', 'c']
    assert set(rows['user_code_expands_characters']['pool']) == set('600519.XSHG')
    assert rows['false_hold_flag_filters_existing']['pool'] == ['a', 'c']
    assert rows['unresolved_wizard_call']['threshold'] == .2
    assert rows['unresolved_wizard_call']['callee_not_executed']
    assert rows['daily_reset']['daily_risk_management'] and rows['daily_reset']['sell_days'] == {'a': 1}
    assert rows['wall_clock_and_ignored_days']['at_2020_backtest_but_2026_clock'] == ['a']
    assert rows['smart_legacy_sort']['error'] == 'AttributeError'
    assert rows['empty_risk_pool']['error'] == 'ZeroDivisionError'
    assert rows['date_close_zero_index']['error'] == 'KeyError'
    assert rows['unprovided_isnan']['error'] == 'NameError'
    assert rows['negative_northbound_top_is_still_selected']['selection'] == ['a']
    assert rows['negative_northbound_top_is_still_selected']['net'] == -20.
    assert rows['first_calendar_negative_wrap']['result'] == '2020-01-03'
    assert rows['holding_number_times_price']['values'] == {'a': 50., 'b': 40.}
    assert rows['empty_holding_query']['empty']
    assert not rows['mismatched_columns_nonempty_nan']['empty']
    assert rows['mismatched_columns_nonempty_nan']['all_missing']
    duplicate = rows['duplicate_holding_codes']
    assert duplicate.get('duplicate_columns') or duplicate.get('error') == 'ValueError'
    assert rows['holding_rank_legacy_by']['error'] == 'TypeError'


def test_actual_source_atr_preserves_24_row_warmup_and_separate_clip(sources):
    holder, ns, _ = batch.atr_kernel(sources)
    x = np.arange(24., dtype=float) + 100.; x[-1] = 1000.
    holder['arrays'] = {'close': x.copy(), 'high': x + 2, 'low': x - 2}
    saved = {n: a.copy() for n, a in holder['arrays'].items()}
    actual = ns['fun_getATR']('a')
    clipped = {n: batch.reference_clip(a) for n, a in saved.items()}
    assert actual == pytest.approx(batch.reference_atr(clipped['high'], clipped['low'], clipped['close']), rel=1e-12)
    assert clipped['close'][-1] < 1000.
    for name in saved: np.testing.assert_array_equal(holder['arrays'][name], saved[name])


def test_source_per_field_clipping_can_invert_valid_ohlc(sources):
    _, ns, _ = batch.atr_kernel(sources)
    close = np.array([100.] * 23 + [50.]); high = close + 1.; high[0] = 200.; low = close - 1.
    assert ((high >= close) & (close >= low)).all()
    clipped = [ns['fun_normalizeData'](x.copy()) for x in (high, low, close)]
    assert clipped[2][-1] > clipped[0][-1]
    reference = [batch.reference_clip(x) for x in (high, low, close)]
    assert reference[2][-1] > reference[0][-1]
    for a, b in zip(clipped, reference, strict=True): np.testing.assert_allclose(a, b, rtol=1e-12)


@pytest.mark.parametrize('seed', [1, 5, 11])
def test_reference_atr_agrees_with_engine_on_nonconstant_histories(seed):
    rng = np.random.default_rng(seed); close = np.cumsum(rng.normal(size=24)) + 100.
    high = close + rng.uniform(.1, 2, 24); low = close - rng.uniform(.1, 2, 24)
    assert batch.reference_atr(high, low, close) == pytest.approx(talib.ATR(high, low, close, timeperiod=14)[-1], abs=1e-12)


def test_reference_atr_zero_range_and_initial_gap_seed():
    flat = np.full(24, 10.); assert batch.reference_atr(flat, flat, flat) == 0.
    close = np.array([10.] + [20.] * 23); high = close + 1.; low = close - 1.
    assert batch.reference_atr(high, low, close) == pytest.approx(talib.ATR(high, low, close, timeperiod=14)[-1])


@pytest.mark.parametrize('values', [[], [1.], [1., np.nan], [[1., 2.]]])
def test_bad_clipping_input_is_not_filled(values):
    with pytest.raises(ValueError, match='Invalid clip'): batch.reference_clip(values)


def test_bad_atr_input_is_not_filled():
    with pytest.raises(ValueError, match='Invalid ATR'): batch.reference_atr(np.ones(24), np.ones(23), np.ones(24))
    with pytest.raises(ValueError, match='Invalid ATR'): batch.reference_atr(np.ones(14), np.ones(14), np.ones(14))


def test_atomic_expression_checks_sha_and_ambiguity(sources):
    with pytest.raises(ValueError, match='Missing/ambiguous'): batch.atomic_expression(sources, 1, "df['cum_vol']")
    doc = batch.read(sources / 'source-reviews/review.json'); doc['sources'][2]['source_sha256'] = '0' * 64
    write_json(sources / 'source-reviews/review.json', doc)
    with pytest.raises(ValueError, match='Selected source changed'): batch.atomic_expression(sources, 2, "df['net'] =")


def test_byte_drift_refused_before_parsing(tmp_path):
    p = tmp_path / 'price-input.parquet'; p.write_bytes(b'not parquet')
    write_json(tmp_path / 'input-analysis.json', {'snapshot': batch.SNAPSHOT, 'not_a_backtest': True,
        'pool': list(batch.previous.prior.INSTRUMENTS), 'file': str(p), 'sha256': '0' * 64})
    with pytest.raises(ValueError, match='Component input changed'): batch.inputs(tmp_path)


@pytest.mark.parametrize('change', ['none', 'source', 'review', 'component', 'clip', 'after'])
def test_review_supplement_keeps_original_review_and_proof_bindings(tmp_path, change):
    after = deepcopy(batch.REVIEWS[batch.SOURCES[1]]); before = deepcopy(after); before['gaps'].remove(batch.CLIP_GAP)
    write_json(tmp_path / 'source-reviews/review.json', {'sources': [{}, {'source_sha256': 'original', 'review': before}]})
    for name in ('component-research.json', 'clip-order-diagnostics.json'): write_json(tmp_path / name, {'synthetic': True})
    doc = {'before': before, 'after': after, 'source_sha256': 'original',
        'review_sha256': file_sha(tmp_path / 'source-reviews/review.json'),
        'component_sha256': file_sha(tmp_path / 'component-research.json'), 'clip_sha256': file_sha(tmp_path / 'clip-order-diagnostics.json')}
    if change in ('source', 'review', 'component', 'clip'): doc[f'{change}_sha256'] = '0' * 64
    if change == 'after': doc['after']['gaps'] = []
    write_json(tmp_path / 'source-review-supplement.json', doc)
    if change == 'none': assert batch.validate_supplement(tmp_path)['after'] == batch.REVIEWS[batch.SOURCES[1]]
    else:
        with pytest.raises(ValueError, match='Review supplement differs'): batch.validate_supplement(tmp_path)


@pytest.mark.parametrize('change', ['query', 'sha', 'publication', 'failed_data'])
def test_probe_binding_refuses_substitution(tmp_path, monkeypatch, change):
    api = {'apis': []}; write_json(tmp_path / 'existing-apis.json', api); monkeypatch.setattr(batch, 'api_evidence', lambda: api)
    for endpoint, query in batch.QUERIES.items():
        row = {'endpoint': endpoint, 'query': query, 'api_sha256': file_sha(tmp_path / 'existing-apis.json'),
            'status': 'failed', 'published': False, 'files': [], 'wire': [], 'error': 'synthetic failure'}
        if change == 'query': row['query'] = {}
        if change == 'sha': row['api_sha256'] = '0' * 64
        if change == 'publication': row['published'] = True
        if change == 'failed_data': row['files'] = [{'file': 'unaccepted'}]
        write_json(tmp_path / f'probe-{endpoint}.json', row)
    with pytest.raises(ValueError): batch.validate_probes(tmp_path)
