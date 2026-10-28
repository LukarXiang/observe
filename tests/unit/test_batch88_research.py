"""Original source boundaries and per-date evidence, not strategy PnL tests."""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from scripts import review_strategy_batch88 as batch


@pytest.fixture
def frozen(tmp_path):
    rows = []
    for name, sha in zip(batch.SOURCES, batch.SOURCE_SHA, strict=True):
        source = Path('repo/量化策略源代码')/name
        if not source.exists():
            pytest.skip('Original payload sources require handoff restoration')
        copied = tmp_path/(str(len(rows))+'.source')
        copied.write_bytes(source.read_bytes())
        rows.append({'source_copy': str(copied), 'source_path': str(source), 'source_sha256': sha})
    (tmp_path/'sources-frozen.json').write_text(json.dumps(rows), encoding='utf-8')
    return tmp_path


def test_loaders_leave_platform_and_chart_top_level_inert(frozen):
    chan = batch.namespace(frozen, 0)
    notebook = batch.namespace(frozen, 1)
    assert 'KBarChan' in chan and 'draw_chan' not in chan and 'stock_df' not in chan
    assert 'get_result' in notebook and 'today' not in notebook and 'get_price' not in notebook


def test_sha_change_blocks_execution(frozen):
    (frozen/'0.source').write_bytes(b'print("unreviewed")')
    with pytest.raises(ValueError, match='Unreviewed source'):
        batch.namespace(frozen, 0)


def test_payload_ignores_structured_padding_and_keeps_float_bits():
    source = np.zeros(2, dtype=[('high', 'f8'), ('unused', 'i8'), ('low', 'f8')])
    source['high'] = [1., np.nextafter(1., 2.)]
    first = source[['high', 'low']]
    second = source.copy(); second['unused'] = [123, 456]
    assert batch.payload(first) == batch.payload(second[['high', 'low']])
    assert batch.payload(first)['rows'][0] != batch.payload(first)['rows'][1]


def test_dynamic_window_anchor_uses_last_factor():
    bars = pd.DataFrame({'date': ['2020-01-01', '2020-01-02'], 'back_factor': [1., 2.],
        **{name+'_adj': [10., 12.] for name in ('open', 'high', 'low', 'close')}})
    array = batch.window_array(bars)
    np.testing.assert_array_equal(array['close'], [5., 6.])
    assert array.dtype['date'] == np.dtype('datetime64[ns]')


def test_original_notebook_edges_preserved(frozen):
    result = batch.diagnostics(frozen)
    assert result['count'] == 15
    cases = {r['case']: r for r in result['cases']}
    assert cases['momentum-horizon-20']['leader_count'] == 7
    assert cases['nested-gap-union-shrinks']['output'] == [[1., 3.]]
    assert cases['unsupported-horizon']['error'].startswith('NameError:')
    assert cases['insufficient-20day-lookback']['error'].startswith('IndexError:')
    assert cases['round-before-stable-sort']['values'] == [.11, .11, 0.]
    assert cases['missing-industry-name-string-list-delete']['error'].startswith('TypeError:')


def test_original_chan_terminal_assumption_and_retry(frozen):
    ns = batch.namespace(frozen, 0)
    values = np.array([10., 11., 12., 13., 12., 11., 10., 9., 10., 11., 12., 13.])
    bars = pd.DataFrame({'date': pd.bdate_range('2020-01-01', periods=len(values)), 'back_factor': 1.,
        'open_adj': values, 'high_adj': values+0.5, 'low_adj': values-0.5, 'close_adj': values})
    a = batch.window_array(bars)
    state = batch.chan_state(ns, a)
    assert state == batch.chan_state(ns, a)
    assert state['kDataFrame_origin']['rows'][0][5] == 1
    obj = ns['KBarChan'](a.copy())
    obj.standardize()
    with pytest.raises(ValueError):
        obj.standardize()
