"""Reject changes that would invalidate the approved short-island evidence."""
import copy

import pytest
import yaml

from scripts import backtest_pair66_short_windows as batch


@pytest.mark.parametrize('field,value', [('initial_cash', 2000000), ('benchmark', '000016.SH'),
    ('cost_scenarios', ['base']), ('execution', {'slippage': 0.002, 'liquidity_window': 20, 'fee_multiplier': 1.0}),
    ('start', '2011-03-17')])
def test_changed_economics_or_dates_rejected_before_execution(monkeypatch, field, value):
    original = yaml.safe_load(batch.CONFIG.read_text(encoding='utf-8'))
    monkeypatch.setattr(batch, 'original', lambda: original)
    row = {'start': '2011-03-16', 'end': '2011-04-08'}
    value_cfg = {**original, 'snapshot': batch.SNAPSHOT, **row, 'cache': False}
    batch.checked_config(value_cfg, row)
    bad = copy.deepcopy(value_cfg); bad[field] = value
    with pytest.raises(ValueError): batch.checked_config(bad, row)


def test_payload_mutation_is_not_absorbed_as_new_baseline(tmp_path):
    path = tmp_path/'input'; path.write_bytes(b'accepted')
    sha = batch.file_sha(path); batch.bound(path, sha)
    path.write_bytes(b'new-data')
    with pytest.raises(ValueError, match='Bound file changed'): batch.bound(path, sha)


def test_unknown_label_cannot_start_orders(monkeypatch):
    monkeypatch.setattr(batch, 'validate', lambda: {'selected': [{'label': 'pair66-short-1'}]})
    def forbidden(*args, **kwargs): raise AssertionError('No orders for unknown label')
    monkeypatch.setattr(batch, 'run_one', forbidden)
    with pytest.raises(ValueError, match='Unknown frozen label'): batch.run('other')
