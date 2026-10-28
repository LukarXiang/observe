import ast
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from observe.runs import file_sha
from observe.data.store import fingerprint
from observe.strategy_catalog import REVIEWS, read_source
from scripts import review_strategy_batch67 as batch


def originals(monkeypatch):
    paths = [Path('repo/量化策略源代码') / name for name in batch.SOURCES]
    if not all(p.is_file() for p in paths): pytest.skip('Read-only source checkout is not installed')
    assert [file_sha(p) for p in paths] == list(batch.SOURCE_SHA)
    trees = [ast.parse(read_source(p)[0]) for p in paths]
    monkeypatch.setattr(batch, 'source_tree', lambda d, i: trees[i])
    return trees


def prices(dates):
    values = np.linspace(5, 8, len(dates))
    return pd.DataFrame({'date': dates.strftime('%Y-%m-%d'), 'instrument': 'A', 'close': values,
        'close_adj': values*2, 'back_factor': 2., 'is_trading': True})


def test_duplicate_matches_canonical_without_new_strategy_claim(monkeypatch):
    trees = originals(monkeypatch)
    assert file_sha(Path('repo/量化策略源代码') / batch.CANONICAL) == batch.SOURCE_SHA[0]
    assert len(batch.definitions(trees[0])) == 11 and len(batch.definitions(trees[1])) == 7
    assert REVIEWS[batch.SOURCES[0]]['implementations'] == REVIEWS[batch.SOURCES[1]]['implementations'] == []


def test_original_lowest_momentum_and_minute_state_diagnostics(monkeypatch):
    originals(monkeypatch)
    result = batch.diagnostics(Path('unused')); cases = {r['case']: r for r in result['cases']}
    assert result['not_a_backtest'] and result['real_historical_trade_windows'] == 0
    assert cases['lowest_momentum_not_title_and_insertion_ties']['selected'] == ['B', 'C']
    assert cases['empty_new_targets_divide_by_zero_after_sell']['error'] == 'ZeroDivisionError'
    assert cases['datetime_series_minus1_is_label']['error'] == 'KeyError'
    assert cases['weekly_mean_skips_four_missing_closes']['finite_rows'] == 1
    assert cases['held_buy_disables_sell_but_buy_remains_enabled']['buy_flag'] == 1
    assert cases['no_closeable_comparison_does_not_assign_sell_flag']['sell_flag'] == 1


def test_diagnostic_result_matches_json_storage_roundtrip(monkeypatch):
    originals(monkeypatch)
    result = batch.diagnostics(Path('unused'))
    assert result == json.loads(json.dumps(result))


def test_91_observations_span90_intervals_and_mean_anchor(monkeypatch):
    tree = originals(monkeypatch)[0]; dates = pd.bdate_range('2020-01-01', periods=92)
    values, audit = batch.numeric_values(prices(dates), dates, tree)
    assert audit['statuses']['momentum91'] == {'insufficient_91_rows': 90, 'eligible': 2}
    assert values.momentum91.iloc[90] == pytest.approx(prices(dates).close.iloc[90]/5-1)
    assert values.mean5.iloc[-1] == pytest.approx(prices(dates).close.tail(5).mean())
    assert values.momentum91.iloc[:90].isna().all()


def test_missing_internal_day_blocks_until_it_leaves_window(monkeypatch):
    tree = originals(monkeypatch)[0]; dates = pd.bdate_range('2020-01-01', periods=95)
    values, audit = batch.numeric_values(prices(dates).drop(index=[2]), dates, tree)
    assert audit['statuses']['momentum91']['unknown_internal_calendar_day'] == 2
    assert audit['statuses']['momentum91']['eligible'] == 2
    assert values.loc[values.momentum91_status.eq('unknown_internal_calendar_day'), 'momentum91'].isna().all()
    assert values.loc[values.mean5_status.eq('unknown_calendar_day'), 'mean5'].isna().all()


def test_only_explicit_pause_skipped_for91_and_not_filled_for5(monkeypatch):
    tree = originals(monkeypatch)[0]; dates = pd.bdate_range('2020-01-01', periods=94)
    frame = prices(dates); frame.loc[2, 'is_trading'] = False
    values, audit = batch.numeric_values(frame, dates, tree)
    assert audit['known_paused_skipped'] == 1
    assert audit['statuses']['momentum91']['eligible'] == 3
    assert audit['statuses']['mean5']['known_pause_without_platform_fill_proof'] == 3
    assert values.momentum91.iloc[-1] == pytest.approx(frame.close.iloc[-1]/frame.close.iloc[3]-1)


def test_invalid_inputs_and_future_rows_do_not_change_earlier_component(monkeypatch):
    tree = originals(monkeypatch)[0]; dates = pd.bdate_range('2020-01-01', periods=94)
    frame = prices(dates); frame.loc[1, 'close_adj'] = np.nan
    values, audit = batch.numeric_values(frame, dates, tree)
    assert audit['statuses']['momentum91']['invalid_price'] == 2
    assert values.loc[values.momentum91_status.eq('invalid_price'), 'momentum91'].isna().all()
    original, _ = batch.numeric_values(prices(dates), dates, tree)
    changed = prices(dates); changed.loc[93, ['close', 'close_adj']] = 999
    again, _ = batch.numeric_values(changed, dates, tree)
    pd.testing.assert_frame_equal(original.iloc[:93], again.iloc[:93], check_exact=True)


def test_selected_rejects_definition_time_side_effects(monkeypatch):
    for text in ('@unsafe\ndef f(): pass', 'def f(x=open("secret")): pass', 'def f(*, x=open("secret")): pass'):
        monkeypatch.setattr(batch, 'source_tree', lambda *a: ast.parse(text))
        with pytest.raises(ValueError, match='Unsafe'): batch.selected(Path('unused'), 0, ['f'], {})


def test_duplicate_and_unknown_dates_are_rejected(monkeypatch):
    tree = originals(monkeypatch)[0]; dates = pd.bdate_range('2020-01-01', periods=5)
    frame = prices(dates)
    with pytest.raises(ValueError, match='Duplicate'):
        batch.numeric_values(pd.concat([frame, frame.head(1)]), dates, tree)
    with pytest.raises(ValueError, match='Unknown'):
        batch.numeric_values(frame, dates[:-1], tree)


def test_official_delisted_dates_do_not_silently_replace_master():
    official = pd.DataFrame({'证券代码': ['000001', '000002'], '上市日期': ['1990-01-01', '1991-01-01'],
        '终止上市日期': ['2024-01-01', '2026-10-01']})
    master = pd.DataFrame({'instrument': ['000001.SZ'], 'list_date': ['1990-01-01'], 'delist_date': [None]})
    original = master.copy(deep=True); result = batch.delisted_audit(official, master)
    assert result['matched_master_codes'] == 1 and result['missing_master_codes'] == ['000002.SZ']
    assert result['master_date_differences_or_missing'][0]['master_delist_date'] is None
    assert result['official_delist_after_numeric_end'] == 1 and not result['strict_usable']
    pd.testing.assert_frame_equal(master, original, check_exact=True)
    with pytest.raises(ValueError, match='duplicate'):
        batch.delisted_audit(pd.concat([official, official.head(1)]), master)


def test_snapshot_content_fingerprint_is_distinct_from_parquet_byte_sha(tmp_path, monkeypatch):
    master = pd.DataFrame({'instrument': ['000001.SZ'], 'list_date': ['1990-01-01'], 'delist_date': [None]})
    universe = pd.DataFrame({'year': [2022], 'instrument': ['000001.SZ']})
    tables = {}; frames = {'instruments': master, 'minute_universe': universe}
    for name, frame in frames.items():
        path = tmp_path / f'{name}.parquet'; frame.to_parquet(path, index=False)
        content = fingerprint(pd.read_parquet(path))
        assert content != file_sha(path)
        tables[name] = {'all': {'file': path.name, 'sha': content, 'rows': len(frame)}}
    monkeypatch.setattr(batch, 'Store', lambda root: SimpleNamespace(state=lambda s: {'tables': tables},
        load_state=lambda s, n: frames[n]))
    original_read = batch.read
    monkeypatch.setattr(batch, 'read', lambda p: {'probes': [{'case': 'sz-delisted', 'status': 'failed'}]}
        if Path(p).name == 'probes.json' else original_read(p))
    result = batch.dependency_audit(tmp_path, tmp_path, freeze=True)
    for name in frames:
        row = result['inputs'][name][0]
        assert row['sha256'] == file_sha(tmp_path / f'{name}.parquet')
        assert row['content_fingerprint'] == tables[name]['all']['sha']
    path = tmp_path / 'instruments.parquet'
    master.to_parquet(path, index=False, compression='gzip')
    with pytest.raises(ValueError, match='Dependency binding changed'):
        batch.dependency_audit(tmp_path, tmp_path)
    changed = master.assign(list_date='1999-01-01'); changed.to_parquet(path, index=False)
    with pytest.raises(ValueError, match='content fingerprint differs'):
        batch.dependency_audit(tmp_path, tmp_path)


def test_recovery_chain_guards_original_bytes_and_rejects_cycles(tmp_path):
    old = tmp_path / 'old'; old.mkdir(); original = old / 'source.txt'; original.write_text('original')
    new = tmp_path / 'new'; new.mkdir()
    doc = {'original_directory': str(old), 'original_left_read_only': True,
        'original_files_sha256': {'source.txt': file_sha(original)}, 'failure_files': {}}
    binding = new / 'recovery-binding.json'; binding.write_text(json.dumps(doc))
    batch.validate_recovery(new)
    original.write_text('changed')
    with pytest.raises(ValueError, match='Original failed attempt changed'): batch.validate_recovery(new)
    doc['original_directory'] = str(new); binding.write_text(json.dumps(doc))
    with pytest.raises(ValueError, match='Recovery scope changed'): batch.validate_recovery(new)
