"""冻结输入、完整复现与统一状态：公开入口 run_offline / reproduce / CLI / 队列使用同一套状态定义。"""
import json
import shutil
from pathlib import Path

import pandas as pd
import pytest
import yaml

from observe import replay
from observe.cli import main
from observe.data.store import Store
from observe.jobs import Jobs
from observe.replay import CORE, ReproduceRefused, reproduce, run_offline, run_params
from tests.integration.helpers import A, D, bar, flat, snapshot, tree_hash

WINDOW = {'liquidity_window': 2}


def run(root, sid, **kw): return run_offline(root, snapshot = sid, initial_cash = 100000, execution = WINDOW, **kw)


def read(path, name): return json.loads((Path(path) / name).read_text(encoding = 'utf-8'))


def write(path, name, value): (Path(path) / name).write_text(json.dumps(value, ensure_ascii = False, indent = 1), encoding = 'utf-8')


def cli(*argv):
    try: main([*argv]); return 0
    except SystemExit as e: return e.code


def queue(root, kind, params):
    q = Jobs(root); jid = q.submit(kind, params); q.run_next(); return q.get(jid)


def dividend_run(root):
    rows = flat(A, days = D[:4]) + [bar(d, A, 9.5) for d in D[4:]]
    from tests.integration.helpers import action
    return run(root, snapshot(root, rows, actions = [action(A, D[4], cash = 0.5, pay = D[6])]))


# 完整复现 ---------------------------------------------------------------------------------------------
def test_reproduce_compares_every_core_table(tmp_path):
    src = dividend_run(tmp_path); r = reproduce(tmp_path, src['output'])
    comparison = read(r['output'], 'comparison.json')
    assert r['status'] == 'success' and r['reproduction']['result'] == 'match' and comparison['differences'] == []
    assert set(CORE) | {'status'} == set(comparison['tables']) and comparison['tables']['fills']['rows_expected'] == 1
    assert read(r['output'], 'status.json')['status'] == 'success' and read(r['output'], 'manifest.json')['core_hash'] == read(src['output'], 'manifest.json')['core_hash']


def _bump_fee(rows): rows[0]['fee'] = round(rows[0]['fee'] + 0.01, 2)
def _shift_cash(rows): rows[3]['cash'] += 1.0; rows[3]['market_value'] -= 1.0          # 当日净值不变
def _rename_reason(rows): rows[0]['reason'] = 'enter_elsewhere'
def _drop_position_day(rows): rows.pop(1)


@pytest.mark.parametrize('name, tamper, field', [('fills', _bump_fee, 'fee'), ('equity', _shift_cash, 'cash'), ('orders', _rename_reason, 'reason'),
                                                  ('positions_daily', _drop_position_day, None)])
def test_tampered_source_is_detected_even_with_same_final_equity(tmp_path, name, tamper, field):
    src = dividend_run(tmp_path)['output']; rows = read(src, f'{name}.json'); tamper(rows); write(src, f'{name}.json', rows)
    assert read(src, 'equity.json')[-1]['equity'] == 99974.26
    before = tree_hash(src)
    r = reproduce(tmp_path, src); comparison = read(r['output'], 'comparison.json')
    assert r['status'] == 'mismatch' and r['reproduction']['result'] == 'mismatch' and f'{name}.json' in comparison['source_integrity']['modified_files']
    hit = [d for d in comparison['differences'] if d['table'] == name]
    assert hit and (hit[0]['field'] == field if field else hit[0]['kind'] == 'only_in_actual')   # 源实验少了一行，重跑多出一行
    if name == 'equity': assert hit[0]['key'] == {'date': str(D[5])}                       # 逐日净值从 D[2] 开始，第 4 行是 D[5]
    status = read(r['output'], 'status.json'); assert status['status'] == 'mismatch' and status['execution_status'] == 'success'
    assert tree_hash(src) == before
    assert cli('--root', str(tmp_path), 'reproduce', src) == 2


def test_workspace_rule_change_uses_frozen_rules(tmp_path):
    rules = tmp_path / 'rules.yaml'; shutil.copy('configs/rule_profiles/main_board.yaml', rules)
    src = dividend_run_with_rules(tmp_path, rules)
    y = yaml.safe_load(rules.read_text(encoding = 'utf-8')); y['fees'][-1]['commission_rate'] = 0.003; rules.write_text(yaml.safe_dump(y), encoding = 'utf-8')
    r = reproduce(tmp_path, src['output']); comparison = read(r['output'], 'comparison.json')
    assert r['status'] == 'success' and comparison['rules'] == {'used': 'frozen', 'workspace_differs': True}
    assert read(r['output'], 'fills.json')[0]['commission'] == 24.75                       # 仍是冻结规则的佣金


def dividend_run_with_rules(root, rules):
    return run_offline(root, snapshot = snapshot(root, flat(A)), initial_cash = 100000, execution = WINDOW, rules = str(rules))


def test_altered_frozen_rules_or_snapshot_data_are_refused(tmp_path):
    src = run(tmp_path, snapshot(tmp_path, flat(A)))['output']
    frozen = Path(src) / 'rules.yaml'; text = frozen.read_text(encoding = 'utf-8')
    frozen.write_text(text.replace('commission_rate: 0.00025', 'commission_rate: 0.001', 1), encoding = 'utf-8')
    with pytest.raises(ReproduceRefused, match = '规则指纹'): reproduce(tmp_path, src)
    frozen.write_text(text, encoding = 'utf-8')
    s = Store(tmp_path); sid = read(src, 'config.json')['snapshot_id']; snap = read(tmp_path / 'snapshots', f'{sid}.json')
    other = s.write_partition('bars_1d', '2024', pd.DataFrame(flat(A, px = 11.0)))
    snap['tables']['bars_1d']['2024'] = other; write(tmp_path / 'snapshots', f'{sid}.json', snap)
    with pytest.raises(ReproduceRefused, match = 'bars_1d/2024'): reproduce(tmp_path, src)
    assert not [p for p in (tmp_path / 'runs').iterdir() if 'repro' in p.name]            # 拒绝发生在创建输出目录之前


def test_code_drift_is_reported_not_ignored(tmp_path, monkeypatch):
    src = run(tmp_path, snapshot(tmp_path, flat(A)))['output']
    real = replay.environment(); monkeypatch.setattr(replay, 'environment', lambda: {**real, 'git_commit': 'another-commit'})
    r = reproduce(tmp_path, src)
    assert r['reproduction']['result'] == 'match' and r['reproduction']['code_drift'] == ['git_commit']
    assert read(r['output'], 'comparison.json')['code_drift']['git_commit']['current'] == 'another-commit'


def test_config_records_frozen_inputs(tmp_path):
    r = run(tmp_path, snapshot(tmp_path, flat(A))); doc = read(r['output'], 'config.json'); data = read(r['output'], 'data_manifest.json')
    assert doc['config']['execution']['liquidity_window'] == 2 and doc['config']['portfolio']['participation'] == 0.05 and doc['scores']['name'] == 'lexicographic_engineering_baseline_v1'
    assert doc['rules']['fingerprint'] and doc['environment']['lock_sha256'] and set(data['used']) == {'calendar', 'bars_1d', 'instruments', 'corp_actions'}
    assert all('file_sha256' in v for parts in data['used'].values() for v in parts.values())
    assert set(read(r['output'], 'scores.json')[0]) == {'decision_date', 'instrument', 'score'}


# 配置校验 ---------------------------------------------------------------------------------------------
def test_cli_arguments_override_yaml_and_yaml_overrides_defaults():
    p = run_params({'snapshot_id': 's1', 'cash': 5000, 'execution': {'liquidity_window': 5}}, initial_cash = 200000.0, start = None)
    assert p['config'].snapshot == 's1' and p['config'].initial_cash == 200000.0 and p['config'].execution.liquidity_window == 5 and p['config'].portfolio.n == 1
    assert run_params({'snapshot': 's1', 'cash': 5000})['config'].initial_cash == 5000


@pytest.mark.parametrize('bad, message', [({'snapshot': 's', 'cash': 1, 'initial_cash': 2}, '同时给出'), ({'snapshot': 's', 'unknown_key': 1}, 'unknown_key'),
                                          ({'snapshot': 's', 'initial_cash': -1}, 'initial_cash'), ({'snapshot': 's', 'initial_cash': float('nan')}, 'initial_cash'),
                                          ({'snapshot': 's', 'portfolio': {'n': 0}}, 'portfolio'), ({'snapshot': 's', 'boards': ['nasdaq']}, 'boards'),
                                          ({'snapshot': 's', 'start': '2024-02-01', 'end': '2024-01-01'}, '晚于')])
def test_invalid_config_is_rejected(bad, message):
    with pytest.raises(ValueError, match = message): run_params(bad)


# 统一状态：函数 / 文件 / CLI / 队列 -----------------------------------------------------------------------
def test_status_is_consistent_across_function_file_cli_and_queue(tmp_path):
    sid_ok = snapshot(tmp_path / 'ok', flat(A)); sid_missing = snapshot(tmp_path / 'gap', [x for x in flat(A) if x['date'] != D[4]])
    cfg = tmp_path / 'limited.yaml'; cfg.write_text(yaml.safe_dump({'snapshot': sid_ok, 'start': str(D[0]), 'execution': WINDOW}), encoding = 'utf-8')
    cases = [('blocked', tmp_path / 'gap', {'snapshot': sid_missing, 'execution': WINDOW}, 3), ('success_limited', tmp_path / 'ok', {'snapshot': sid_ok, 'start': str(D[0]), 'execution': WINDOW}, 0),
             ('success', tmp_path / 'ok', {'snapshot': sid_ok, 'execution': WINDOW}, 0)]
    for expected, root, params, code in cases:
        r = run_offline(root, **params)
        assert r['status'] == expected and read(r['output'], 'status.json')['status'] == expected and read(r['output'], 'manifest.json')['status'] == expected
        job = queue(root, 'run_experiment', params); assert job['status'] == expected and json.loads(job['result'])['status'] == expected
        yml = root / f'{expected}.yaml'; yml.write_text(yaml.safe_dump(params), encoding = 'utf-8')
        assert cli('--root', str(root), 'run', '--config', str(yml)) == code
    assert queue(tmp_path / 'ok', 'run_experiment', {'snapshot': 'missing'})['status'] == 'failed'
    assert queue(tmp_path / 'ok', 'run_experiment', {'snapshot': sid_ok, 'cash': 1, 'initial_cash': 2})['status'] == 'failed'
    with pytest.raises(FileNotFoundError): cli('--root', str(tmp_path / 'ok'), 'run', '--snapshot', 'missing')


def test_blocked_with_limitations_stays_blocked(tmp_path):
    rows = flat(A, days = D[:4]) + [bar(d, A, 5.0) for d in D[4:]]                         # 除权但快照没有公司行动
    r = run_offline(tmp_path, snapshot = snapshot(tmp_path, rows), execution = WINDOW, start = str(D[0]))
    assert r['status'] == 'blocked' and r['limitations'] and read(r['output'], 'status.json')['status'] == 'blocked'


def test_reproduce_mismatch_reaches_queue(tmp_path):
    src = run(tmp_path, snapshot(tmp_path, flat(A)))['output']; rows = read(src, 'fills.json'); _bump_fee(rows); write(src, 'fills.json', rows)
    job = queue(tmp_path, 'reproduce', {'run': src}); assert job['status'] == 'mismatch'
    assert queue(tmp_path, 'reproduce', {'run': src, 'output': src})['status'] == 'failed'
