"""审计范围：API 与运行前检查返回 scope / input_range；增量通过不显示成全快照通过；规则加载失败不绕过指纹校验。"""
import pandas as pd
from fastapi.testclient import TestClient

from observe.api.app import create_app
from observe.cli import run_kind
from observe.data.store import Store
from observe.data.update import default_rules
from observe.replay import run_offline
from tests.integration.helpers import A, flat, snapshot


def test_incremental_pass_is_not_a_full_snapshot_pass(tmp_path):
    sid = snapshot(tmp_path, flat(A), audit = False); s = Store(tmp_path); batch = s.published()['batch_id']
    s.commit_audit(batch, pd.DataFrame(columns = ['level', 'rule']), default_rules().config_fingerprint(),
                   {'start': '2024-01-11', 'end': '2024-01-11', 'days': 1, 'rows': 1}, scope = 'incremental')
    body = TestClient(create_app(tmp_path)).get('/api/data/issues').json()
    assert body['status'] == 'passed_incremental' and body['scope'] == 'incremental' and body['input_range']['start'] == '2024-01-11'
    r = run_offline(tmp_path, snapshot = sid, execution = {'liquidity_window': 2})
    assert r['status'] == 'success_limited' and r['limitations'][0]['kind'] == 'data_audit' and r['limitations'][0]['audit']['scope'] == 'incremental'
    run_kind(tmp_path, 'data_audit', {})
    body = TestClient(create_app(tmp_path)).get('/api/data/issues').json()
    assert body['status'] == 'passed' and body['scope'] == 'snapshot' and body['input_range']['days'] == 8


def test_rule_loading_failure_does_not_bypass_fingerprint(tmp_path, monkeypatch):
    snapshot(tmp_path, flat(A))
    from observe.data import update
    def broken(): raise RuntimeError('rules file unreadable')
    monkeypatch.setattr(update, 'default_rules', broken)
    assert TestClient(create_app(tmp_path)).get('/api/data/issues').json()['status'] == 'rules_unavailable'
