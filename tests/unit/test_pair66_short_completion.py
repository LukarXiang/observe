"""Mutation guards for a frozen experiment's inputs and result registration."""
import json

import pytest

from scripts import complete_pair66_short_windows as batch


@pytest.mark.parametrize('missing', ['bars_1d', 'calendar', 'corp_actions'])
def test_payload_inventory_cannot_drop_whole_table(monkeypatch, missing):
    tables = {t: {str(n): {'file': f'std/{t}/{n}'}} for t in
        ('bars_1d', 'calendar', 'instruments', 'corp_actions', 'adj_factors', 'adj_coverage', 'index_1d') for n in [0]}
    class Store:
        def __init__(self, root): pass
        def state(self, snapshot): return {'tables': tables}
    monkeypatch.setattr(batch.frozen, 'Store', Store)
    def forbidden(*args): raise AssertionError('Inventory rejected before bytes or orders')
    monkeypatch.setattr(batch.frozen, 'bound', forbidden)
    payload = {'data/'+next(iter(v.values()))['file']: 'sha' for t, v in tables.items() if t != missing}
    with pytest.raises(ValueError, match='Exact payload inventory'): batch.validate_payload({'payload_sha256': payload})


def test_payload_bytes_rechecked_before_order(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    paths = []
    for n in range(28):
        path = batch.Path('data/std/bars_1d')/str(n)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b'original'); paths.append(path)
    payload = {str(p): batch.file_sha(p) for p in paths}
    class Store:
        def __init__(self, root): pass
        def state(self, snapshot):
            return {'tables': {t: ({str(n): {'file': str(p)[5:]} for n, p in enumerate(paths)} if t == 'bars_1d' else {})
                for t in ('bars_1d', 'calendar', 'instruments', 'corp_actions', 'adj_factors', 'adj_coverage', 'index_1d')}}
    # Explicit synthetic paths exercise the same exact-set and byte guards.
    monkeypatch.setattr(batch.frozen, 'Store', Store)
    batch.validate_payload({'payload_sha256': payload})
    paths[-1].write_bytes(b'mutated')
    with pytest.raises(ValueError, match='Bound file changed'): batch.validate_payload({'payload_sha256': payload})


@pytest.mark.parametrize('registered', [False, True])
def test_missing_or_changed_verification_registration_rejected(monkeypatch, tmp_path, registered):
    path = tmp_path/'verified.json'; path.write_bytes(b'accepted')
    result = {'run_id': 'run'}
    record = {'run_id': 'run', 'validation_file': str(path), 'validation_sha256': batch.file_sha(path)}
    monkeypatch.setattr(batch, 'read', lambda p: [record] if registered else [])
    path.write_bytes(b'changed')
    with pytest.raises(ValueError, match='Verification registration or SHA'):
        batch.registered_verification(path, result, {'status': 'ok', 'original_unchanged': True, 'run': result})


def test_duplicate_cost_scenarios_rejected(tmp_path, monkeypatch):
    result = {'run_id': 'synthetic', 'output': 'data/runs/synthetic', 'status': 'success_limited', 'subruns': {}}
    monkeypatch.setattr(batch.Path, 'is_dir', lambda p: True)
    monkeypatch.setattr(batch, 'verify_run', lambda *args: {'status': 'ok'})
    monkeypatch.setattr(batch.frozen, 'checked_config', lambda *args: None)
    cfg = {'kind': 'strategy', 'snapshot_id': batch.frozen.SNAPSHOT, 'source': {'bytes_sha256': batch.frozen.SOURCE_SHA}, 'config': {}}
    monkeypatch.setattr(batch, 'read', lambda p: cfg if p.name == 'config.json' else {} if p.name == 'subruns.json'
        else {'results': [{'scenario': 'base'}]*3})
    with pytest.raises(ValueError, match='Three-cost report inventory'): batch.actual_run(result, {})


@pytest.mark.parametrize('mutation', [None, 'role', 'scope', 'status', 'implementation_sha256'])
def test_review_body_must_match_current_returned_role(monkeypatch, tmp_path, mutation):
    hashes = {'guard.py': 'current'}; rows = []
    for role in ('security', 'abuse', 'architecture', 'composition', 'assumptions', 'cascade'):
        body = {'status': 'ok', 'role': role, 'scope': 'pair66_short_completion_v2',
            'implementation_sha256': hashes, 'findings': []}
        if role == 'security' and mutation:
            body[mutation] = {'guard.py': 'old'} if mutation == 'implementation_sha256' else 'different'
        path = tmp_path/(role+'.json'); path.write_text(json.dumps(body))
        rows.append({'role': role, 'status': 'ok', 'file': str(path), 'sha256': batch.file_sha(path)})
    summary = tmp_path/'reviews.json'
    summary.write_text(json.dumps({'status': 'ok', 'implementation_sha256': hashes, 'reviews': rows}))
    monkeypatch.setattr(batch, 'REVIEWS', summary)
    if mutation:
        with pytest.raises(ValueError, match='Review body identity'): batch.review_binding(hashes)
    else: batch.review_binding(hashes)


def test_changed_summary_cannot_replace_actual_report(monkeypatch, tmp_path):
    row = {'label': 'pair66-short-1'}; result = {'run_id': 'run'}
    monkeypatch.setattr(batch, 'DIRECTORY', tmp_path)
    monkeypatch.setattr(batch, 'actual_run', lambda *args: (tmp_path, {'return': 0.01}, 3))
    monkeypatch.setattr(batch, 'registered_verification', lambda *args: None)
    monkeypatch.setattr(batch, 'read', lambda p: result if p.name.endswith('-run.json') else
        {'report': {'return': 9.99}, 'source_sha256': batch.frozen.SOURCE_SHA})
    with pytest.raises(ValueError, match='Verification summary differs'): batch.checked_verification(row)


@pytest.mark.parametrize('field', [None, 'worker_pid', 'worker_parent_pid', 'reproduction_created', 'run_id', 'validation_sha256'])
def test_fresh_worker_claim_must_match_captured_stdout(monkeypatch, tmp_path, field):
    recovery = tmp_path/'recovery'; recovery.mkdir()
    directory = tmp_path/'experiment'; directory.mkdir()
    monkeypatch.setattr(batch, 'RECOVERY', recovery); monkeypatch.setattr(batch, 'DIRECTORY', directory)
    label = 'pair66-short-1'; hashes = {'guard': 'sha'}
    (recovery/'implementation.json').write_text(json.dumps(hashes))
    verification = directory/(label+'-verification.json')
    verification.write_text(json.dumps({'run': {'run_id': 'run'}, 'reproduction': {'run_id': 'again'}}))
    captured = {'run_id': 'run', 'reproduction_run_id': 'again', 'worker_pid': 202,
        'worker_parent_pid': 101, 'reproduction_created': True, 'validation_sha256': batch.file_sha(verification)}
    claim = {**captured, 'command': batch.fresh_command(label), 'returncode': 0,
        'manager_pid': 101, 'implementation_sha256': hashes}
    if field: captured[field] = False if field == 'reproduction_created' else 'contradiction'
    stdout = recovery/'stdout'; stderr = recovery/'stderr'
    stdout.write_text(json.dumps(captured)+'\n'); stderr.write_text('')
    for key, path in (('stdout', stdout), ('stderr', stderr)):
        claim[key+'_file'] = str(path); claim[key+'_sha256'] = batch.file_sha(path)
    (recovery/(label+'-fresh-worker.json')).write_text(json.dumps(claim))
    if field:
        with pytest.raises(ValueError, match='Fresh-worker stdout identity'): batch.fresh_evidence(label)
    else: batch.fresh_evidence(label)
