from copy import deepcopy
from pathlib import Path

import pytest

from observe.runs import file_sha, write_json
import scripts.review_strategy_batch16 as batch


@pytest.mark.parametrize('fault', ['missing_baseline', 'changed_baseline', 'missing_code', 'changed_code'])
def test_final_checks_require_baseline_and_implementation_custody(tmp_path, monkeypatch, fault):
    monkeypatch.chdir(tmp_path); directory = tmp_path / 'stage'; directory.mkdir()
    code = tmp_path / 'implementation.py'; code.write_bytes(b'checked implementation')
    monkeypatch.setattr(batch, 'FILES', {'implementation.py'})
    for name in batch.CORE: write_json(directory / name, {})
    codes = {'implementation.py': file_sha(code)}
    evidence = {name: file_sha(directory / name) for name in batch.CORE}
    if fault == 'missing_baseline': evidence.pop('baseline.json')
    elif fault == 'changed_baseline': write_json(directory / 'baseline.json', {'controls': {}})
    elif fault == 'missing_code': codes.clear()
    else: code.write_bytes(b'changed implementation')
    write_json(directory / 'checks.json', {'commands': [{'returncode': 0}],
        'implementation_sha256': codes, 'evidence_sha256': evidence})
    with pytest.raises(ValueError, match='sets incomplete|evidence changed|implementation changed'):
        batch.verify_checks(directory)


def evidence_fixture(directory):
    rows = []
    for name in batch.SOURCES:
        source = Path('repo/量化策略源代码') / name; source.parent.mkdir(parents=True, exist_ok=True); source.write_bytes(b'frozen source')
        copied = directory / 'source-reviews' / (source.stem + '.source'); copied.parent.mkdir(parents=True, exist_ok=True); copied.write_bytes(source.read_bytes())
        review = deepcopy(batch.REVIEWS[name])
        if name in batch.RULE_CORRECTIONS: review['scope'] = 'initial description'
        rows.append({'source_path': str(source), 'source_copy': str(copied), 'source_sha256': file_sha(source),
            'source_copy_sha256': file_sha(copied), 'review': review})
    for name in batch.CORE: write_json(directory / name, {})
    api = directory / 'existing-apis.json'; api_sha = file_sha(api)
    reviewed = directory / 'source-reviews/review.json'
    write_json(reviewed, {'sources': rows, 'snapshot': batch.SNAPSHOT, 'api_sha256': api_sha})
    corrections = [{'path': name, 'source_sha256': row['source_sha256'], 'before': row['review'],
        'after': batch.REVIEWS[name], 'reason': batch.RULE_CORRECTIONS[name]}
        for name, row in zip(batch.SOURCES, rows, strict=True) if name in batch.RULE_CORRECTIONS]
    write_json(directory / 'rule-clarifications-final.json', {'original_review_sha256': file_sha(reviewed),
        'original_archive_retained': True, 'superseded_clarification_sha256': file_sha(directory / 'rule-clarifications.json'), 'corrections': corrections})
    raw = directory / 'raw.parquet'; raw.write_bytes(b'frozen raw data')
    records = [{'endpoint': name, 'evidence_files': [], 'result': {'status': 'failed'}} for name in batch.PROBES]
    records[0]['result'] = {'status': 'partial', 'raw_file': str(raw), 'raw_sha256': file_sha(raw)}
    terminal = directory / 'probes/money_flow/result.json'; write_json(terminal, records[0]['result'])
    probes = directory / 'probe-results.json'
    write_json(probes, {'snapshot': batch.SNAPSHOT, 'api_evidence_sha256': api_sha, 'results': records})
    write_json(directory / 'minute-fallback.json', {'snapshot': batch.SNAPSHOT, 'original_probe_sha256': file_sha(probes),
        'api_evidence_sha256': file_sha(directory / 'existing-api-sina.json'),
        'record': {'endpoint': 'minute_sina', 'evidence_files': [], 'result': {'status': 'failed'}}})
    write_json(directory / 'diagnostics.json', {'not_a_backtest': True,
        'source_sha256': {r['source_path']: r['source_sha256'] for r in rows},
        'offline_count_recheck': {'result': 'match', 'differences': 0, 'source_unchanged': True}})
    return raw, terminal


@pytest.mark.parametrize('fault', ['changed_rules', 'changed_raw', 'changed_terminal', 'changed_api'])
def test_rule_and_probe_bindings_reject_changed_evidence(tmp_path, monkeypatch, fault):
    monkeypatch.chdir(tmp_path); directory = tmp_path / 'stage'; directory.mkdir()
    raw, terminal = evidence_fixture(directory)
    batch.checked_evidence(directory)
    if fault == 'changed_rules':
        path = directory / 'rule-clarifications-final.json'; doc = batch.read(path)
        doc['corrections'][0]['after']['scope'] = 'unsupported changed rules'; write_json(path, doc)
    elif fault == 'changed_raw': raw.write_bytes(b'changed raw data')
    elif fault == 'changed_terminal': write_json(terminal, {'status': 'success'})
    else: write_json(directory / 'existing-apis.json', {'source': 'changed'})
    with pytest.raises(ValueError, match='clarification differs|raw table changed|terminal result differs|API evidence changed'):
        batch.checked_evidence(directory)
