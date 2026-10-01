import json
from concurrent.futures import ThreadPoolExecutor

import pandas as pd
import pytest

from observe.cache import StageCache


def test_corrupt_cache_is_preserved_and_rebuilt_without_mutating_the_first_run(tmp_path):
    first = StageCache(tmp_path, {'snapshot': 's'}); out = tmp_path / 'one'
    first.frame('factor', {'formula': 'a'}, out, 'factor', lambda: pd.DataFrame({'v': [1]}))
    event = first.events[0]; entry = first.root / 'factor' / event['key']; (entry / 'factor.parquet').write_bytes(b'broken')
    second = StageCache(tmp_path, {'snapshot': 's'})
    result = second.frame('factor', {'formula': 'a'}, tmp_path / 'two', 'factor', lambda: pd.DataFrame({'v': [2]}))
    assert result.v.tolist() == [2] and second.events[0]['result'] == 'corrupt_rebuilt'
    assert pd.read_parquet(out / 'factor.parquet').v.tolist() == [1]
    assert (tmp_path / 'cache' / 'quarantine').exists()
    with pytest.raises(FileExistsError): second.frame('factor', {'formula': 'a'}, out, 'factor', lambda: None)


def test_concurrent_identical_producers_publish_once_and_copy_independent_files(tmp_path):
    calls = []
    def run(k):
        cache = StageCache(tmp_path, {'snapshot': 's'})
        def produce(out): calls.append(k); (out / 'data.json').write_text(json.dumps({'v': 42}))
        cache.materialize('models', {'version': 1}, tmp_path / str(k), produce)
        return cache.report()
    with ThreadPoolExecutor(max_workers = 2) as pool: reports = list(pool.map(run, [1, 2]))
    assert len(calls) == 1 and sum(r['hits'] for r in reports) == 1
    (tmp_path / '1' / 'data.json').write_text('{}')
    assert json.loads((tmp_path / '2' / 'data.json').read_text()) == {'v': 42}


def test_failed_or_bypassed_producers_are_never_published(tmp_path):
    cache = StageCache(tmp_path, {'snapshot': 's'})
    def fail(out): (out / 'partial.json').write_text('{}'); raise RuntimeError('intentional')
    with pytest.raises(RuntimeError): cache.materialize('models', {}, tmp_path / 'one', fail)
    assert cache.events[0]['result'] == 'failed' and not (cache.root / 'models' / cache.events[0]['key']).exists()
    bypass = StageCache(tmp_path, {'snapshot': 's'}, False)
    bypass.materialize('models', {}, tmp_path / 'two', lambda out: (out / 'ok.json').write_text('{}'))
    assert bypass.report()['hits'] == 0 and bypass.events[0]['result'] == 'bypass'
