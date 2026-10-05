"""实验表排序、分页与 CSV 共享查询；导出有界读取，目标不能覆盖。"""
import csv
import io
import json

import pandas as pd
import pytest
from fastapi.testclient import TestClient

from observe.api.app import create_app
from observe.artifacts import TableCSV, export_table, run_table
from observe.cli import main
from observe.runs import file_sha, write_json


def make(root, parquet = False):
    out = root / 'runs/example'; out.mkdir(parents = True)
    write_json(out / 'status.json', {'run_id': 'example', 'status': 'success'})
    write_json(out / 'config.json', {'kind': 'research'})
    rows = [{'decision_date': '2024-01-02', 'instrument': '000003.SZ', 'model_id': 'ridge', 'score': None, 'memo': '中文,带逗号'},
            {'decision_date': '2024-01-03', 'instrument': '000002.SZ', 'model_id': 'ridge', 'score': 2.0, 'memo': '含"引号"\n和换行'},
            {'decision_date': '2024-01-02', 'instrument': '000002.SZ', 'model_id': 'ridge', 'score': 2.0, 'memo': ''},
            {'decision_date': '2024-01-02', 'instrument': '000001.SZ', 'model_id': 'ridge', 'score': 1.0, 'memo': '保留'},
            {'decision_date': '2024-01-02', 'instrument': '000001.SZ', 'model_id': 'lgbm', 'score': 3.0, 'memo': '其他模型'}]
    if parquet: pd.DataFrame(rows).to_parquet(out / 'predictions.parquet', index = False)
    else: write_json(out / 'predictions.json', rows)
    return out


def csv_rows(stream):
    try: return list(csv.DictReader(io.StringIO(''.join(stream))))
    finally: stream.close()


@pytest.mark.parametrize('parquet', [False, True])
def test_sorted_pages_equal_complete_filtered_export(tmp_path, parquet):
    make(tmp_path, parquet)
    filters = {'model': 'ridge', 'sort_by': 'score', 'descending': True, 'start': '2024-01-02', 'end': '2024-01-03'}
    first = run_table(tmp_path, 'example', 'predictions', limit = 2, **filters)
    second = run_table(tmp_path, 'example', 'predictions', limit = 2, offset = 2, **filters)
    assert first['total'] == 4 and [x['score'] for x in first['rows'] + second['rows']] == [2, 2, 1, None]
    assert first['rows'][0]['decision_date'].startswith('2024-01-02')
    exported = csv_rows(TableCSV(tmp_path, 'example', 'predictions', **filters))
    assert [x['instrument'] for x in exported] == [x['instrument'] for x in first['rows'] + second['rows']]
    assert exported[1]['memo'] == '含"引号"\n和换行' and exported[-1]['score'] == ''
    assert exported[-1]['memo'] == '中文,带逗号'


def test_bound_values_and_sort_names_cannot_change_query(tmp_path):
    make(tmp_path)
    assert run_table(tmp_path, 'example', 'predictions', instrument = "x' OR 1=1 --")['total'] == 0
    for field in ('score desc; drop table x', 'missing', 'score" desc'):
        with pytest.raises(ValueError, match = '排序字段'): TableCSV(tmp_path, 'example', 'predictions', sort_by = field)
    with pytest.raises(ValueError, match = 'sort_by'): run_table(tmp_path, 'example', 'predictions', descending = True)


def test_quoted_real_column_name_is_safe(tmp_path):
    out = make(tmp_path); write_json(out / 'equity.json', [{'date': '2024-01-02', 'a"b': 2}, {'date': '2024-01-02', 'a"b': 1}])
    assert [x['a"b'] for x in run_table(tmp_path, 'example', 'equity', sort_by = 'a"b')['rows']] == [1, 2]


@pytest.mark.parametrize('filters', [{'start': '2024-02-01', 'end': '2024-01-01'}, {'start': 'not-date'}, {'benchmark': 'not-applicable'}])
def test_invalid_filters_fail_before_creating_output(tmp_path, filters):
    make(tmp_path); dest = tmp_path / 'output.csv'
    with pytest.raises(ValueError): export_table(tmp_path, 'example', 'predictions', dest, **filters)
    assert not dest.exists()


def test_export_is_exclusive_and_closes_stream(tmp_path, monkeypatch):
    out = make(tmp_path); target = out / 'predictions.json'; before = file_sha(target)
    closed = []; original = TableCSV.close
    def close(self): original(self); closed.append(self.closed)
    monkeypatch.setattr(TableCSV, 'close', close)
    with pytest.raises(FileExistsError): export_table(tmp_path, 'example', 'predictions', target)
    assert file_sha(target) == before and closed == [True]


def test_export_failure_removes_only_its_partial_file(tmp_path, monkeypatch):
    make(tmp_path); dest = tmp_path / 'broken.csv'; original = TableCSV.__next__
    def fail(self):
        if self.header: raise OSError('模拟读取失败')
        return original(self)
    monkeypatch.setattr(TableCSV, '__next__', fail)
    with pytest.raises(OSError): export_table(tmp_path, 'example', 'predictions', dest)
    assert not dest.exists()


def test_large_export_is_batched_and_not_capped_at_page_size(tmp_path):
    out = make(tmp_path)
    pd.DataFrame({'decision_date': ['2024-01-02'] * 10001, 'instrument': [f'{n:06}.SZ' for n in range(10001)],
                  'model_id': 'ridge', 'score': range(10001)}).to_parquet(out / 'predictions.parquet', index = False)
    stream = TableCSV(tmp_path, 'example', 'predictions', sort_by = 'score')
    try:
        assert len(list(csv.reader(io.StringIO(next(stream))))) == 1
        sizes = [len(list(csv.reader(io.StringIO(chunk)))) for chunk in stream]
        assert sizes == [4096, 4096, 1809] and stream.rows == 10001 and stream.closed
    finally: stream.close()


def test_empty_json_and_empty_filtered_tables(tmp_path):
    out = make(tmp_path); write_json(out / 'fills.json', [])
    assert csv_rows(TableCSV(tmp_path, 'example', 'fills', start = '2024-01-02', instrument = '000001.SZ')) == []
    dest = tmp_path / 'empty.csv'; r = export_table(tmp_path, 'example', 'predictions', dest, model = 'absent')
    assert r['rows'] == 0 and 'instrument' in dest.read_text() and len(dest.read_text().splitlines()) == 1


def test_cash_events_ties_have_stable_secondary_order(tmp_path):
    out = make(tmp_path); write_json(out / 'cash_events.json', [{'date': '2024-01-02', 'kind': 'fee', 'amount': n} for n in [3, 1, 2]])
    assert [run_table(tmp_path, 'example', 'cash_events', limit = 1, offset = n)['rows'][0]['amount'] for n in range(3)] == [1, 2, 3]


def test_cli_api_and_core_share_filters_and_order(tmp_path, capsys):
    make(tmp_path); client = TestClient(create_app(tmp_path))
    filters = {'date': '2024-01-02', 'model': 'ridge', 'sort_by': 'score', 'descending': True}
    page = client.get('/api/runs/example/predictions', params = filters).json()
    main(['--root', str(tmp_path), 'runs', 'table', 'example', 'predictions', '--date', '2024-01-02', '--model', 'ridge', '--sort-by', 'score', '--descending'])
    assert json.loads(capsys.readouterr().out) == page
    response = client.get('/api/runs/example/predictions/csv', params = filters)
    assert response.status_code == 200 and 'text/csv' in response.headers['content-type'] and 'attachment' in response.headers['content-disposition']
    dest = tmp_path / 'cli.csv'
    main(['--root', str(tmp_path), 'runs', 'export', 'example', 'predictions', '--date', '2024-01-02', '--model', 'ridge', '--sort-by', 'score', '--descending', '--output', str(dest)])
    result = json.loads(capsys.readouterr().out)
    assert result['rows'] == page['total'] == 3 and dest.read_bytes() == response.content
    assert client.get('/api/runs/example/predictions/csv', params = {'sort_by': 'missing'}).status_code == 400
    assert client.get('/api/runs/example/predictions/csv', params = {'start': '2025-01-01', 'end': '2024-01-01'}).status_code == 400
    assert client.get('/api/runs/missing/predictions/csv').status_code == 404
