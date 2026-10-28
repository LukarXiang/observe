"""Bounded event-page recovery must not retry DNS or execute supplier JavaScript."""
from pathlib import Path
from unittest.mock import patch

import pytest
import requests

from scripts import probe_strategy_batch89_events as batch


@pytest.fixture
def restored():
    if not batch.stage.DIRECTORY.exists() or not batch.REFERENCE.exists():
        pytest.skip('Restore prior all-weather source stage and API project')


class Response:
    status_code = 200

    def __init__(self, payload):
        self.payload = payload
        self.closed = False

    def raise_for_status(self):
        pass

    def iter_content(self, chunk_size):
        yield self.payload

    def close(self):
        self.closed = True


class Session:
    def __init__(self, payload=None):
        self.payload = payload; self.calls = []; self.closed = False; self.responses = []

    def get(self, url, **kwargs):
        self.calls.append((url, kwargs))
        if self.payload is None:
            raise requests.ConnectionError('NameResolutionError: Failed to resolve synthetic.invalid')
        response = Response(self.payload); self.responses.append(response)
        return response

    def close(self):
        self.closed = True


def test_event_parser_keeps_labels_and_cells_without_inferred_event_times():
    result = batch.parse_tables('<table><tr><th>event_date</th><th>ratio</th></tr>'
                               '<tr><td>2022-01-14</td><td>1:5</td></tr></table>')
    assert result == [{'columns': ['event_date', 'ratio'], 'rows': [['2022-01-14', '1:5']]}]


def test_dns_failure_stops_remaining_same_domain_requests(tmp_path, restored):
    client = Session()
    result = batch.probe(tmp_path/'new-events', client)
    assert result['attempted_requests'] == 1 and result['responses'] == result['candidate_pages'] == 0
    assert result['results'][0]['DNS_failure'] is True
    assert [r['status'] for r in result['results']] == ['failed', 'not_requested', 'not_requested', 'not_requested']
    assert len(client.calls) == 1 and client.closed and not client.trust_env
    assert client.calls[0][1] == {'timeout': (4, 8), 'allow_redirects': False, 'stream': True}
    assert result['strict_usable'] is False and result['not_a_backtest']


def test_success_archives_pages_as_candidates_never_downloads_JS(tmp_path, restored):
    client = Session(b'<table><tr><th>event_date</th><th>ratio</th></tr>'
                     b'<tr><td>2022-01-14</td><td>1:5</td></tr></table>')
    result = batch.probe(tmp_path/'new-events', client)
    assert result['candidate_pages'] == result['responses'] == result['attempted_requests'] == 4
    assert all('fhsp_' in url and url.endswith('.html') for url, _ in client.calls)
    assert all(r['strict_usable'] is False and r['table_rows'] == 1 for r in result['results'])
    assert all(Path(r['raw_file']).is_file() and Path(r['tables_file']).is_file() for r in result['results'])
    assert all(r.closed for r in client.responses) and client.closed


def test_response_limit_blocks_parse_and_preserves_no_fake_sample(tmp_path, restored):
    client = Session(b'oversize')
    with patch.object(batch, 'MAX_BYTES', 3):
        result = batch.probe(tmp_path/'new-events', client)
    assert result['candidate_pages'] == 0 and result['responses'] == 4
    assert all(r['status'] == 'failed' and 'raw_file' not in r for r in result['results'])
    assert all(r.closed for r in client.responses)


def test_existing_archive_is_not_overwritten(tmp_path, restored):
    dest = tmp_path/'events'; dest.mkdir(); original = dest/'untouched'
    original.write_text('frozen', encoding='ascii')
    client = Session()
    with pytest.raises(FileExistsError):
        batch.probe(dest, client)
    assert original.read_text() == 'frozen' and not client.calls
