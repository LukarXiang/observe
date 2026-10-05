import json
from pathlib import Path

import pytest
import yaml

from observe.cli import main
from observe.integrity import verify_run
from observe.replay import ReproduceRefused, reproduce
from observe.strategy.report import write_report
from observe.strategy.run import run_strategy
from tests.integration.helpers import tree_hash
from tests.integration.test_strategies import _fixture


def _spec(tmp_path):
    path = tmp_path / 'merged-test.yaml'
    path.write_text(yaml.safe_dump({'id': 'merged-test', 'title': 'Merge test', 'source': {'file': str(tmp_path / 'original.txt')},
                                   'archetype': 'valuation', 'idea': 'Lowest positive PB',
                                   'select': {'n': 1, 'pipelines': [[{'filter': 'pb_mrq > 0'}, {'sort': 'pb_mrq', 'take': 1}]]},
                                   'schedule': {'freq': 'daily'}}), encoding = 'utf-8')
    return path


@pytest.mark.parametrize('args', [['strategy', 'run'], ['strategy', 'run', 'x', '--config', 'y'],
                                  ['strategy', 'run', 'x', '--queue'], ['strategy', 'run', 'x', 'y', '--output', 'out']])
def test_strategy_cli_rejects_ambiguous_or_unsupported_combinations(args):
    with pytest.raises(SystemExit) as exc: main(args)
    assert exc.value.code == 2


def test_strategy_cli_keeps_both_config_formats(tmp_path, monkeypatch, capsys):
    calls = []
    def rule(root, **params): calls.append(('rule', params)); return {'status': 'success_limited'}
    def spec(root, path, **params): calls.append(('spec', str(path), params)); return {'status': 'success_limited'}
    monkeypatch.setattr('observe.strategies.run_strategy', rule)
    monkeypatch.setattr('observe.strategy.run.run_strategy', spec)
    config = tmp_path / 'rule.yaml'; config.write_text('implementation: bp_component_v1\n', encoding = 'utf-8')
    assert main(['strategy', 'run', '--config', str(config), '--start', '2024-01-01']) == 0
    assert main(['strategy', 'run', 'y2024b-081', '--snapshot', 'frozen']) == 0
    assert calls[0] == ('rule', {'implementation': 'bp_component_v1', 'start': '2024-01-01'})
    assert Path(calls[1][1]).as_posix() == 'strategies/specs/y2024b-081.yaml' and calls[1][2]['snapshot'] == 'frozen'
    assert main(['strategy', 'validate', str(_spec(tmp_path))]) is None
    assert '"invalid": {}' in capsys.readouterr().out


def test_declarative_strategy_freezes_and_reproduces_through_shared_entry(tmp_path, monkeypatch):
    sid, days, _ = _fixture(tmp_path)
    path = _spec(tmp_path)
    def forbidden(*args, **kwargs): raise AssertionError('Must remain offline')
    monkeypatch.setattr('observe.data.sources.baostock.BaoStock.session', forbidden)
    result = run_strategy(tmp_path, path, snapshot = sid, start = days[25], end = days[-1])
    out = Path(result['output'])
    assert result['status'] == 'success_limited'
    assert verify_run(tmp_path, out)['status'] == 'ok'
    before = tree_hash(out)
    again = reproduce(tmp_path, out)
    assert again['reproduction']['result'] == 'match' and again['reproduction']['differences'] == 0
    assert tree_hash(out) == before
    (out / 'spec.yaml').write_text('modified', encoding = 'utf-8')
    with pytest.raises(ReproduceRefused): reproduce(tmp_path, out)


def test_report_before_first_run(tmp_path):
    path = _spec(tmp_path)
    result = write_report(tmp_path, out_csv = tmp_path / 'results.csv', out_md = tmp_path / 'report.md',
                          catalog = tmp_path / 'no-catalog.csv', specs = path.parent)
    assert result['strategies'] == 1 and result['run'] == 0
    assert 'not_run' in (tmp_path / 'report.md').read_text(encoding = 'utf-8')


def test_default_declarative_run_freezes_published_state(tmp_path):
    _, days, _ = _fixture(tmp_path)
    result = run_strategy(tmp_path, _spec(tmp_path), start = days[25], end = days[-1])
    doc = json.loads((Path(result['output']) / 'config.json').read_text(encoding = 'utf-8'))
    assert doc['snapshot_id'] and (tmp_path / 'snapshots' / f"{doc['snapshot_id']}.json").is_file()
