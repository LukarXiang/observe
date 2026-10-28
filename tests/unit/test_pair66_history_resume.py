"""A later frozen window can resume without revisiting an immutable earlier result."""
import pytest

from scripts import supplement_pair66_history as batch


@pytest.mark.parametrize('action,helper,key', [('run', 'run_one', 'config_file'), ('verify', 'verify_one', 'label')])
def test_resume_later_window_does_not_retry_completed_first(monkeypatch, tmp_path, action, helper, key):
    rows = [{'label': 'first', 'config_file': 'first.yaml'}, {'label': 'second', 'config_file': 'second.yaml'}]
    monkeypatch.setattr(batch, 'validate', lambda directory: {'selected': rows})
    calls = []
    def execute(root, directory, value, **kwargs):
        if value == rows[0][key]:
            raise FileExistsError('First window is complete')
        calls.append(value)
        return 'ok'
    monkeypatch.setattr(batch, helper, execute)
    assert getattr(batch, action)('data', tmp_path, 'second') == ['ok']
    assert calls == [rows[1][key]]


@pytest.mark.parametrize('action,helper', [('run', 'run_one'), ('verify', 'verify_one')])
def test_unknown_resume_label_is_rejected_before_execution(monkeypatch, tmp_path, action, helper):
    monkeypatch.setattr(batch, 'validate', lambda directory: {'selected': [{'label': 'first', 'config_file': 'first.yaml'}]})
    def forbidden(*args, **kwargs):
        raise AssertionError('No run may execute for an unknown label')
    monkeypatch.setattr(batch, helper, forbidden)
    with pytest.raises(ValueError, match='Unknown frozen window label'):
        getattr(batch, action)('data', tmp_path, '../outside')
