import json

import pytest

from observe.data.locks import operation_lock
from observe.data.publication import BatchPublisher


def test_batch_is_published_atomically(tmp_path):
    publisher = BatchPublisher(tmp_path)
    publisher.write_batch("b1", {"daily": ["daily/2024.parquet"], "actions": ["actions.parquet"]})
    assert publisher.read() is None
    state = publisher.publish("b1")
    assert state.batch_id == "b1" and publisher.read().tables["daily"]
    assert json.loads((tmp_path / "PUBLISHED.json").read_text())["batch_id"] == "b1"


def test_batch_manifest_and_lock_are_protected(tmp_path):
    publisher = BatchPublisher(tmp_path)
    publisher.write_batch("b1", {"daily": ["x"]})
    with pytest.raises(FileExistsError): publisher.write_batch("b1", {"daily": ["y"]})
    with operation_lock(tmp_path, "data-writer"):
        assert (tmp_path / "locks").exists()
