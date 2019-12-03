import os

import pytest

from evo_algorithm import EvoHistory
from evo_storage import EvoStorage


@pytest.fixture(scope='session')
def resource():
    yield 'resource'
    evo_storage_dump_file = 'test_evo_history.db'
    if os.path.exists(evo_storage_dump_file):
        os.remove(evo_storage_dump_file)


def test_evo_storage_save_run_correct(resource):
    dump_file = 'test_evo_history.db'
    storage = EvoStorage(dump_file_path=dump_file)
    history = EvoHistory()
    storage.save_run(key='test_history', evo_history=history)

    decoded_history = storage.run_by_key(key='test_history')

    assert history.last_run_idx == decoded_history.last_run_idx
