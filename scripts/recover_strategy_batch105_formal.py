"""Recover the omitted precision-helper copy in a new formal-check directory."""
import argparse
import json
from pathlib import Path
from types import SimpleNamespace

from observe.runs import file_sha
from scripts.run_strategy_batch4 import read
from scripts.verify_strategy_batch3 import require

ORIGINAL = Path('scripts/accept_strategy_batch105.py')
FAILURE = Path('data/staging/strategies-batch105/20261011-formal-source-validation')
CHECK = Path('data/staging/strategies-batch105/20261011-formal-source-validation-recovery')


def program():
    manifest = read(FAILURE/'copy-manifest.json')
    require(file_sha(ORIGINAL) == manifest[str(ORIGINAL)] == file_sha(FAILURE/'isolated'/ORIGINAL),
        'Original failed formal driver changed')
    source = ORIGINAL.read_text()
    replacements = {
        "Path('data/staging/strategies-batch105/20261011-formal-source-validation')":'Path('+repr(str(CHECK))+')',
        "|{'scripts/accept_strategy_batch105.py'}":
            "|{'scripts/accept_strategy_batch105.py','scripts/recover_strategy_batch105_stage.py','scripts/recover_strategy_batch105_formal.py'}",
        "('scripts.accept_strategy_batch105','scripts.reference_strategy_batch105','scripts.reference_strategy_batch105_age','observe.factors.expr')":
            "('scripts.accept_strategy_batch105','scripts.reference_strategy_batch105','scripts.reference_strategy_batch105_age','observe.factors.expr','scripts.recover_strategy_batch105_stage')",
    }
    for before,after in replacements.items():
        require(source.count(before) == 1,'Formal recovery patch scope differs: '+before)
        source = source.replace(before,after,1)
    namespace = {'__name__':'scripts.batch105_recovered_acceptance','__file__':str(ORIGINAL.resolve())}
    exec(compile(source,'<batch105-explicit-formal-copy-recovery>','exec'),namespace)
    return SimpleNamespace(**namespace)


acceptance = program()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument('action',choices=('check','prepare','protect','publish'))
    action = parser.parse_args().action
    result = acceptance.check() if action == 'check' else acceptance.prepare() if action == 'prepare' else \
        acceptance.protect('data',acceptance.DIRECTORY) if action == 'protect' else acceptance.publish()
    print(json.dumps(result,ensure_ascii=False))
