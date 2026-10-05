"""按阶段内容寻址的本地缓存；校验后复制产物，实验文件不与缓存共享可写 inode。"""
import hashlib, inspect, json, os, secrets, shutil, tempfile
from pathlib import Path

import pandas as pd
from filelock import FileLock

from .runs import canonical, file_sha, write_json


def digest(value): return hashlib.sha256(json.dumps(canonical(value), sort_keys = True, separators = (',', ':')).encode()).hexdigest()


def code_version(*modules):
    base = Path(__file__).parent
    return {name: file_sha(base / name) for name in modules}


def function_version(*functions): return {f.__name__: digest(inspect.getsource(f)) for f in functions}


class StageCache:
    def __init__(self, root, inputs, enabled = True):
        self.root, self.inputs, self.enabled, self.events = Path(root) / 'cache' / 'stages', inputs, enabled, []

    def materialize(self, stage, payload, output, produce):
        """produce(dir) 只写该阶段产物。失败不发布；损坏条目隔离保留后重算。"""
        if not stage.replace('_', '').isalnum(): raise ValueError('缓存阶段名不合法')
        output = Path(output); output.mkdir(parents = True, exist_ok = True)
        identity = canonical({'schema': 1, 'stage': stage, 'inputs': self.inputs, 'payload': payload}); key = digest(identity)
        event = {'stage': stage, 'key': key, 'result': 'bypass' if not self.enabled else 'miss'}; self.events.append(event)
        if not self.enabled: produce(output); return
        parent = self.root / stage; parent.mkdir(parents = True, exist_ok = True); entry = parent / key
        with FileLock(str(parent / f'{key}.lock')):
            valid = False
            if entry.exists():
                try:
                    manifest = json.loads((entry / 'manifest.json').read_text(encoding = 'utf-8'))
                    valid = manifest['identity'] == identity and bool(manifest['files'])
                    for name, sha in manifest['files'].items():
                        p = Path(name)
                        if p.is_absolute() or '..' in p.parts or p.as_posix() == 'manifest.json': valid = False; break
                        f = entry / p
                        if f.is_symlink() or not f.is_file() or file_sha(f) != sha: valid = False; break
                except (OSError, ValueError, KeyError, TypeError): valid = False
                if valid: event['result'] = 'hit'
                else:
                    quarantine = self.root.parent / 'quarantine'; quarantine.mkdir(parents = True, exist_ok = True)
                    dest = quarantine / f'{stage}-{key}-{secrets.token_hex(4)}'; os.replace(entry, dest)
                    event.update(result = 'corrupt_rebuilt', quarantine = str(dest))
            if not valid:
                temp = Path(tempfile.mkdtemp(prefix = f'.{key}-', dir = parent))
                try:
                    produce(temp)
                    files = {p.relative_to(temp).as_posix(): file_sha(p) for p in sorted(temp.rglob('*')) if p.is_file()}
                    if not files or 'manifest.json' in files: raise ValueError('缓存阶段必须产出文件且不能写 manifest.json')
                    manifest = {'identity': identity, 'files': files}; write_json(temp / 'manifest.json', manifest); os.replace(temp, entry)
                except Exception:
                    event['result'] = 'failed'; event['partial_path'] = str(temp); raise
            for name in manifest['files']:
                target = output / name
                if target.exists(): raise FileExistsError(f'缓存不能覆盖实验产物：{target}')
                target.parent.mkdir(parents = True, exist_ok = True); shutil.copyfile(entry / name, target)
            event['files'] = manifest['files']

    def frame(self, stage, payload, output, name, compute):
        self.materialize(stage, payload, output, lambda dest: compute().to_parquet(dest / f'{name}.parquet', index = False))
        return pd.read_parquet(Path(output) / f'{name}.parquet')

    def report(self):
        return {'enabled': self.enabled, 'events': self.events, 'hits': sum(e['result'] == 'hit' for e in self.events),
                'note': '命中由配置、上游内容、快照、相关代码和依赖版本共同决定；离线复现强制绕过缓存'}
