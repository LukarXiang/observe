"""按交接计划打包冻结数据；校验搬迁后的文件，不修改发布状态或实验产物。"""
import argparse
import hashlib
import io
import json
from pathlib import Path, PureWindowsPath
import subprocess
import tarfile


DEFAULT_PLAN = 'docs/handoff/2026-10-04-plan.json'
SKIP = {'.DS_Store', 'registry.sqlite', 'registry.sqlite-wal', 'registry.sqlite-shm', 'jobs.sqlite', 'jobs.sqlite-wal', 'jobs.sqlite-shm'}


def read(path): return json.loads(Path(path).read_text(encoding = 'utf-8'))


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''): h.update(chunk)
    return h.hexdigest()


def safe(root, name):
    rel = Path(name); path = root / rel
    if not name or '\\' in name or PureWindowsPath(name).drive or rel.is_absolute() or '..' in rel.parts or path.resolve() == root or not path.resolve().is_relative_to(root):
        raise ValueError(f'交接路径越过项目目录：{name}')
    if any(p.is_symlink() for p in (path, *path.parents) if p != root and p.is_relative_to(root)):
        raise ValueError(f'交接文件不能使用符号链接：{name}')
    return path


def collect(root, plan):
    """取全部指定快照和当前发布状态的分区并集，保留历史分区；不打包暂存转换副本。"""
    published = read(root / 'data/PUBLISHED.json')
    if published['batch_id'] != plan['published_batch']: raise ValueError('当前发布批次已变化，请创建新的交接计划')
    names = set()
    for item in plan['include'] + [f'data/runs/{rid}' for rid in plan['runs']]:
        path = safe(root, item)
        if not path.exists(): raise FileNotFoundError(path)
        candidates = [path] if path.is_file() else sorted(path.rglob('*'))
        for candidate in candidates:
            if candidate.name in SKIP: continue
            name = candidate.relative_to(root).as_posix(); safe(root, name)
            if candidate.is_file(): names.add(name)
    states = [published]
    for sid in plan['snapshots']:
        name = f'data/snapshots/{sid}.json'; names.add(name); states.append(read(safe(root, name)))
    for state in states:
        for parts in state['tables'].values():
            for entry in parts.values():
                name = 'data/' + entry['file']
                if not safe(root, name).is_file(): raise FileNotFoundError(name)
                names.add(name)
    return sorted(names)


def record(root, name):
    path = safe(root, name); st = path.stat(); digest = sha(path)
    if (st.st_size, st.st_mtime_ns) != (path.stat().st_size, path.stat().st_mtime_ns): raise ValueError(f'校验期间文件变化：{name}')
    return {'path': name, 'bytes': st.st_size, 'sha256': digest}


def project_files(root):
    names = ['pyproject.toml', 'uv.lock', 'scripts/prepare_handoff.py']
    names += [p.relative_to(root).as_posix() for p in (root / 'src').rglob('*.py')]
    names += [p.relative_to(root).as_posix() for p in (root / 'configs').rglob('*.yaml')]
    return [record(root, name) for name in sorted(names)]


def build(root, plan, output):
    output = Path(output).resolve()
    if not output.name.endswith('.tar.gz'): raise ValueError('输出必须以 .tar.gz 结尾')
    manifest_path = output.with_name(output.name[:-7] + '.manifest.json')
    receipt_path = output.with_name(output.name[:-7] + '.receipt.json')
    for path in (output, manifest_path, receipt_path):
        if path.exists(): raise FileExistsError(f'拒绝覆盖已有交接文件：{path}')
    names = collect(root, plan)
    if any(path.is_relative_to(safe(root, item)) for item in plan['include'] for path in (output, manifest_path, receipt_path)):
        raise ValueError('输出不能放入待打包目录')
    print(f'正在校验 {len(names)} 个交接文件……', flush = True)
    signatures = {name: (safe(root, name).stat().st_size, safe(root, name).stat().st_mtime_ns) for name in names}
    manifest = {'schema': 1, 'label': plan['label'], 'base_git_commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd = root, text = True).strip(),
                'published_batch': plan['published_batch'], 'snapshots': plan['snapshots'], 'runs': plan['runs'], 'financial_inputs': plan['financial_inputs'],
                'project_files': project_files(root), 'files': [record(root, name) for name in names]}
    manifest['file_count'] = len(names); manifest['payload_bytes'] = sum(x['bytes'] for x in manifest['files'])
    payload = (json.dumps(manifest, ensure_ascii = False, indent = 2) + '\n').encode()
    output.parent.mkdir(parents = True, exist_ok = True)
    print(f'正在写入 {manifest["payload_bytes"] / 2**30:.2f} GiB 数据……', flush = True)
    with output.open('xb') as stream, tarfile.open(fileobj = stream, mode = 'w:gz', compresslevel = 1) as archive:
        for name in names:
            path = safe(root, name)
            if (path.stat().st_size, path.stat().st_mtime_ns) != signatures[name]: raise ValueError(f'打包期间文件变化：{name}')
            archive.add(path, arcname = name, recursive = False)
            if (path.stat().st_size, path.stat().st_mtime_ns) != signatures[name]: raise ValueError(f'打包期间文件变化：{name}')
        info = tarfile.TarInfo(f'data/handoff/{manifest_path.name}'); info.size = len(payload)
        archive.addfile(info, io.BytesIO(payload))
    with manifest_path.open('xb') as stream: stream.write(payload)
    receipt = {'archive': output.name, 'archive_bytes': output.stat().st_size, 'archive_sha256': sha(output),
               'manifest': manifest_path.name, 'manifest_sha256': sha(manifest_path), 'file_count': len(names), 'payload_bytes': manifest['payload_bytes']}
    with receipt_path.open('x', encoding = 'utf-8') as stream: json.dump(receipt, stream, ensure_ascii = False, indent = 2); stream.write('\n')
    return receipt


def verify(root, manifest, financial_inputs = False):
    if manifest.get('schema') != 1: raise ValueError('不支持的交接清单版本')
    records = manifest['project_files'] + manifest['files'] + (manifest['financial_inputs'] if financial_inputs else [])
    issues = []
    for entry in records:
        path = safe(root, entry['path'])
        if not path.is_file(): issues.append({'path': entry['path'], 'problem': 'missing'})
        elif ('bytes' in entry and path.stat().st_size != entry['bytes']) or sha(path) != entry['sha256']:
            issues.append({'path': entry['path'], 'problem': 'hash_mismatch'})
    return {'status': 'error' if issues else 'ok', 'checked_files': len(records), 'financial_inputs_checked': financial_inputs, 'issues': issues}


def main():
    parser = argparse.ArgumentParser(description = __doc__); parser.add_argument('--root', default = '.')
    subs = parser.add_subparsers(dest = 'action', required = True)
    b = subs.add_parser('build'); b.add_argument('--plan', default = DEFAULT_PLAN); b.add_argument('--output', required = True)
    v = subs.add_parser('verify'); v.add_argument('--manifest', required = True); v.add_argument('--financial-inputs', action = 'store_true')
    args = parser.parse_args(); root = Path(args.root).resolve()
    if args.action == 'build': result = build(root, read(root / args.plan), args.output)
    else: result = verify(root, read(args.manifest), args.financial_inputs)
    print(json.dumps(result, ensure_ascii = False, indent = 2))
    return int(result.get('status') == 'error')


if __name__ == '__main__': raise SystemExit(main())
