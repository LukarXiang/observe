"""只读核验冻结实验产物及 subruns 引用；不加载模型、不计算指标、不重跑。"""
import hashlib
import json
import re
from pathlib import Path, PureWindowsPath

from .runs import file_sha, resolve_run


SHA = re.compile(r'[0-9a-f]{64}')


def verify_run(root, run, recursive = True):
    """核验 manifest 所列文件。根 manifest 是比对基准，不是外部真实性证明。"""
    source = resolve_run(root, run)
    nodes, links, issues, seen, active, hashes = [], [], [], {}, set(), {}

    def issue(path, code, detail, severity = 'error'):
        issues.append({'path': str(path), 'code': code, 'detail': detail, 'severity': severity})

    def safe(base, name):
        if not isinstance(name, str) or not name or '\\' in name or PureWindowsPath(name).drive:
            raise ValueError('产物路径必须是实验内的相对 POSIX 路径')
        rel = Path(name)
        path = (base / rel).resolve()
        if rel.is_absolute() or '..' in rel.parts or not path.is_relative_to(base) or path == base:
            raise ValueError('产物路径越过实验目录边界')
        return path

    def digest(path):
        st = path.stat(); signature = (st.st_size, st.st_mtime_ns, st.st_ctime_ns, st.st_ino)
        if path not in hashes or hashes[path][0] != signature:
            value = file_sha(path)
            now = path.stat()
            if (now.st_size, now.st_mtime_ns, now.st_ctime_ns, now.st_ino) != signature:
                raise ValueError('核验期间文件发生变化')
            hashes[path] = (signature, value)
        return hashes[path][1]

    def document(base, name):
        try:
            path = safe(base, name); raw = path.read_bytes(); value = json.loads(raw)
            if not isinstance(value, dict): raise ValueError('JSON 必须是对象')
            return value, hashlib.sha256(raw).hexdigest()
        except (OSError, ValueError, RuntimeError) as exc:
            issue(base / name, 'unreadable_document', str(exc)); return None, None

    def references(sub):
        refs = []
        for key in ('research',):
            if sub.get(key) is not None: refs.append((key, sub[key]))
        if 'arms' in sub:
            if not isinstance(sub['arms'], dict): raise ValueError('arms 必须是对象')
            refs.extend((f'arms/{key}', value) for key, value in sorted(sub['arms'].items()))
        for key in ('backtests', 'benchmarks'):
            if key not in sub: continue
            if not isinstance(sub[key], list): raise ValueError(f'{key} 必须是列表')
            refs.extend((f'{key}/{i}', value) for i, value in enumerate(sub[key]))
        unknown = set(sub) - {'research', 'arms', 'backtests', 'benchmarks'}
        if unknown: raise ValueError(f'未支持的子实验引用字段：{sorted(unknown)}')
        return refs

    def locate(parent, ref):
        rid = ref.get('run_id')
        if not isinstance(rid, str) or not rid or Path(rid).name != rid or rid in ('.', '..') or '\\' in rid:
            raise ValueError('子实验缺少合法 run_id')
        output = ref.get('output')
        if output is not None and not isinstance(output, str): raise ValueError('output 必须是路径字符串')
        if output and Path(output).is_dir(): return Path(output).resolve()
        # 实验迁移后先按其自身布局定位，不依赖登记库曾经索引过。
        for candidate in (parent / 'research' / rid, parent / 'variants' / rid, parent / 'benchmarks' / rid, parent.parent / rid):
            if candidate.is_dir(): return candidate.resolve()
        return resolve_run(root, rid)

    def visit(out, depth = 0):
        if out in active:
            issue(out, 'reference_cycle', '子实验引用形成环'); return seen[out]
        if out in seen: return seen[out]
        if depth > 64 or len(nodes) >= 1000:
            issue(out, 'traversal_limit', '子实验引用超过 64 层或 1000 个实验'); return None
        node = {'output': str(out), 'run_id': None, 'run_status': None, 'manifest_sha256': None, 'checked_files': 0}
        seen[out] = node; nodes.append(node); active.add(out)
        status, status_sha = document(out, 'status.json')
        if status is not None:
            node.update(run_id = status.get('run_id'), run_status = status.get('status'))
            if not isinstance(node['run_id'], str) or not node['run_id']: issue(out / 'status.json', 'invalid_identity', '缺少 run_id')
            if node['run_status'] not in ('pending', 'running', 'success', 'success_limited', 'partial', 'blocked', 'mismatch', 'failed'):
                issue(out / 'status.json', 'invalid_status', '未知或缺失的运行状态')
        if not (out / 'manifest.json').exists():
            incomplete = node['run_status'] in ('pending', 'running', 'failed')
            issue(out / 'manifest.json', 'unsealed', '实验尚未封存，不能确认产物完整性', 'warning' if incomplete else 'error')
            active.remove(out); return node
        manifest, manifest_sha = document(out, 'manifest.json'); node['manifest_sha256'] = manifest_sha
        if manifest is None: active.remove(out); return node
        if node['run_status'] in ('pending', 'running'):
            issue(out / 'status.json', 'unfinished_run', '运行未结束，不能确认最终产物完整性', 'warning')
        if status is not None and (manifest.get('run_id') != node['run_id'] or manifest.get('status') != node['run_status']):
            issue(out / 'manifest.json', 'identity_mismatch', 'manifest 与 status 的实验编号或运行状态不一致')
        files = manifest.get('files')
        if not isinstance(files, dict) or not files:
            issue(out / 'manifest.json', 'invalid_manifest', 'files 必须是非空的路径 → SHA256 对象')
            active.remove(out); return node
        valid = set()
        for name, expected in files.items():
            try:
                path = safe(out, name)
                if path == out / 'manifest.json': raise ValueError('manifest 不得引用自身')
                if not isinstance(expected, str) or SHA.fullmatch(expected) is None: raise ValueError('SHA256 必须为 64 位小写十六进制')
                actual = digest(path); node['checked_files'] += 1
                if actual != expected: issue(path, 'hash_mismatch', f'expected={expected}, actual={actual}')
                else: valid.add(name)
            except FileNotFoundError: issue(out / name, 'missing_file', '冻结产物不存在')
            except (OSError, ValueError, RuntimeError) as exc: issue(out / name, 'invalid_artifact', str(exc))
        for name in ('status.json', 'config.json'):
            if name not in files: issue(out / name, 'untracked_control', '必要控制文件未被 manifest 冻结')
        if status_sha is not None and files.get('status.json') != status_sha:
            issue(out / 'status.json', 'untrusted_status', '所读取状态与冻结指纹不一致')
        config = None
        if 'config.json' in valid:
            config, sha = document(out, 'config.json')
            if sha != files['config.json']: issue(out / 'config.json', 'changed_during_check', '控制文件在核验期间变化'); config = None
        kind = (config or {}).get('kind') or (status or {}).get('kind') or manifest.get('kind')
        has_subruns = 'subruns.json' in files or (out / 'subruns.json').exists()
        if kind in ('experiment', 'paired') and not has_subruns:
            issue(out / 'subruns.json', 'missing_references', '组合实验缺少子实验引用')
        if recursive and has_subruns:
            if 'subruns.json' not in valid:
                issue(out / 'subruns.json', 'untrusted_references', '引用未冻结或指纹不匹配，不沿其继续读取')
            else:
                sub, sha = document(out, 'subruns.json')
                if sha != files['subruns.json']:
                    issue(out / 'subruns.json', 'changed_during_check', '引用在核验期间变化'); sub = None
                if sub is not None:
                    if kind == 'experiment' and (not isinstance(sub.get('research'), dict) or 'backtests' not in sub):
                        issue(out / 'subruns.json', 'missing_references', '完整实验必须声明 research 与 backtests')
                    if kind == 'paired' and (not isinstance(sub.get('arms'), dict) or not sub['arms'] or 'backtests' not in sub):
                        issue(out / 'subruns.json', 'missing_references', '成对实验必须声明 arms 与 backtests')
                    if kind == 'paired' and node['run_status'] in ('success', 'success_limited') and isinstance(sub.get('arms'), dict) and set(sub['arms']) != {'base', 'extended'}:
                        issue(out / 'subruns.json', 'missing_references', '成功的成对实验必须包含 base 与 extended 两侧')
                    try: refs = references(sub)
                    except ValueError as exc: issue(out / 'subruns.json', 'invalid_references', str(exc)); refs = []
                    for label, ref in refs:
                        link = {'parent': str(out), 'reference': label}; links.append(link)
                        try:
                            if not isinstance(ref, dict): raise ValueError('子实验引用必须是对象')
                            child = locate(out, ref); link['output'] = str(child)
                            target = visit(child, depth + 1)
                            if target is None: continue
                            if ref['run_id'] != target['run_id']: issue(child, 'reference_identity_mismatch', f'{label} 的 run_id 与实际子实验不同')
                            if ref.get('status') is not None and ref['status'] != target['run_status']:
                                issue(child, 'reference_status_mismatch', f'{label} 的状态与实际子实验不同')
                            expected = ref.get('manifest_sha256')
                            if expected is None: issue(child, 'unpinned_reference', f'{label} 未记录子实验 manifest 指纹', 'warning')
                            elif expected != target['manifest_sha256']: issue(child / 'manifest.json', 'reference_hash_mismatch', f'{label} 的子实验 manifest 指纹与冻结引用不同')
                        except (OSError, ValueError, RuntimeError) as exc: issue(out / 'subruns.json', 'unresolved_reference', f'{label}: {exc}')
        # 捕捉读取后的常见并发改写；不会将正在变化的实验报告成完整。
        try:
            if digest(safe(out, 'manifest.json')) != manifest_sha: issue(out / 'manifest.json', 'changed_during_check', 'manifest 在核验期间变化')
        except (OSError, ValueError, RuntimeError) as exc: issue(out / 'manifest.json', 'changed_during_check', str(exc))
        active.remove(out); return node

    visit(source)
    for path, (signature, _) in hashes.items():
        try:
            st = path.stat()
            if (st.st_size, st.st_mtime_ns, st.st_ctime_ns, st.st_ino) != signature:
                issue(path, 'changed_during_check', '文件在核验期间变化')
        except OSError as exc: issue(path, 'changed_during_check', str(exc))
    errors = sum(x['severity'] == 'error' for x in issues); warnings = len(issues) - errors
    return {'status': 'error' if errors else ('warning' if warnings else 'ok'), 'output': str(source), 'recursive': recursive,
            'scope': 'manifest_listed_artifacts_and_subruns' if recursive else 'manifest_listed_artifacts',
            'summary': {'runs': len(nodes), 'references': len(links), 'unique_files_hashed': len(hashes),
                        'bytes_hashed': sum(v[0][0] for v in hashes.values()), 'errors': errors, 'warnings': warnings},
            'runs': nodes, 'references': links, 'issues': issues}
