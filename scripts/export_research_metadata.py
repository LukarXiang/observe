"""Export bounded, read-only research metadata; verify it without research payloads."""
import argparse
from collections import Counter
import hashlib
import json
import math
import os
from pathlib import Path, PureWindowsPath
import tempfile


MAX_FILE = 5 * 1024**2
MAX_EXPORT = 20 * 1024**2


def safe(root, name):
    if not isinstance(name, str) or not name: raise ValueError(f'Unsafe repository path: {name}')
    root = Path(root).resolve(); rel = Path(name)
    if '\\' in name or PureWindowsPath(name).drive or rel.is_absolute() or '..' in rel.parts:
        raise ValueError(f'Unsafe repository path: {name}')
    path = root / rel
    # A relative path without parent traversal is confined once every symlink is rejected.
    if path == root: raise ValueError(f'Path escapes root: {name}')
    for parent in (path, *path.parents):
        if parent == root: break
        if parent.is_symlink(): raise ValueError(f'Symlink is not an input/output: {name}')
    return path


def digest(data): return hashlib.sha256(data).hexdigest()


def file_sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024**2), b''): h.update(chunk)
    return h.hexdigest()


def finite(value):
    if isinstance(value, float) and not math.isfinite(value): return None
    if isinstance(value, list): return [finite(v) for v in value]
    if isinstance(value, dict): return {k: finite(v) for k, v in value.items()}
    return value


def encoded(value): return (json.dumps(finite(value), ensure_ascii = False, sort_keys = True, indent = 2, allow_nan = False) + '\n').encode()


def validated_report(document, run_id, report, reproduction_id):
    if document.get('status') != 'ok': return False
    candidates = document.get('strategies', [document])
    return any((v.get('run_id') or v.get('run', {}).get('run_id')) == run_id
               and v.get('report') == report
               and v.get('reproduction', {}).get('run_id') == reproduction_id for v in candidates)


def coverage_summary(value):
    if not isinstance(value, list): return value
    return {'rows': len(value), 'first': value[0] if value else None, 'last': value[-1] if value else None,
            'detail_policy': 'daily coverage remains in source report payload; these are boundary examples'}


class Export:
    def __init__(self, root):
        self.root = Path(root).resolve(); self.files = {}; self.inputs = {}; self.assets = {}; self.run_index = []; self.originals = set()

    def read(self, name):
        data = safe(self.root, name).read_bytes(); self.bind(name, data)
        return json.loads(data)

    def bind(self, name, data):
        sha = digest(data)
        if name in self.inputs and self.inputs[name] != sha: raise ValueError(f'Input changed during export: {name}')
        self.inputs[name] = sha

    def put(self, name, value, raw = False):
        safe(self.root, 'data/metadata/' + name)
        data = value if raw else encoded(value)
        if len(data) > MAX_FILE: raise ValueError(f'Metadata file exceeds 5 MiB: {name}')
        if name in self.files and self.files[name] != data: raise ValueError(f'Conflicting export: {name}')
        self.files[name] = data
        return 'data/metadata/' + name

    def copy(self, source, target):
        data = safe(self.root, source).read_bytes(); self.bind(source, data)
        self.originals.add(target)
        return self.put(target, data, raw = True)

    def reference(self, name):
        data = safe(self.root, name).read_bytes(); self.bind(name, data)
        return {'path': name, 'sha256': digest(data), 'availability': 'git_document' if not name.startswith(('data/', 'repo/')) else 'payload_reference'}

    def asset(self, name, sha = None, **info):
        safe(self.root, name)
        old = self.assets.setdefault(name, {'path': name, 'file_sha256': None})
        if sha and old['file_sha256'] and old['file_sha256'] != sha: raise ValueError(f'Conflicting payload SHA: {name}')
        if sha: old['file_sha256'] = sha
        for key, value in info.items():
            if key in old and old[key] != value: raise ValueError(f'Conflicting payload metadata: {name}/{key}')
            old[key] = value

    def state(self, state):
        for table, parts in state['tables'].items():
            for entry in parts.values():
                self.asset('data/' + entry['file'], entry.get('file_sha256'), table = table, rows = entry['rows'], store_fingerprint = entry['sha'])


def export_runs(builder, catalog, evidence, superseded):
    identities = {(r['path'], r['bytes_sha256']): r['strategy_id'] for r in catalog}
    matches = {}
    for item in evidence:
        matches.setdefault(item['run_id'], []).append(item)
    runs_root = safe(builder.root, 'data/runs')
    directories = {p.parent for pattern in ('*/config.json', '*/status.json') for p in runs_root.glob(pattern)}
    for out in sorted(directories):
        path = out / 'config.json'; rel = path.relative_to(builder.root).as_posix()
        config = builder.read(rel) if path.is_file() else {}; rid = out.name
        status_path = out / 'status.json'
        status = builder.read(status_path.relative_to(builder.root).as_posix()) if status_path.is_file() else {'status': 'incomplete_missing_status'}
        report_path = out / 'report.json'; report = builder.read(report_path.relative_to(builder.root).as_posix()) if report_path.is_file() else {}
        source = config.get('source', {}); source_path = source.get('path', '').removeprefix('repo/量化策略源代码/')
        sid = source.get('strategy_id') or identities.get((source_path, source.get('bytes_sha256')))
        validation = []
        for item in matches.get(rid, []):
            ref = builder.reference(item['validation_file'])
            if ref['sha256'] != item['validation_sha256']: raise ValueError(f'Validation SHA mismatch: {rid}')
            v = builder.read(item['validation_file'])
            if not validated_report(v, rid, report, item['reproduction_run_id']): raise ValueError(f'Validation/report mismatch: {rid}')
            comparison = f"data/runs/{item['reproduction_run_id']}/comparison.json"; c = builder.read(comparison)
            if c.get('result') != 'match' or c.get('differences'): raise ValueError(f'Reproduction mismatch: {rid}')
            validation.append({'validation': ref, 'comparison': builder.reference(comparison), 'reproduction_run_id': item['reproduction_run_id'], 'result': 'match', 'differences': 0})
        metrics = out / 'metrics.json'; trading = out / 'trading.json'; limits = out / 'limitations.json'
        summary = {'schema': 1, 'run_id': rid, 'kind': config.get('kind') or status.get('kind', 'backtest'), 'strategy_id': sid,
                   'implementation': config.get('config', {}).get('implementation', config.get('config', {}).get('name')),
                   'execution_status': status['status'], 'evidence_status': 'superseded_numeric_validation_failed' if rid in superseded else 'accepted_limited' if validation else 'archived_unreviewed',
                   'superseded_by': superseded.get(rid), 'snapshot': config.get('snapshot_id'), 'configuration': config.get('config'),
                   'recorded_environment': config.get('environment'), 'rules': report.get('strategy', {}).get('review', {}),
                   'parameters': report.get('parameters'), 'period': report.get('period'), 'decision_time': report.get('decision_time'),
                   'execution_time': report.get('execution_time'), 'portfolio': report.get('portfolio'), 'universe': report.get('universe'),
                   'results': report.get('results'), 'benchmark': report.get('benchmark'), 'coverage': coverage_summary(report.get('coverage')),
                   'metrics': builder.read(metrics.relative_to(builder.root).as_posix()) if metrics.is_file() else None,
                   'trading': builder.read(trading.relative_to(builder.root).as_posix()) if trading.is_file() else None,
                   'limitations': builder.read(limits.relative_to(builder.root).as_posix()) if limits.is_file() else report.get('limitations'),
                   'reproduction': status.get('reproduction'), 'validation': validation, 'original_strategy_complete': False,
                   'payload_verification': 'not_reaudited', 'references': [],
                   'configuration_available': path.is_file(), 'reproduce_of': config.get('reproduce_of') or status.get('reproduce_of'),
                   'result_scope': 'configured_run' if path.is_file() else 'status_or_comparison_only'}
        for reference_path in (path, status_path):
            if reference_path.is_file(): summary['references'].append(builder.reference(reference_path.relative_to(builder.root).as_posix()))
        if report_path.is_file(): summary['references'].append(builder.reference(report_path.relative_to(builder.root).as_posix()))
        comparison_path = out / 'comparison.json'
        if comparison_path.is_file():
            summary['comparison'] = builder.read(comparison_path.relative_to(builder.root).as_posix())
            summary['references'].append(builder.reference(comparison_path.relative_to(builder.root).as_posix()))
        target = builder.put(f'runs/{rid}/summary.json', summary)
        builder.run_index.append({k: summary[k] for k in ('run_id', 'kind', 'strategy_id', 'implementation', 'execution_status', 'evidence_status', 'superseded_by', 'snapshot')} | {'summary': target})
        manifest = out / 'manifest.json'
        if manifest.is_file():
            doc = builder.read(manifest.relative_to(builder.root).as_posix())
            builder.asset(manifest.relative_to(builder.root).as_posix(), file_sha(manifest), kind = 'run_manifest')
            for name, sha in doc.get('files', {}).items(): builder.asset(out.relative_to(builder.root).as_posix() + '/' + name, sha, kind = 'run_artifact')
        dm = out / 'data_manifest.json'
        if dm.is_file():
            doc = builder.read(dm.relative_to(builder.root).as_posix())
            builder.state({'tables': doc.get('used', {})})


def prepare(root):
    b = Export(root); policy = b.read('configs/research_metadata.json')
    for capability in policy['capabilities']:
        for path in capability.get('code', []): b.reference(path)
    progress = b.read('data/catalog/strategies/progress/latest.json')
    latest = b.read(progress['directory'] + '/summary.json')
    if digest(safe(b.root, progress['directory'] + '/summary.json').read_bytes()) != progress['summary_sha256']: raise ValueError('Progress summary SHA mismatch')
    accepted = b.read(f"data/catalog/strategies/progress/{policy['accepted_checkpoint']}/summary.json")
    receipt = b.read(policy['accepted_receipt'])
    if not str(receipt.get('status')).startswith('ok') or receipt['progress']['checkpoint'] != accepted['checkpoint']: raise ValueError('Accepted checkpoint lacks matching receipt')
    catalogs = {}
    for state in (accepted, latest):
        base = f"data/catalog/strategies/{state['catalog_id']}"; records = b.read(base + '/catalog.json')
        if b.inputs[base + '/catalog.json'] != state['catalog_sha256']: raise ValueError('Catalog SHA mismatch')
        catalogs[state['catalog_id']] = b.copy(base + '/catalog.json', f"catalogs/{state['catalog_id']}/catalog.json")
        b.copy(base + '/summary.json', f"catalogs/{state['catalog_id']}/summary.json")
        b.copy(f"data/catalog/strategies/progress/{state['checkpoint']}/summary.json", f"progress/{state['checkpoint']}/summary.json")
    catalog = b.read(f"data/catalog/strategies/{latest['catalog_id']}/catalog.json")
    if len(catalog) != latest['sources'] or sum(r['review_status'] == '人工审查完成' for r in catalog) != latest['manually_reviewed']: raise ValueError('Catalog/progress count mismatch')
    for r in catalog: b.asset('repo/量化策略源代码/' + r['path'], r['bytes_sha256'], kind = 'strategy_source')
    evidence = b.read('data/catalog/strategies/implementation-evidence.json')
    b.put('evidence/implementation-evidence.json', evidence)
    resolution = b.read(policy['numeric_resolution'])
    b.copy(policy['numeric_resolution'], 'evidence/numeric-resolution.json')
    superseded = {x['original_run_id']: x['stable_run_id'] for x in resolution['variants']}
    export_runs(b, catalog, evidence, superseded)
    b.put('runs/index.json', b.run_index)
    local_strategies = [r for r in b.run_index if r['kind'] == 'strategy']
    if len(local_strategies) != latest['local_strategy_runs']: raise ValueError('Parent strategy run count mismatch')
    published = b.read('data/PUBLISHED.json'); b.state(published)
    b.asset('data/PUBLISHED.json', b.inputs['data/PUBLISHED.json'], kind = 'release_pointer')
    if published['batch_id'] != latest['published_batch']: raise ValueError('Published state/progress mismatch')
    b.copy('data/PUBLISHED.json', f"releases/{published['batch_id']}/published.json")
    snapshots = []
    for path in sorted(safe(b.root, 'data/snapshots').glob('*.json')):
        name = path.relative_to(b.root).as_posix(); s = b.read(name); b.state(s)
        b.asset(name, b.inputs[name], kind = 'snapshot')
        snapshots.append({'snapshot_id': path.stem, 'batch_id': s['batch_id'], 'metadata': b.copy(name, 'snapshots/' + path.name)})
    missing_snapshots = sorted({r['snapshot'] for r in b.run_index if r['snapshot']} - {s['snapshot_id'] for s in snapshots})
    for path in sorted(safe(b.root, 'data/batches').glob('*.json')):
        name = path.relative_to(b.root).as_posix(); s = b.read(name)
        if s.get('status') in ('published', 'rejected'): b.copy(name, f"releases/{s['batch_id']}/batch.json")
    imports = []
    for path in sorted(safe(b.root, 'data/raw/external_financials').glob('*.json')):
        name = path.relative_to(b.root).as_posix(); doc = b.read(name)
        if doc.get('status') != 'published': continue
        iid = path.stem; audit = f'data/audits/financials/{iid}.json'; a = b.read(audit)
        imports.append({'import_id': iid, 'rows': doc['result']['rows'], 'strict_usable_rows': doc['result']['strict_usable_rows'],
                        'fields': b.copy(f'data/catalog/financial-fields-{iid}.json', f'financials/{iid}/fields.json'),
                        'receipt': b.copy(name, f'financials/{iid}/import-receipt.json'), 'audit': b.copy(audit, f'financials/{iid}/audit.json'),
                        'limitations': a.get('limitations')})
        for name, sha in doc.get('partition_bytes', {}).items(): b.asset('data/' + name, sha)
    evidence_index = []
    for path in sorted(safe(b.root, 'docs/handoff').glob('*.json')):
        name = path.relative_to(b.root).as_posix(); doc = b.read(name)
        evidence_index.append({'reference': b.reference(name), 'status': doc.get('status', 'not_a_batch_receipt'), 'coverage': 'git_document'})
    b.put('evidence/index.json', evidence_index)
    for name, records in [('tasks/index.json', policy['tasks']), ('blockers.json', policy['blockers']), ('capabilities.json', policy['capabilities'])]:
        b.put(name, records)
    decisions = []
    for decision in policy['decisions']:
        decision = dict(decision)
        if decision.get('approval_file'):
            doc = b.read(decision['approval_file'])
            if doc.get('status') not in ('approved', 'approved_by_user'): raise ValueError(f"Approval is not approved: {decision['id']}")
            decision['approval_copy'] = b.copy(decision['approval_file'], f"decisions/{decision['id']}-approval.json")
        decisions.append(decision)
    b.put('decisions/index.json', decisions)
    for case in policy['examples']:
        value = b.read(case['source'])
        if case.get('fields'): value = {key: value[key] for key in case['fields']}
        b.put(f"examples/{case['id']}/case.json", {'case_id': case['id'], 'nature': case['nature'], 'purpose': case['purpose'],
                                               'reference': b.reference(case['source']), 'case': value})
    # These are source-bound summaries, not a second calculation of economic results.
    representative = b.read('docs/handoff/2026-10-08-strategy-progress-report.json')
    b.put('runs/representative.json', representative)
    bundles = []
    for path in sorted(safe(b.root, 'data/handoff').glob('*.receipt.json')) if safe(b.root, 'data/handoff').exists() else []:
        if path.name.endswith('.bundle.receipt.json'): continue  # Git bundle sidecars are not payload archive receipts.
        name = path.relative_to(b.root).as_posix(); r = b.read(name)
        archive = 'data/handoff/' + r['archive']; manifest = 'data/handoff/' + r['manifest']
        present = safe(b.root, archive).is_file() and safe(b.root, manifest).is_file()
        record = {'receipt': b.copy(name, 'bundles/' + path.name), 'archive': archive, 'manifest': manifest,
                  'availability_at_export': 'present_not_reaudited' if present else 'missing_local', 'archive_sha256': r['archive_sha256']}
        if safe(b.root, manifest).is_file():
            if file_sha(safe(b.root, manifest)) != r['manifest_sha256']: raise ValueError('Bundle manifest SHA mismatch')
            record['metadata_manifest'] = b.copy(manifest, 'bundles/' + Path(manifest).name)
        bundles.append(record)
    b.put('bundles/index.json', {'bundles': bundles, 'current_full_archive': 'not_packaged' if not bundles else 'see_individual_bundle_scope',
                               'old_remote_handoff': 'missing_local', 'location': 'repository-local data/handoff; not uploaded by Git'})
    # Audit controls affect replay limitations even when trading inputs are identical.
    for path in sorted(safe(b.root, 'data/audits').rglob('*')):
        if path.is_file() and path.suffix in ('.json', '.csv'):
            name = path.relative_to(b.root).as_posix(); ref = b.reference(name)
            b.asset(name, ref['sha256'], kind = 'audit_control')
    b.put('assets/index.json', {'assets': list(sorted(b.assets.values(), key = lambda r: r['path'])),
                              'sha_scope': 'recorded byte hashes; export does not rehash all payloads', 'missing_snapshots': missing_snapshots})
    code = []
    for directory in ('src', 'scripts', 'configs', 'tests', 'strategies/specs'):
        for path in sorted(safe(b.root, directory).rglob('*')):
            if path.is_file() and path.suffix in ('.py', '.yaml', '.json') and '__pycache__' not in path.parts:
                code.append(b.reference(path.relative_to(b.root).as_posix()))
    for name in ('pyproject.toml', 'uv.lock'): code.append(b.reference(name))
    b.put('evidence/implementation-files.json', code)
    project = {'schema': 1, 'goal': policy['goal'], 'accepted': accepted, 'current': latest,
               'accepted_receipt': b.reference(policy['accepted_receipt']), 'in_progress_acceptance': 'pending' if latest['checkpoint'] != accepted['checkpoint'] else 'passed',
               'published_batch': published['batch_id'], 'catalogs': catalogs, 'snapshots': snapshots, 'financial_imports': imports,
               'indexes': {k: 'data/metadata/' + v for k, v in {'runs': 'runs/index.json', 'representative_runs': 'runs/representative.json', 'evidence': 'evidence/index.json',
                          'tasks': 'tasks/index.json', 'decisions': 'decisions/index.json', 'blockers': 'blockers.json', 'capabilities': 'capabilities.json', 'assets': 'assets/index.json', 'bundles': 'bundles/index.json'}.items()},
               'total_parent_runs': len(b.run_index), 'strategy_parent_runs': len(local_strategies), 'run_kinds': dict(Counter(r['kind'] for r in b.run_index)),
               'metadata_is_runtime_input': False, 'payload_integrity_reaudited': False,
               'publication': policy['publication']}
    b.put('README.md', render(project, policy, metadata = True).encode(), raw = True)
    b.put('project-state.json', project)
    return b, project, render(project, policy)


def render(project, policy, metadata = False):
    c = project['current']; a = project['accepted']
    prefix = '../../../../' if metadata else '../'
    lines = ['# observe 项目状态与续接', '', f"状态检查点：`{c['checkpoint']}`；正式验收检查点：`{a['checkpoint']}`。", '',
             '本页由元数据导出生成。研究结果与数据载荷分开管理；仅Git文件足以阅读，完整离线复现需要迁移所需输入。', '',
             f"来源登记{c['sources']}；人工审查{c['manually_reviewed']}（正式验收范围{a['manually_reviewed']}）；本机真实回测及匹配复现来源{c['locally_backtested_sources']}；策略父运行{c['local_strategy_runs']}。原策略完整等价复现0。", '',
             '14个代表配置不是14份来源；67个策略父运行含短长窗口、复现和失败。目录“已验证11”与本机回测来源12口径不同，旧声明式来源仍未正式审查。', '',
             f"发布`{project['published_batch']}`。年度财务95,486行、季度405,867行、年度控制变量73,738行；严格历史财务可用0。", '',
             '## 当前结论', '',
             '第63批SVR/价值财务研究已验收，但真实模型/交易回测0；第64批价格形态/K线组件和禁网匹配完成，正式检查/冻结终审待做。', '',
             'EP主板组件累计+46.97%、回撤-16.65%；伊利/招行累计+1766.31%、回撤-48.76%。固定池、平台语义及非独立留出限制保留。', '',
             '茅台布林累计收益高，但2023年至今各年亏损；三进兵近期也有连续亏损，滑点加倍累计收益从585.07%降至387.22%。国航和复星2万元版本为负面结果；复星100万元低回撤对应99.44%平均现金。', '',
             'BP长区间被退市/换股结算阻断，收益不可用。旧滚动均值三实验已标为数值验收失效并关联稳定修正版。', '',
             '分钟成对研究没有可确认的模型层增量，组合阻断与退市分钟缺失限制保留。不能把组件或合成诊断计作交易回测。', '',
             '## 阅读入口', '',
             f'- [领域术语]({prefix}CONTEXT.md)',
             f'- [技术环境]({prefix}docs/03-技术选型与版本.md)',
             f'- [策略阶段结果]({prefix}docs/tasks/2026-10-08-策略研究阶段汇报.md)',
             f'- [分钟研究]({prefix}docs/06-分钟特征成对实验报告.md)',
             f'- [Git与载荷管理]({prefix}docs/tasks/2026-10-08-Git追踪与跨机数据管理计划.md)', '',
             '## 下一步与阻断', '']
    lines += [f"- `{t['id']}`：{t['action']}；状态`{t['status']}`；完成条件：{t['done_when']}。" for t in policy['tasks']]
    lines += ['', '历史股本/PCF/名称、严格财报版本、退市/换股权益、ETF/盘中/期货专用输入仍缺。广汽2020/12和七股低波2022/24待回复，既有批准变体持续有效。', '',
              '## 使用与核验', '',
              '只读核验元数据：', '', '```bash', 'python scripts/export_research_metadata.py verify --root .', '```', '',
              '载荷核验需要原始策略、分区和实验；缺失会返回blocked并列明路径。metadata不会替换活跃PUBLISHED或初始化运行数据库。', '',
              'Git上传状态与研究验收状态分开：2026-10-08已推送到私有远端master，本机工作区Git已对齐；凭据见docs/handoff/2026-10-08-git-publication.json，经过见工作区整理与Git对齐记录。', '']
    if metadata:
        lines += ['[权威索引](../../index.json)、[项目状态](project-state.json)、[任务](tasks/index.json)、[决策](decisions/index.json)、[阻断](blockers.json)、[能力](capabilities.json)、[实验索引](runs/index.json)、[代表结果](runs/representative.json)、[证据索引](evidence/index.json)、[载荷](assets/index.json)、[数据包](bundles/index.json)。', '']
    else: lines += ['[权威索引](../data/metadata/index.json)；按其indexes字段追读具体对象。', '']
    return '\n'.join(lines)


def publish(builder, project, page):
    from filelock import FileLock
    root = builder.root; base = safe(root, 'data/metadata'); base.mkdir(parents = True, exist_ok = True)
    with FileLock(str(safe(root, 'data/metadata/.export.lock'))):
        for name, sha in builder.inputs.items():
            if file_sha(safe(root, name)) != sha: raise ValueError(f'Input changed during export: {name}')
        state_id = digest(encoded({name: digest(data) for name, data in sorted(builder.files.items())}))[:24]
        destinations = {f'data/metadata/{name}': f'data/metadata/states/{state_id}/{name}' for name in builder.files if name not in builder.originals}

        def relocate(value):
            if isinstance(value, str): return destinations.get(value, value)
            if isinstance(value, list): return [relocate(v) for v in value]
            if isinstance(value, dict): return {k: relocate(v) for k, v in value.items()}
            return value

        files = {}
        for name, data in builder.files.items():
            target = destinations.get('data/metadata/' + name, 'data/metadata/' + name)
            if name not in builder.originals and name.endswith('.json'): data = encoded(relocate(json.loads(data)))
            if len(data) > MAX_FILE: raise ValueError(f'Metadata file exceeds 5 MiB: {target}')
            files[target] = data
        files['data/metadata/README.md'] = ('# observe 研究元数据\n\n'
            '从[权威索引](index.json)的project字段读取当前状态，indexes字段进入对应任务、决策、实验和证据。\n\n'
            '人读入口：[当前项目状态](../../docs/10-项目状态与续接.md)。完整载荷不在Git；元数据不作为Store运行输入。\n').encode()
        records = [{'path': name, 'bytes': len(data), 'sha256': digest(data)} for name, data in sorted(files.items())]
        total = sum(r['bytes'] for r in records)
        if total > MAX_EXPORT:
            largest = sorted(records, key = lambda r: r['bytes'], reverse = True)[:8]
            raise ValueError(f'Metadata export exceeds 20 MiB: {total} bytes; largest={largest}')
        manifest = {'schema': 1, 'files': records, 'source_bindings': [{'path': p, 'sha256': s} for p, s in sorted(builder.inputs.items())],
                    'bytes': total, 'copy_policy': 'original copies retain bytes; derived summaries normalize nonfinite JSON values to null'}
        identity = digest(encoded(manifest))[:24]; manifest_name = f'data/metadata/exports/{identity}/manifest.json'
        for name, data in files.items():
            target = safe(root, name)
            if target.exists() and target.read_bytes() != data: raise ValueError(f'Immutable metadata conflict: {name}')
        for name, data in sorted(files.items()):
            target = safe(root, name)
            if not target.exists():
                target.parent.mkdir(parents = True, exist_ok = True)
                install(target, data)
        target = safe(root, manifest_name); target.parent.mkdir(parents = True, exist_ok = True); data = encoded(manifest)
        if target.exists() and target.read_bytes() != data: raise ValueError('Manifest identity collision')
        if not target.exists():
            install(target, data)
        atomic(safe(root, 'docs/10-项目状态与续接.md'), page.encode())
        index = {'schema': 1, 'export_id': identity, 'manifest': manifest_name, 'manifest_sha256': digest(data),
                 'project': destinations['data/metadata/project-state.json'], 'current_checkpoint': project['current']['checkpoint'], 'accepted_checkpoint': project['accepted']['checkpoint'],
                 'implementation_files': destinations['data/metadata/evidence/implementation-files.json'],
                 'indexes': relocate(project['indexes']), 'payload_verified': False}
        atomic(safe(root, 'data/metadata/index.json'), encoded(index))
        return {'status': 'ok', 'export_id': identity, 'files': len(records), 'bytes': total, 'index': 'data/metadata/index.json'}


def atomic(path, data):
    path.parent.mkdir(parents = True, exist_ok = True)
    fd, name = tempfile.mkstemp(prefix = '.metadata-', suffix = '.tmp', dir = path.parent)
    try:
        with os.fdopen(fd, 'wb') as stream: stream.write(data); stream.flush(); os.fsync(stream.fileno())
        os.replace(name, path)
    finally:
        if os.path.exists(name): os.unlink(name)


def install(path, data):
    """Publish complete immutable bytes without replacing an existing object."""
    fd, name = tempfile.mkstemp(prefix = '.metadata-', suffix = '.tmp', dir = path.parent)
    try:
        with os.fdopen(fd, 'wb') as stream: stream.write(data); stream.flush(); os.fsync(stream.fileno())
        os.link(name, path)
    finally:
        if os.path.exists(name): os.unlink(name)


def verify(root, payload = False, run_id = None):
    root = Path(root).resolve(); index = json.loads(safe(root, 'data/metadata/index.json').read_bytes())
    path = safe(root, index['manifest']); data = path.read_bytes()
    if digest(data) != index['manifest_sha256']: raise ValueError('Export manifest SHA mismatch')
    manifest = json.loads(data); issues = []
    if digest(encoded(manifest))[:24] != index['export_id']: raise ValueError('Export identity mismatch')
    records = {r['path']: r for r in manifest['files']}
    if len(records) != len(manifest['files']): raise ValueError('Duplicate manifest path')
    if manifest['bytes'] != sum(r['bytes'] for r in records.values()) or manifest['bytes'] > MAX_EXPORT: raise ValueError('Invalid export size')
    for name in [index['project'], index['implementation_files'], *index['indexes'].values()]:
        if name not in records: raise ValueError(f'Index reference outside manifest: {name}')
    for r in manifest['files']:
        path = safe(root, r['path'])
        if not path.is_file(): issues.append({'path': r['path'], 'problem': 'missing_metadata'})
        elif path.stat().st_size != r['bytes'] or file_sha(path) != r['sha256']: issues.append({'path': r['path'], 'problem': 'metadata_sha_mismatch'})
    if issues: return {'status': 'error', 'scope': 'metadata', 'issues': issues}
    project = json.loads(safe(root, index['project']).read_bytes())
    if (project['indexes'] != index['indexes'] or project['current']['checkpoint'] != index['current_checkpoint']
            or project['accepted']['checkpoint'] != index['accepted_checkpoint']): raise ValueError('Index/project state mismatch')
    for binding in manifest['source_bindings']:
        if not binding['path'].startswith(('data/', 'repo/')):
            path = safe(root, binding['path'])
            if not path.is_file() or file_sha(path) != binding['sha256']: issues.append({'path': binding['path'], 'problem': 'source_document_missing_or_changed'})
    implementations = json.loads(safe(root, index['implementation_files']).read_bytes())
    for r in implementations:
        path = safe(root, r['path'])
        if not path.is_file() or file_sha(path) != r['sha256']: issues.append({'path': r['path'], 'problem': 'implementation_missing_or_changed'})
    if issues: return {'status': 'error', 'scope': 'metadata', 'issues': issues}
    if not payload: return {'status': 'ok', 'scope': 'metadata_only', 'payload_verified': False, 'checked': len(manifest['files']) + len(implementations), 'issues': []}
    asset_document = json.loads(safe(root, index['indexes']['assets']).read_bytes())
    assets = asset_document['assets']
    if run_id:
        runs = json.loads(safe(root, index['indexes']['runs']).read_bytes()); row = next((r for r in runs if r['run_id'] == run_id), None)
        if row is None: raise ValueError(f'Unknown run: {run_id}')
        dm = safe(root, f'data/runs/{run_id}/data_manifest.json')
        if not dm.is_file(): return {'status': 'blocked', 'scope': 'payload', 'payload_verified': False, 'issues': [{'path': str(dm.relative_to(root)), 'problem': 'missing_payload'}]}
        frozen = {a['path']: a for a in assets}
        config_path = f'data/runs/{run_id}/config.json'
        for name in (dm.relative_to(root).as_posix(), config_path):
            path = safe(root, name); expected = frozen.get(name, {}).get('file_sha256')
            if not path.is_file() or not expected or file_sha(path) != expected:
                issues.append({'path': name, 'problem': 'frozen_input_missing_or_changed'})
        if issues: return {'status': 'blocked', 'scope': 'payload', 'payload_verified': False, 'issues': issues}
        doc = json.loads(dm.read_bytes()); paths = {f'data/runs/{run_id}/manifest.json'}
        for parts in doc.get('used', {}).values(): paths.update('data/' + e['file'] for e in parts.values())
        paths.update(a['path'] for a in assets if a['path'].startswith(f'data/runs/{run_id}/'))
        config = json.loads(safe(root, config_path).read_bytes()); source = config.get('source', {})
        if config.get('snapshot_id'): paths.add(f"data/snapshots/{config['snapshot_id']}.json")
        paths.update(a['path'] for a in assets if a.get('kind') == 'audit_control')
        if source.get('path'):
            name = source['path'] if source['path'].startswith('repo/') else 'repo/量化策略源代码/' + source['path']
            paths.add(name)
        assets = [a for a in assets if a['path'] in paths]
        if paths - {a['path'] for a in assets}: issues += [{'path': p, 'problem': 'missing_asset_record'} for p in sorted(paths - {a['path'] for a in assets})]
    else:
        issues += [{'path': f'data/snapshots/{sid}.json', 'problem': 'missing_snapshot'} for sid in asset_document['missing_snapshots']]
    for asset in assets:
        path = safe(root, asset['path'])
        if not path.is_file(): issues.append({'path': asset['path'], 'problem': 'missing_payload'})
        elif not asset['file_sha256']: issues.append({'path': asset['path'], 'problem': 'byte_sha_not_recorded'})
        elif file_sha(path) != asset['file_sha256']: issues.append({'path': asset['path'], 'problem': 'payload_sha_mismatch'})
    return {'status': 'blocked' if issues else 'ok', 'scope': 'payload', 'run_id': run_id, 'payload_verified': not issues,
            'payload_state': 'payload_partial' if issues else 'payload_verified', 'checked': len(assets), 'issues': issues}


def main():
    parser = argparse.ArgumentParser(description = __doc__); sub = parser.add_subparsers(dest = 'action', required = True)
    exp = sub.add_parser('export'); exp.add_argument('--root', default = '.')
    check = sub.add_parser('verify'); check.add_argument('--root', default = '.'); check.add_argument('--payload', action = 'store_true'); check.add_argument('--run-id')
    args = parser.parse_args()
    if args.action == 'export': result = publish(*prepare(args.root))
    else: result = verify(args.root, args.payload, args.run_id)
    print(json.dumps(result, ensure_ascii = False, indent = 2))
    return 0 if result['status'] == 'ok' else 3 if result['status'] == 'blocked' else 1


if __name__ == '__main__': raise SystemExit(main())
