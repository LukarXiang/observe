"""只读环境与研究预检：不下载、不修复、不建实验、不执行训练或账本。"""
import importlib.metadata
import json
from pathlib import Path
import platform
import re
import shutil
import sys
import tomllib

REPO = Path(__file__).resolve().parents[2]
EXIT_CODES = {'ok': 0, 'warning': 0, 'error': 1}


def doctor(root, snapshot = None, config = None, verify_files = False):
    root = Path(root).resolve(); checks = []
    report = {'root': str(root), 'snapshot_id': snapshot, 'batch_id': None, 'verify_files': verify_files,
              'checks': checks, 'tables': {}, 'plan': None,
              'scope': '环境、配置、快照分区与日期切分预检；不验证逐证券因子/标签可用性、模型表现或账本可执行性'}

    def add(name, status, detail, **data): checks.append({'name': name, 'status': status, 'detail': detail, **data})

    def finish():
        report['summary'] = {s: sum(c['status'] == s for c in checks) for s in ('ok', 'warning', 'error')}
        report['status'] = 'error' if report['summary']['error'] else 'warning' if report['summary']['warning'] else 'ok'
        return report

    def attempt(name, fn):
        try: return fn()
        except Exception as exc:
            add(name, 'error', f'{type(exc).__name__}: {exc}')
            return None

    add('python', 'ok' if sys.version_info >= (3, 12) else 'error', platform.python_version(), executable = sys.executable, prefix = sys.prefix)
    uv = shutil.which('uv'); add('uv', 'ok' if uv else 'warning', uv or '未找到 uv；运行环境重建需要 uv')
    project = tomllib.loads((REPO / 'pyproject.toml').read_text(encoding = 'utf-8'))['project']
    versions = {}
    for spec in project['dependencies']:
        name = re.split(r'[<>=!~\[]', spec)[0]
        try: versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError: versions[name] = None
    missing = [k for k, v in versions.items() if v is None]
    add('dependencies', 'error' if missing else 'ok', f'缺少依赖：{missing}' if missing else '核心依赖已安装（版本列示，不代替锁文件同步）', versions = versions)
    if missing: return finish()
    # 安装元数据存在不等于二进制扩展能加载；错误也必须返回结构化报告。
    failed_imports = []
    for name in versions:
        module = {'scikit-learn': 'sklearn', 'pyyaml': 'yaml'}.get(name, name)
        try: importlib.import_module(module)
        except Exception as exc: failed_imports.append({'package': name, 'error': f'{type(exc).__name__}: {exc}'})
    add('runtime_imports', 'error' if failed_imports else 'ok', '核心依赖实际导入检查', failures = failed_imports)
    if failed_imports: return finish()

    import duckdb
    import numpy as np
    import pandas as pd
    import pyarrow.parquet as pq
    import yaml
    from .data.audit import audit_status
    from .data.standardize import flag
    from .data.store import Store, KEYS, fingerprint
    from .dataset import plan_splits
    from .experiments import ExperimentConfig
    from .execution import sessions
    from .factors.intraday import INTRADAY_FIELDS
    from .features import load_factor_set
    from .ledger.rules import RuleSet
    from .runs import canonical, file_sha

    def configuration():
        raw = yaml.safe_load(Path(config).read_text(encoding = 'utf-8')) if isinstance(config, (str, Path)) else config
        if raw is None: raw = {}
        if not isinstance(raw, dict): raise ValueError('配置必须是对象')
        raw = dict(raw); raw.pop('output', None)
        if snapshot is not None: raw['snapshot'] = snapshot
        raw.setdefault('snapshot', '__published__')
        raw.setdefault('factor_set', str(REPO / 'configs/factor_sets/daily_basic_v1.yaml'))
        raw.setdefault('rules', str(REPO / 'configs/rule_profiles/main_board.yaml'))
        cfg = ExperimentConfig.model_validate(raw)
        if cfg.start and cfg.end and cfg.start > cfg.end: raise ValueError('start 不能晚于 end')
        add('config', 'ok', '已通过完整实验配置校验')
        return cfg

    cfg = attempt('config', configuration)
    selected = cfg.snapshot if cfg else snapshot
    selected = None if selected == '__published__' else selected
    report['snapshot_id'] = selected
    if selected is not None and (not isinstance(selected, str) or not re.fullmatch(r'[\w-]+', selected)):
        add('snapshot', 'error', '非法快照编号'); return finish()
    store = Store(root); state_path = root / 'snapshots' / f'{selected}.json' if selected else store.published_path

    def capture():
        if not state_path.is_relative_to(root) or not state_path.resolve().is_relative_to(root): raise ValueError('快照路径超出数据根目录')
        before = file_sha(state_path); state = json.loads(state_path.read_text(encoding = 'utf-8'))
        if not isinstance(state, dict) or not state.get('batch_id') or not isinstance(state.get('tables'), dict): raise ValueError('数据状态缺少 batch_id 或 tables')
        if selected and state.get('snapshot_id') != selected: raise ValueError('快照文件名与 snapshot_id 不一致')
        add('snapshot', 'ok' if selected else 'warning', '固定快照已读取' if selected else '正在检查当前发布状态；正式实验需显式冻结快照')
        return state, before

    captured = attempt('snapshot', capture)
    if captured is None: return finish()
    state, before = captured; report['batch_id'] = state['batch_id']; usable = {}
    required_columns = {'calendar': {'date', 'is_open'}, 'instruments': {'instrument', 'kind', 'list_date'},
                        'bars_1d': {'date', 'instrument', 'open', 'high', 'low', 'close', 'preclose', 'volume', 'amount', 'is_trading', 'is_st', 'board'}}
    for table, entries in state['tables'].items():
        def partitions():
            if not isinstance(entries, dict): raise ValueError(f'{table} 分区清单不是对象')
            files, rows, failures, restored = [], 0, [], []
            for part, entry in entries.items():
                try:
                    path = (root / entry['file']).resolve()
                    if not path.is_relative_to(root): raise ValueError(f'{table}/{part} 文件路径超出数据根目录')
                    with pq.ParquetFile(path) as parquet:
                        n = parquet.metadata.num_rows; columns = set(parquet.schema_arrow.names)
                    if n != entry['rows']: raise ValueError(f'{table}/{part} 行数与清单不符：{n} != {entry["rows"]}')
                    required = required_columns.get(table, set(KEYS.get(table, ())))
                    if not required <= columns: raise ValueError(f'{table}/{part} 缺少字段 {sorted(required - columns)}')
                    if verify_files:
                        frame = pd.read_parquet(path); actual = fingerprint(frame)
                        if actual != entry['sha']:
                            # 历史 Store 在序列化之前算指纹；秒精度被 Parquet 无损提升为毫秒。
                            # 仅允许毫秒→秒→毫秒逐值一致的还原，并要求完整表指纹精确命中旧值。
                            candidate = frame.copy(); columns = []
                            for name in frame.select_dtypes(include = ['datetime', 'datetimetz']).columns:
                                if frame[name].dt.unit != 'ms': continue
                                seconds = frame[name].dt.as_unit('s')
                                if seconds.dt.as_unit('ms').equals(frame[name]): candidate[name] = seconds; columns.append(name)
                            if not columns or fingerprint(candidate) != entry['sha']:
                                raise ValueError(f'{table}/{part} 内容指纹与清单不符：expected={entry["sha"]}, actual={actual}；需核对内容或历史指纹口径')
                            restored.append({'partition': part, 'columns': columns, 'stored_unit': 'ms', 'original_unit': 's'})
                        keys = KEYS.get(table)
                        if keys and (frame[keys].isna().any().any() or frame.duplicated(keys).any()): raise ValueError(f'{table}/{part} 主键为空或重复')
                    files.append(str(path)); rows += n
                except Exception as exc:
                    failures.append({'partition': part, 'error': f'{type(exc).__name__}: {exc}'})
            report['tables'][table] = {'partitions': len(entries), 'verified_partitions': len(files), 'rows': rows if not failures else None}
            if failures:
                add(f'partitions:{table}', 'error', f'{len(failures)} 个分区未通过；{failures[0]["error"]}', failures = failures)
                return
            usable[table] = files
            add(f'partitions:{table}', 'warning' if restored else 'ok',
                '历史秒精度无损还原后指纹匹配，数据与清单未修改' if restored else '元数据、内容指纹及分区内主键通过' if verify_files else '文件、表头与行数通过；未检查完整内容指纹',
                partitions = len(files), rows = rows, representation_adjustments = restored)
        attempt(f'partitions:{table}', partitions)
    for table in ('calendar', 'instruments', 'bars_1d', 'adj_coverage'):
        if not usable.get(table) or not report['tables'][table]['rows']: add(f'required:{table}', 'error', f'缺少可用的 {table} 表')
    for table in ('adj_factors', 'corp_actions'):
        if not usable.get(table): add(f'optional:{table}', 'warning', f'缺少 {table}；需确认复权覆盖和公司行动，不能据此推断没有事件')

    def query(table, sql):
        if not usable.get(table): raise ValueError(f'{table} 分区未通过检查，跳过依赖计算')
        with duckdb.connect() as db: return db.execute(sql.replace('{t}', 'read_parquet(?)'), [usable[table]]).df()

    def calendar_check():
        frame = query('calendar', 'select * from {t}'); frame['date'] = pd.to_datetime(frame.date).dt.date
        if frame.date.isna().any() or frame.date.duplicated().any() or frame.is_open.isna().any(): raise ValueError('日历日期为空、重复或开市标记缺失')
        if frame.is_open.astype(str).str.strip().str.lower().isin(('', 'none', 'nan', '<na>', 'nat')).any(): raise ValueError('开市标记为空，不能视为休市')
        frame['is_open'] = frame.is_open.map(flag)
        days = sessions(frame)
        if not days: raise ValueError('日历没有交易日')
        gaps = sorted(set(pd.date_range(frame.date.min(), frame.date.max()).date) - set(frame.date))
        add('calendar', 'warning' if gaps else 'ok', f'自然日日历缺口 {len(gaps)} 个', missing_dates = list(map(str, gaps)), sessions = len(days))
        return frame, days

    calendar = attempt('calendar', calendar_check) if usable.get('calendar') else None
    fset = attempt('factors', lambda: load_factor_set(cfg.factor_set)) if cfg else None
    rules = attempt('rules', lambda: RuleSet.from_yaml(cfg.rules)) if cfg else None
    if fset:
        names = [f['name'] for f in fset['factors']]
        intraday = sorted({field for f in fset['factors'] for field in f['fields']} & set(INTRADAY_FIELDS))
        add('factors', 'error' if cfg.models.baseline_factor not in names or (intraday and not cfg.minute_pool) else 'ok',
            '检查基线因子与分钟股票池约束', count = len(names), baseline_present = cfg.models.baseline_factor in names, intraday_fields = intraday)
    if cfg and cfg.models.lgbm:
        def lgbm_check():
            from .models import lightgbm
            module = lightgbm(); add('lightgbm', 'ok', f'Python 包及原生运行库可加载：{module.__version__}')
        attempt('lightgbm', lgbm_check)

    def audit_check():
        result = audit_status(root, state['batch_id'], rules); rows = result.pop('rows')
        blocking = int((rows.level == 'block').sum()) if 'level' in rows else 0
        add('daily_audit', 'error' if blocking else 'ok' if result['status'] == 'passed' else 'warning',
            '沿用批次绑定的日线审计；本次没有重新审计', audit = result, problems = len(rows), blocking = blocking)
    attempt('daily_audit', audit_check)

    def planning():
        cal, sessions_ = calendar
        have = set(pd.to_datetime(query('bars_1d', 'select distinct date from {t}').date).dt.date)
        if not have: raise ValueError('日线没有数据')
        first, last = min(have), max(have)
        days = [d for d in sessions_ if first <= d <= last]
        missing = sorted(set(days) - have)
        add('daily_coverage', 'error' if missing else 'ok', f'交易日整日缺失 {len(missing)} 个', first = str(first), last = str(last), missing_dates = list(map(str, missing)))
        if cfg.start and cfg.start < cal.date.min() or cfg.end and cfg.end > cal.date.max(): add('requested_range', 'error', '请求区间超出快照日历范围')
        requested = [d for d in sessions_ if (cfg.start is None or d >= cfg.start) and (cfg.end is None or d <= cfg.end)]
        if set(requested) - have: add('requested_bars', 'error', '请求区间存在缺失的交易日日线', missing_dates = sorted(str(d) for d in set(requested) - have))
        warmup = max(cfg.universe.min_listed_sessions, max(f['lookback'] for f in fset['factors']) + 1, cfg.universe.liquidity_window, cfg.universe.suspend_window)
        days = [d for d in days[warmup:] if (cfg.start is None or d >= cfg.start) and (cfg.end is None or d <= cfg.end)]
        if cfg.minute_pool:
            pool = query('minute_universe', 'select distinct year from {t}'); years = sorted(pool.year.astype(int))
            if not years or years != list(range(years[0], years[-1] + 1)): raise ValueError('分钟股票池年份不连续或为空')
            # 与研究入口使用同一截止日逻辑（包含请求结束月份最后一天所在分区）。
            from .research import _minute_inputs
            if not usable.get('bars_5m'): raise ValueError('缺少可用的分钟数据')
            last_minute = _minute_inputs(store, state, cfg, False)['last_day']
            days = [d for d in days if years[0] <= d.year <= years[-1] and d <= last_minute]
            add('minute_sample', 'warning', '分钟股票池缺退市证券，证据级别为探索', last_day = str(last_minute))
        s = cfg.split
        if not days or len(days) <= s.holdout: raise ValueError(f'预热后仅 {len(days)} 个决策日，不足以保留 {s.holdout} 个最终留出日')
        holdout = days[-s.holdout] if s.holdout else None
        plan = plan_splits(days, s.train, s.valid, s.test, holdout)
        if plan.empty: raise ValueError(f'扣除 {s.holdout} 日留出后不足 train + valid + test = {s.train + s.valid + s.test} 个决策日')
        report['plan'] = canonical({'warmup': warmup, 'decision_days': len(days), 'windows': len(plan), 'holdout_start': holdout, 'test_start': plan.test_start.min(), 'test_end': plan.test_end.max()})
        add('split_plan', 'ok', '复用 plan_splits；尚未计算候选、因子和成熟标签', **report['plan'])
        if rules:
            for day in days:
                for board in cfg.universe.boards:
                    for st in (False, True): rules.on(day, board, st)
            add('rules', 'warning' if rules.used_unverified else 'ok', '费用与交易规则覆盖决策区间', unverified = canonical(sorted(rules.used_unverified)))
        if cfg.benchmarks.enabled:
            from .evaluation.benchmarks import price_levels
            index = query('index_1d', 'select * from {t}') if usable.get('index_1d') else pd.DataFrame()
            test_days = [d for d in sessions_ if plan.test_start.min() <= d <= plan.test_end.max()]
            levels, meta = price_levels(index, sessions_, test_days, cfg.benchmarks.index)
            valid = np.isfinite(levels[:-1]) & np.isfinite(levels[1:])
            add('price_benchmark', 'ok' if valid.all() else 'warning', '按预计测试区间及前一交易日检查指数收盘价', metadata = meta, missing_return_days = int((~valid).sum()))
    if cfg and fset and calendar and usable.get('bars_1d'): attempt('research_plan', planning)
    # 并发发布不污染已捕获的计算输入；明确告知调用者报告已不代表最新状态。
    attempt('state_stability', lambda: add('state_stability', 'ok' if file_sha(state_path) == before else 'warning', '检查期间数据状态文件是否变化'))
    return canonical(finish())
