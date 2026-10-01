"""命令行入口：observe [--root data] <data|jobs|serve> ...（模块 19）。与 Web 接口调用同一组函数、同一把锁。"""
import argparse, json, sys, traceback
from pathlib import Path


def _json(x): print(json.dumps(x, ensure_ascii = False, indent = 1, default = str))


def _config(path):
    if not path: return {}
    import yaml
    data = yaml.safe_load(Path(path).read_text(encoding='utf-8')) or {}
    if not isinstance(data, dict): raise ValueError('run 配置必须是对象')
    return data


def job_status(kind, result):
    """任务结果 → 队列状态：回放与复现直接沿用运行状态（与 status.json、函数返回值、命令退出码同一定义）"""
    if kind in ('run_experiment', 'reproduce', 'research', 'paired', 'experiment', 'backtest_variant', 'factor_eval'): return result['status']
    if isinstance(result, dict) and result.get('status') == 'refresh_failed_no_data': return 'failed'
    return 'partial' if isinstance(result, dict) and result.get('status') in ('rejected', 'published_partial', 'refresh_failed') else 'success'


def run_kind(root, kind, params):
    """任务种类 → 研究核心函数；命令行直接执行和队列执行都走这里"""
    from .data.store import Store
    if kind == 'data_update':
        from .data import standardize as std
        from .data.sources.tdx import Tdx
        from .data.update import update_daily
        codes = params.get('factors') or []
        if params.get('factors_all'): codes = Store(root).load('instruments').query("kind == 'stock'").instrument.tolist()
        # update_daily is the public writer service and owns the complete
        # DATA_WRITER scope.  The adapter is constructed here only; network
        # initialization remains inside update_daily's protected call chain.
        tdx = None if params.get('no_actions') else Tdx(root)
        try: return update_daily(root, params['start'], params['end'], tdx = tdx, factor_codes = [std.to_baostock(c) for c in codes], force = params.get('force', False))
        finally:
            if tdx: tdx.close()
    if kind == 'minute_import':
        from .data.minute import import_minute
        return import_minute(root, params['source'], params.get('start'), params.get('end'), top = int(params.get('top', 800)), workers = int(params.get('workers', 4)), sevenzip = params.get('sevenzip'),
                             log = lambda m: print(m, file = sys.stderr, flush = True))
    if kind == 'snapshot':
        from .data.locks import DATA_WRITER, operation_lock
        with operation_lock(root, DATA_WRITER): return {'snapshot_id': Store(root).snapshot(params.get('note', ''))}
    if kind == 'gc':
        from .data.locks import DATA_WRITER, operation_lock
        with operation_lock(root, DATA_WRITER): return {'files': Store(root).gc(apply = params.get('apply', False)), 'applied': params.get('apply', False)}
    if kind == 'data_audit':
        from .data.audit import audit_daily
        from .data.update import default_rules
        from .data.locks import DATA_WRITER, operation_lock
        st = Store(root)
        import secrets
        with operation_lock(root, DATA_WRITER):
            state = st.published(); batch_id = state['batch_id']
            if not batch_id: raise RuntimeError('还没有已发布的数据')
            pin = f'audit-running-{__import__("os").getpid()}-{secrets.token_hex(4)}'
            st.pin_state(pin, state)
        try:
            b = st.load_state(state, 'bars_1d'); cal = st.load_state(state, 'calendar'); inst = st.load_state(state, 'instruments')
            days = sorted(set(cal[cal.is_open].date)) if len(cal) else sorted(b.date.unique())
            rules = default_rules(); iss = audit_daily(b, days, inst, rules)
            rule_fingerprint = rules.config_fingerprint() if rules is not None else 'builtin-v1'
            aid = st.commit_audit(batch_id, iss, rule_fingerprint, {'start': str(min(days)) if days else None, 'end': str(max(days)) if days else None, 'days': len(days), 'rows': len(b)}, scope='snapshot', input_state=state)
        finally:
            st.unpin(pin)
        return {'rows': len(b), 'days': len(days), 'issues': {f'{l}/{r}': int(n) for (l, r), n in iss.groupby(['level', 'rule']).size().items()} if len(iss) else {}, 'audit_id': aid, 'batch_id': batch_id}
    if kind == 'run_experiment':
        from .replay import _run, run_params
        p = run_params(params); return _run(root, p['config'], p['output'])
    if kind == 'research':
        from .research import _research, research_params
        p = research_params(params); return _research(root, p['config'], p['output'])
    if kind == 'paired':
        from .paired import _paired, paired_params
        p = paired_params(params); return _paired(root, p['config'], p['output'])
    if kind == 'experiment':
        from .experiments import run_experiment
        return run_experiment(root, **params)
    if kind == 'backtest_variant':
        from .experiments import run_variant
        return run_variant(root, **params)
    if kind == 'factor_eval':
        from .experiments import run_factor_eval
        return run_factor_eval(root, **params)
    if kind == 'reproduce':
        from .replay import reproduce
        unknown = set(params) - {'run', 'output', 'abs_tol', 'rel_tol'}
        if unknown: raise ValueError(f'reproduce 不认识的参数：{sorted(unknown)}')
        return reproduce(root, params['run'], params.get('output'), **{k: float(params[k]) for k in ('abs_tol', 'rel_tol') if params.get(k) is not None})
    raise NotImplementedError(f'任务种类 {kind} 尚未实现')


def main(argv = None):
    ap = argparse.ArgumentParser(prog = 'observe'); ap.add_argument('--root', default = 'data', help = '数据目录'); sub = ap.add_subparsers(dest = 'cmd', required = True)
    d = sub.add_parser('data').add_subparsers(dest = 'act', required = True)
    u = d.add_parser('update', help = '下载并发布 [start, end] 的日历、证券资料、全市场日线与复权因子变动')
    u.add_argument('--start', required = True); u.add_argument('--end', required = True); u.add_argument('--factors', default = '', help = '逗号分隔，取全部复权因子历史的证券，如 600519.SH')
    u.add_argument('--factors-all', action = 'store_true'); u.add_argument('--no-actions', action = 'store_true'); u.add_argument('--force', action = 'store_true'); u.add_argument('--queue', action = 'store_true', help = '提交到任务队列而不是直接执行')
    m = d.add_parser('import-minute', help = '导入外部 1 分钟线压缩包：校验、与日线对账、合成 5 分钟，写入 bars_5m 并发布（决策 17）')
    m.add_argument('--source', required = True, help = '目录/年/月/YYYYMMDD.zip 的根目录'); m.add_argument('--start'); m.add_argument('--end'); m.add_argument('--top', type = int, default = 800)
    m.add_argument('--workers', type = int, default = 4); m.add_argument('--sevenzip', help = '7z.exe 路径（个别日期是 7z 格式）'); m.add_argument('--queue', action = 'store_true')
    s = d.add_parser('snapshot'); s.add_argument('--note', default = '')
    g = d.add_parser('gc'); g.add_argument('--apply', action = 'store_true', help = '真正删除（默认只列出）')
    d.add_parser('status'); d.add_parser('audit', help = '按当前执行规则重新审计已发布的日线')
    j = sub.add_parser('jobs').add_subparsers(dest = 'act', required = True)
    js = j.add_parser('submit'); js.add_argument('kind'); js.add_argument('--params', default = '{}')
    j.add_parser('list'); w = j.add_parser('worker'); w.add_argument('--once', action = 'store_true')
    for name in ('exec', 'retry', 'cancel', 'show'): j.add_parser(name).add_argument('job_id')
    sv = sub.add_parser('serve'); sv.add_argument('--port', type = int, default = 8765)
    r = sub.add_parser('run', help = '离线回放：显式参数 > --config YAML > 默认值；退出码 0 成功 / 3 阻断 / 1 出错')
    r.add_argument('--config'); r.add_argument('--snapshot'); r.add_argument('--output'); r.add_argument('--cash', type = float); r.add_argument('--start'); r.add_argument('--end')
    rs = sub.add_parser('research', help = '研究流水线：股票池 → 因子 → 标签 → 切分 → 基线与 Ridge 样本外预测 → 评价；退出码同 run')
    rs.add_argument('--config'); rs.add_argument('--snapshot'); rs.add_argument('--output')
    pr = sub.add_parser('paired', help = '成对对照实验：分钟股票池内「日频基础因子」对「日频 + 分钟特征」，两侧样本外预测与账本回测的成对比较；退出码同 run')
    pr.add_argument('--config'); pr.add_argument('--snapshot'); pr.add_argument('--output')
    re_ = sub.add_parser('reevaluate', help = '只读重新评价已有成对实验（评价口径修正后）：写到新目录，引用原实验编号与产物哈希，不改动源目录')
    re_.add_argument('--source', action = 'append', required = True, metavar = '标签=成对目录', help = '可重复；第一个是主实验，例如 main=data/runs/<id> sensitivity=data/runs/<id>'); re_.add_argument('--output')
    dg = sub.add_parser('diagnose-minute', help = '只读诊断成对实验：缺失模式、打分变化分解（拟合扰动 / 分钟因子数值）、集中持仓贡献；写到新目录，不改动源目录')
    dg.add_argument('--source', required = True, help = '成对实验目录'); dg.add_argument('--source-check', help = 'scripts/check_missing_minute_source.py 输出的核验文件'); dg.add_argument('--output')
    rp = sub.add_parser('reproduce', help = '用冻结输入在新目录重跑并逐表比较；退出码 0 一致 / 2 不一致 / 3 阻断 / 1 出错或拒绝')
    rp.add_argument('run'); rp.add_argument('--output'); rp.add_argument('--abs-tol', type = float); rp.add_argument('--rel-tol', type = float)
    ex = sub.add_parser('experiment', help = '完整研究实验：冻结快照 → 四模型预测 → 三层评价 → 组合变体与成本情景')
    ex.add_argument('--config'); ex.add_argument('--snapshot'); ex.add_argument('--output'); ex.add_argument('--queue', action = 'store_true')
    vr = sub.add_parser('variant', help = '复用已有研究预测，仅重跑组合与账本；产物放入父实验 variants 子目录')
    vr.add_argument('parent'); vr.add_argument('--config'); vr.add_argument('--model'); vr.add_argument('--output'); vr.add_argument('--queue', action = 'store_true')
    co = sub.add_parser('compare', help = '并列读取已有实验的指标'); co.add_argument('runs', nargs = '+')
    rr = sub.add_parser('runs').add_subparsers(dest = 'act', required = True)
    rl = rr.add_parser('list'); rl.add_argument('--kind'); rl.add_argument('--status'); rl.add_argument('--limit', type = int, default = 50)
    rr.add_parser('index'); rr.add_parser('show').add_argument('run')
    fa = sub.add_parser('factor').add_subparsers(dest = 'act', required = True)
    fa.add_parser('list'); fa.add_parser('validate').add_argument('expr')
    fe = fa.add_parser('eval', help = '只读已保存研究产物，计算完整因子诊断；不重新训练')
    fe.add_argument('--config'); fe.add_argument('--run'); fe.add_argument('--output'); fe.add_argument('--queue', action = 'store_true')
    a = ap.parse_args(argv); root = Path(a.root)
    from .jobs import Jobs, SUPPORTED
    if a.cmd == 'experiment':
        from .experiments import experiment_params
        from .runs import EXIT_CODES
        p = experiment_params(_config(a.config), snapshot = a.snapshot, output = a.output)
        params = {**p['config'].model_dump(mode = 'json'), 'output': p['output']}
        if a.queue: return _json({'job_id': Jobs(root).submit('experiment', params)})
        r = run_kind(root, 'experiment', params); _json(r)
        if EXIT_CODES[r['status']]: sys.exit(EXIT_CODES[r['status']])
        return
    if a.cmd == 'variant':
        from .runs import EXIT_CODES
        cfg = _config(a.config); configured_output = cfg.pop('output', None); output = a.output or configured_output
        if a.model: cfg['model'] = a.model
        p = {'parent': a.parent, 'config': cfg, 'output': output}
        if a.queue: return _json({'job_id': Jobs(root).submit('backtest_variant', p)})
        r = run_kind(root, 'backtest_variant', p); _json(r)
        if EXIT_CODES[r['status']]: sys.exit(EXIT_CODES[r['status']])
        return
    if a.cmd == 'factor':
        from .features import load_factor_set
        from .runs import EXIT_CODES, REPO
        if a.act == 'list': return _json(load_factor_set(REPO / 'configs/factor_sets/daily_basic_v1.yaml')['factors'])
        if a.act == 'validate':
            from .factors import parse
            p = parse(a.expr); return _json({'expr': a.expr, 'fields': sorted(p.fields), 'lookback': p.lookback})
        params = {**_config(a.config), **{k: v for k, v in {'run': a.run, 'output': a.output}.items() if v is not None}}
        if a.queue: return _json({'job_id': Jobs(root).submit('factor_eval', params)})
        r = run_kind(root, 'factor_eval', params); _json(r)
        if EXIT_CODES[r['status']]: sys.exit(EXIT_CODES[r['status']])
        return
    if a.cmd == 'runs':
        from .runs import RunRegistry
        from .artifacts import run_detail
        registry = RunRegistry(root / 'runs')
        if a.act == 'show': return _json(run_detail(root, a.run))
        count = registry.index()
        return _json({'indexed': count}) if a.act == 'index' else _json(registry.list(a.limit, kind = a.kind, status = a.status))
    if a.cmd == 'compare':
        from .artifacts import compare_runs
        return _json(compare_runs(root, a.runs))
    if a.cmd == 'data':
        from .data.store import Store
        if a.act == 'update':
            p = {'start': a.start, 'end': a.end, 'factors': [x for x in a.factors.split(',') if x], 'factors_all': a.factors_all, 'no_actions': a.no_actions, 'force': a.force}
            return _json({'job_id': Jobs(root).submit('data_update', p)} if a.queue else run_kind(root, 'data_update', p))
        if a.act == 'import-minute':
            p = {'source': a.source, 'start': a.start, 'end': a.end, 'top': a.top, 'workers': a.workers, 'sevenzip': a.sevenzip}
            if a.queue: return _json({'job_id': Jobs(root).submit('minute_import', p)})
            r = run_kind(root, 'minute_import', p); _json(r)
            if r.get('status') == 'refresh_failed_no_data': sys.exit(1)      # 没有任何可用数据：退出码非零
            return
        if a.act == 'snapshot': return _json(run_kind(root, 'snapshot', {'note': a.note}))
        if a.act == 'gc': return _json(run_kind(root, 'gc', {'apply': a.apply}))
        if a.act == 'audit': return _json(run_kind(root, 'data_audit', {}))
        if a.act == 'status':
            pub = Store(root).published(); return _json({'batch_id': pub['batch_id'], 'tables': {t: {'partitions': len(v), 'rows': sum(x['rows'] for x in v.values())} for t, v in pub['tables'].items()}})
    if a.cmd == 'jobs':
        q = Jobs(root)
        if a.act == 'submit':
            if a.kind not in SUPPORTED: raise SystemExit(f'任务种类 {a.kind} 尚未实现')
            return print(q.submit(a.kind, json.loads(a.params)))
        if a.act == 'list': return _json([{k: r[k] for k in ('job_id', 'kind', 'status', 'created_at', 'finished_at', 'error')} for r in q.list()])
        if a.act == 'worker': return q.worker(once = a.once)
        if a.act == 'retry': return print(q.retry(a.job_id))
        if a.act == 'cancel': return print('cancelled' if q.cancel(a.job_id) else '只能取消排队中的任务')
        if a.act == 'show': return _json(q.get(a.job_id))
        if a.act == 'exec':
            job = q.get(a.job_id)
            try:
                r = run_kind(root, job['kind'], json.loads(job['params']))
                q.finish(a.job_id, job_status(job['kind'], r), r); _json(r)
            except Exception as e:   # noqa: BLE001  任务失败要落盘，不能让子进程静默退出
                traceback.print_exc(); q.finish(a.job_id, 'failed', error = f'{type(e).__name__}: {e}'); sys.exit(1)
            return
    if a.cmd in ('run', 'research', 'paired', 'reproduce'):
        from .runs import EXIT_CODES
        if a.cmd == 'research':
            from .research import _research, research_params
            p = research_params(_config(a.config), snapshot = a.snapshot, output = a.output)
            r = _research(root, p['config'], p['output'])
        elif a.cmd == 'paired':
            from .paired import _paired, paired_params
            p = paired_params(_config(a.config), snapshot = a.snapshot, output = a.output)
            r = _paired(root, p['config'], p['output'])
        elif a.cmd == 'run':
            from .replay import _run, run_params
            p = run_params(_config(a.config), snapshot = a.snapshot, output = a.output, initial_cash = a.cash, start = a.start, end = a.end)
            r = _run(root, p['config'], p['output'])
        else: r = run_kind(root, 'reproduce', {'run': a.run, 'output': a.output, 'abs_tol': a.abs_tol, 'rel_tol': a.rel_tol})
        _json(r)
        if EXIT_CODES[r['status']]: sys.exit(EXIT_CODES[r['status']])
        return
    if a.cmd == 'diagnose-minute':
        from .diagnose import diagnose
        return _json(diagnose(root, a.source, a.source_check, a.output))
    if a.cmd == 'reevaluate':
        from .reeval import reevaluate
        return _json(reevaluate(root, dict(x.split('=', 1) for x in a.source), a.output))
    if a.cmd == 'serve':
        import threading, uvicorn
        from .api.app import create_app
        threading.Thread(target = Jobs(root).worker, daemon = True).start()   # 同进程内的工作线程：网页提交的任务无需另开命令行执行
        uvicorn.run(create_app(root), host = '127.0.0.1', port = a.port)


if __name__ == '__main__': main()
