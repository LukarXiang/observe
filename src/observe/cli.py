"""命令行入口：observe [--root data] <data|jobs|serve> ...（模块 19）。与 Web 接口调用同一组函数、同一把锁。"""
import argparse, json, sys, traceback
from pathlib import Path


def _json(x): print(json.dumps(x, ensure_ascii = False, indent = 1, default = str))


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
    if kind == 'snapshot':
        from .data.locks import DATA_WRITER, operation_lock
        with operation_lock(root, DATA_WRITER): return {'snapshot_id': Store(root).snapshot(params.get('note', ''))}
    if kind == 'gc':
        from .data.locks import DATA_WRITER, operation_lock
        with operation_lock(root, DATA_WRITER): return {'files': Store(root).gc(apply = params.get('apply', False)), 'applied': params.get('apply', False)}
    if kind == 'data_audit':
        import hashlib
        from .data.audit import audit_daily
        from .data.update import default_rules
        st = Store(root); state = st.published(); batch_id = state['batch_id']
        if not batch_id: raise RuntimeError('还没有已发布的数据')
        pin = f'audit-running-{__import__("os").getpid()}'
        st.pin_state(pin, state)
        try:
            b = st.load_state(state, 'bars_1d'); cal = st.load_state(state, 'calendar'); inst = st.load_state(state, 'instruments')
            days = sorted(set(cal[cal.is_open].date)) if len(cal) else sorted(b.date.unique())
            iss = audit_daily(b, days, inst, default_rules())
            rules = default_rules(); rule_fingerprint = hashlib.sha256(repr(rules).encode()).hexdigest() if rules is not None else 'builtin'
            aid = st.commit_audit(batch_id, iss, rule_fingerprint, {'start': str(min(days)) if days else None, 'end': str(max(days)) if days else None, 'days': len(days), 'rows': len(b)})
        finally:
            st.unpin(pin)
        return {'rows': len(b), 'days': len(days), 'issues': {f'{l}/{r}': int(n) for (l, r), n in iss.groupby(['level', 'rule']).size().items()} if len(iss) else {}, 'audit_id': aid, 'batch_id': batch_id}
    raise NotImplementedError(f'任务种类 {kind} 尚未实现')


def main(argv = None):
    ap = argparse.ArgumentParser(prog = 'observe'); ap.add_argument('--root', default = 'data', help = '数据目录'); sub = ap.add_subparsers(dest = 'cmd', required = True)
    d = sub.add_parser('data').add_subparsers(dest = 'act', required = True)
    u = d.add_parser('update', help = '下载并发布 [start, end] 的日历、证券资料、全市场日线与复权因子变动')
    u.add_argument('--start', required = True); u.add_argument('--end', required = True); u.add_argument('--factors', default = '', help = '逗号分隔，取全部复权因子历史的证券，如 600519.SH')
    u.add_argument('--factors-all', action = 'store_true'); u.add_argument('--no-actions', action = 'store_true'); u.add_argument('--force', action = 'store_true'); u.add_argument('--queue', action = 'store_true', help = '提交到任务队列而不是直接执行')
    s = d.add_parser('snapshot'); s.add_argument('--note', default = '')
    g = d.add_parser('gc'); g.add_argument('--apply', action = 'store_true', help = '真正删除（默认只列出）')
    d.add_parser('status'); d.add_parser('audit', help = '按当前执行规则重新审计已发布的日线')
    j = sub.add_parser('jobs').add_subparsers(dest = 'act', required = True)
    js = j.add_parser('submit'); js.add_argument('kind'); js.add_argument('--params', default = '{}')
    j.add_parser('list'); w = j.add_parser('worker'); w.add_argument('--once', action = 'store_true')
    for name in ('exec', 'retry', 'cancel', 'show'): j.add_parser(name).add_argument('job_id')
    sv = sub.add_parser('serve'); sv.add_argument('--port', type = int, default = 8765)
    a = ap.parse_args(argv); root = Path(a.root)
    from .jobs import Jobs, SUPPORTED
    if a.cmd == 'data':
        from .data.store import Store
        if a.act == 'update':
            p = {'start': a.start, 'end': a.end, 'factors': [x for x in a.factors.split(',') if x], 'factors_all': a.factors_all, 'no_actions': a.no_actions, 'force': a.force}
            return _json({'job_id': Jobs(root).submit('data_update', p)} if a.queue else run_kind(root, 'data_update', p))
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
                status = 'partial' if isinstance(r, dict) and r.get('status') == 'rejected' else 'success'
                q.finish(a.job_id, status, r); _json(r)
            except Exception as e:   # noqa: BLE001  任务失败要落盘，不能让子进程静默退出
                traceback.print_exc(); q.finish(a.job_id, 'failed', error = f'{type(e).__name__}: {e}'); sys.exit(1)
            return
    if a.cmd == 'serve':
        import threading, uvicorn
        from .api.app import create_app
        threading.Thread(target = Jobs(root).worker, daemon = True).start()   # 同进程内的工作线程：网页提交的任务无需另开命令行执行
        uvicorn.run(create_app(root), host = '127.0.0.1', port = a.port)


if __name__ == '__main__': main()
