import { useEffect, useState } from 'react'
import { api, type Job } from '../api'
import { KIND_NAMES, STATUS_NAMES, time } from '../format'

const KINDS = [['data_update', '数据更新'], ['data_audit', '数据审计'], ['snapshot', '冻结快照'], ['gc', '清理']]
export default function Jobs() {
  const [jobs, setJobs] = useState<Job[]>([]); const [selected, setSelected] = useState<Job | null>(null); const [log, setLog] = useState(''); const [error, setError] = useState(''); const load = () => api.jobs().then(setJobs).catch(e => setError(String(e)))
  useEffect(() => { load(); const t = setInterval(load, 3000); return () => clearInterval(t) }, [])
  useEffect(() => { if (!selected) return; api.log(selected.job_id, 0).then(x => setLog(x.text)).catch(() => {}) }, [selected])
  const submit = (kind: string) => api.submit(kind).then(load).catch(e => setError(String(e)))
  return <section className="page"><header className="page-head"><div><p className="eyebrow">JOBS</p><h1>任务</h1><p className="lede">统一查看任务参数、日志与失败原因</p></div><div className="actions">{KINDS.map(([k, n]) => <button key={k} onClick={() => submit(k)}>{n}</button>)}</div></header>{error && <p className="error">{error}</p>}<div className="jobs"><div className="results">{jobs.map(j => <button className="result" key={j.job_id} onClick={() => setSelected(j)} aria-current={selected?.job_id === j.job_id}><b>{KIND_NAMES[j.kind] ?? j.kind}</b><span>{STATUS_NAMES[j.status] ?? j.status} · {time(j.created_at)}</span></button>)}{!jobs.length && <p className="empty">暂无任务</p>}</div><div>{selected ? <><dl className="kv"><dt>任务</dt><dd className="mono">{selected.job_id}</dd><dt>状态</dt><dd>{STATUS_NAMES[selected.status] ?? selected.status}</dd><dt>参数</dt><dd className="mono">{selected.params}</dd><dt>结果</dt><dd className="mono">{selected.result ?? selected.error ?? '—'}</dd></dl><pre className="log">{log || '暂无日志'}</pre><div className="actions">{selected.status === 'queued' && <button onClick={() => api.cancel(selected.job_id).then(load)}>取消</button>}{['failed', 'interrupted', 'partial', 'blocked', 'mismatch', 'cancelled'].includes(selected.status) && <button onClick={() => api.retry(selected.job_id).then(load)}>重试</button>}</div></> : <p className="empty">选择任务查看详情</p>}</div></div></section>
}
