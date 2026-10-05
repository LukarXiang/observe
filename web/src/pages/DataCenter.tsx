import { useEffect, useState } from 'react'
import { api, type Row } from '../api'
import { day, int, time, TABLE_NAMES } from '../format'

const AUDIT_STATUS: Record<string, string> = { expired: '审计已过期', problem: '审计存在问题', not_audited: '尚未审计', passed_incremental: '只审计了增量区间，未做全快照审计', rules_unavailable: '执行规则无法加载，不能确认审计有效' }

export default function DataCenter() {
  const [status, setStatus] = useState<any>(null); const [coverage, setCoverage] = useState<Row[]>([]); const [issues, setIssues] = useState<any>(null); const [error, setError] = useState('')
  const load = () => Promise.all([api.status(), api.coverage(), api.issues()]).then(([s, c, i]) => { setStatus(s); setCoverage(c); setIssues(i); setError('') }).catch(e => setError(String(e)))
  useEffect(() => { load() }, [])
  if (error) return <section className="page"><h1>数据中心</h1><p className="error">{error}</p><button onClick={load}>重试</button></section>
  if (!status) return <section className="page"><h1>数据中心</h1><p className="empty">加载中…</p></section>
  return <section className="page"><header className="page-head"><div><p className="eyebrow">DATA CENTER</p><h1>数据中心</h1><p className="lede">发布版本、覆盖范围与审计状态</p></div><div className="actions"><button onClick={load}>刷新</button></div></header>
    <div className="metrics"><div><span>发布批次</span><b className="mono">{status.batch_id ?? '无'}</b><small>{time(status.published_at)}</small></div><div><span>数据表</span><b>{Object.keys(status.tables).length}</b></div><div><span>审计问题</span><b>{issues?.rows?.length ?? 0}</b></div></div>
    <div className="section"><h2>已发布数据</h2><table><thead><tr><th>表</th><th>分区</th><th>记录数</th></tr></thead><tbody>{Object.entries(status.tables).map(([k, v]: any) => <tr key={k}><td>{TABLE_NAMES[k] ?? k}</td><td>{v.partitions}</td><td>{int(v.rows)}</td></tr>)}</tbody></table></div>
    <div className="section"><h2>日线覆盖</h2>{coverage.length ? <table><thead><tr><th>年份</th><th>日期数</th><th>记录数</th><th>交易记录</th><th>证券数</th></tr></thead><tbody>{coverage.map(r => <tr key={r.year}><td>{r.year}</td><td>{int(r.n_days)}</td><td>{int(r.n_rows)}</td><td>{int(r.trading_rows)}</td><td>{int(r.n_instruments)}</td></tr>)}</tbody></table> : <p className="empty">暂无覆盖数据</p>}</div>
    <div className="section"><h2>最近审计</h2>{issues?.status !== 'passed' && <p className="error">{AUDIT_STATUS[issues?.status] ?? '尚未审计'}</p>}{issues?.scope && <p className="empty">范围：{issues.scope === 'snapshot' ? '全快照' : '增量区间'} · {day(issues.input_range?.start)} 至 {day(issues.input_range?.end)} · {int(issues.input_range?.days)} 个交易日 · {time(issues.audited_at)}</p>}{issues?.rows?.length ? <table><thead><tr><th>等级</th><th>规则</th><th>日期</th><th>证券</th><th>说明</th></tr></thead><tbody>{issues.rows.slice(0, 20).map((r: Row, i: number) => <tr key={i}><td>{r.level}</td><td>{r.rule}</td><td>{day(r.date)}</td><td>{r.instrument ?? '—'}</td><td>{r.detail ?? ''}</td></tr>)}</tbody></table> : issues?.status === 'passed' ? <p className="empty">审计通过，暂无问题记录</p> : null}</div>
  </section>
}
