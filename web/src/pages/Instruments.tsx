import { useEffect, useRef, useState } from 'react'
import { api, type Row } from '../api'
import { day, px } from '../format'

export default function Instruments() {
  const [q, setQ] = useState(''); const [rows, setRows] = useState<Row[]>([]); const [selected, setSelected] = useState(''); const [bars, setBars] = useState<Row[]>([]); const [actions, setActions] = useState<Row[]>([]); const [price, setPrice] = useState<'raw' | 'adj'>('raw'); const [error, setError] = useState(''); const request = useRef(0)
  const search = () => api.instruments(q).then(setRows).catch(e => setError(String(e)))
  useEffect(() => { search() }, [])
  useEffect(() => { if (!selected) return; const id = ++request.current; Promise.all([api.bars(selected, price), api.actions(selected)]).then(([b, a]) => { if (id !== request.current) return; setBars(b); setActions(a); setError('') }).catch(e => { if (id === request.current) setError(String(e)) }) }, [selected, price])
  return <section className="page"><header className="page-head"><div><p className="eyebrow">INSTRUMENTS</p><h1>证券浏览</h1><p className="lede">查看原始价格、复权价格与公司行动</p></div></header>
    <div className="browse"><div className="results"><form onSubmit={e => { e.preventDefault(); search() }}><input value={q} onChange={e => setQ(e.target.value)} placeholder="代码或名称" /><button>查询</button></form>{rows.map(r => <button className="result" key={r.instrument} onClick={() => setSelected(r.instrument)} aria-current={selected === r.instrument}>{r.instrument}<span>{r.name ?? ''}</span></button>)}{!rows.length && <p className="empty">暂无证券</p>}</div>
      <div className="detail">{error && <p className="error">{error}</p>}{selected ? <><div className="detail-head"><h2>{selected}</h2><div className="segmented"><button onClick={() => setPrice('raw')} aria-pressed={price === 'raw'}>原始价</button><button onClick={() => setPrice('adj')} aria-pressed={price === 'adj'}>复权价</button></div></div><table><thead><tr><th>日期</th><th>开盘</th><th>收盘</th><th>复权因子</th><th>状态</th></tr></thead><tbody>{bars.slice(-80).map((r, i) => <tr key={i}><td>{day(r.date)}</td><td>{px(price === 'adj' ? r.open_adj : r.open)}</td><td>{px(price === 'adj' ? r.close_adj : r.close)}</td><td>{price === 'adj' ? px(r.back_factor) : '—'}</td><td>{r.is_trading === false ? '停牌' : price === 'adj' ? (r.adjusted_unavailable_reason ?? '') : ''}</td></tr>)}</tbody></table><h3>公司行动</h3>{actions.length ? <table><thead><tr><th>除权日</th><th>现金</th><th>送转</th><th>配股</th></tr></thead><tbody>{actions.map((r, i) => <tr key={i}><td>{day(r.ex_date)}</td><td>{px(r.cash_per_share)}</td><td>{px(r.bonus_ratio)}</td><td>{px(r.rights_ratio)}</td></tr>)}</tbody></table> : <p className="empty">暂无公司行动</p>}</> : <p className="empty">选择证券查看详情</p>}</div></div>
  </section>
}
