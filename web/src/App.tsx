import { useEffect, useState } from 'react'
import { api, type Job } from './api'
import DataCenter from './pages/DataCenter'
import Instruments from './pages/Instruments'
import Jobs from './pages/Jobs'

const PAGES = [{ id: 'data', name: '数据中心' }, { id: 'browse', name: '证券浏览' }, { id: 'jobs', name: '任务' }] as const
type PageId = (typeof PAGES)[number]['id']
const current = (): PageId => (PAGES.find(p => `#/${p.id}` === location.hash.split('?')[0])?.id ?? 'data')

export default function App() {
  const [page, setPage] = useState<PageId>(current)
  const [batch, setBatch] = useState<string | null>(null)
  const [queue, setQueue] = useState({ running: 0, queued: 0 })
  useEffect(() => { const f = () => setPage(current()); addEventListener('hashchange', f); return () => removeEventListener('hashchange', f) }, [])
  useEffect(() => {
    const tick = () => {
      api.status().then(s => setBatch(s.batch_id)).catch(() => setBatch(null))
      api.jobs().then((js: Job[]) => setQueue({ running: js.filter(x => x.status === 'running').length, queued: js.filter(x => x.status === 'queued').length })).catch(() => {})
    }
    tick(); const t = setInterval(tick, 5000); return () => clearInterval(t)
  }, [])
  return (
    <div className="shell">
      <aside className="side">
        <div className="brand"><b>observe</b><small>A 股研究工作台</small></div>
        <nav className="nav" aria-label="页面">
          {PAGES.map(p => <a key={p.id} href={`#/${p.id}`} aria-current={page === p.id ? 'page' : undefined}>{p.name}</a>)}
        </nav>
        <div className="side-foot">
          <div>已发布批次</div><div className="mono">{batch ?? '无'}</div>
          <div style={{ marginTop: 8 }}>任务 <span className="mono">{queue.running}</span> 运行 · <span className="mono">{queue.queued}</span> 排队</div>
        </div>
      </aside>
      <main className="main">
        {page === 'data' && <DataCenter />}
        {page === 'browse' && <Instruments />}
        {page === 'jobs' && <Jobs />}
      </main>
    </div>
  )
}
