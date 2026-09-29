// 后端接口封装：只取数，不做任何金融计算
export type Row = Record<string, any>
export type Job = { job_id: string; kind: string; status: string; params: string; created_at: string; started_at?: string; finished_at?: string; result?: string; error?: string; retry_of?: string }

async function j<T>(url: string, init?: RequestInit): Promise<T> {
  const r = await fetch(url, init)
  if (!r.ok) throw new Error(`${r.status} ${(await r.text()).slice(0, 200)}`)
  return r.json()
}
const post = <T,>(url: string, body?: unknown) => j<T>(url, { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: body === undefined ? undefined : JSON.stringify(body) })

export const api = {
  status: () => j<{ batch_id: string | null; published_at?: string; tables: Record<string, { partitions: number; rows: number }> }>('/api/data/status'),
  coverage: () => j<Row[]>('/api/data/coverage'),
  daily: () => j<Row[]>('/api/data/daily'),
  snapshots: () => j<string[]>('/api/data/snapshots'),
  issues: () => j<{ source: string; batch_id: string | null; status: string; rows: Row[]; audit_id: string | null; scope: 'snapshot' | 'incremental' | null; input_range: { start?: string; end?: string; days?: number; rows?: number } | null; audited_at: string | null }>('/api/data/issues'),
  instruments: (q: string) => j<Row[]>(`/api/instruments?q=${encodeURIComponent(q)}&limit=80`),
  bars: (id: string, price: 'raw' | 'adj') => j<Row[]>(`/api/instruments/${id}/bars?price=${price}`),
  actions: (id: string) => j<Row[]>(`/api/instruments/${id}/actions`),
  jobs: () => j<Job[]>('/api/jobs?limit=100'),
  job: (id: string) => j<Job>(`/api/jobs/${id}`),
  log: (id: string, offset: number) => j<{ offset: number; text: string }>(`/api/jobs/${id}/log?offset=${offset}`),
  submit: (kind: string, params: Row = {}) => post<{ job_id: string }>('/api/jobs', { kind, params }),
  cancel: (id: string) => post<{ cancelled: boolean }>(`/api/jobs/${id}/cancel`),
  retry: (id: string) => post<{ job_id: string }>(`/api/jobs/${id}/retry`),
}
