// 只做显示格式，不做计算
export const int = (v: number | null | undefined) => (v == null || Number.isNaN(v) ? '—' : Math.round(v).toLocaleString('zh-CN'))
export const px = (v: number | null | undefined) => (v == null || Number.isNaN(v) ? '—' : v.toFixed(2))
export const pct = (v: number | null | undefined, d = 2) => (v == null || Number.isNaN(v) ? '—' : `${v >= 0 ? '+' : ''}${(v * 100).toFixed(d)}%`)
export const day = (v: string | null | undefined) => (v ? String(v).slice(0, 10) : '—')
export const time = (v: string | null | undefined) => (v ? String(v).replace('T', ' ').slice(5, 16) : '—')
export const TABLE_NAMES: Record<string, string> = { calendar: '交易日历', instruments: '证券资料', bars_1d: '日线', adj_factors: '复权因子', corp_actions: '公司行动', shares: '股本', index_1d: '指数日线', bars_5m: '5 分钟线' }
export const KIND_NAMES: Record<string, string> = { data_update: '数据更新', data_audit: '数据审计', snapshot: '冻结快照', gc: '清理', run_experiment: '离线回放', reproduce: '复现', research: '研究流水线' }
export const STATUS_NAMES: Record<string, string> = { queued: '排队', running: '运行中', success: '成功', success_limited: '成功（有限制）', partial: '部分完成', blocked: '阻断', mismatch: '复现不一致', failed: '失败', cancelled: '已取消', interrupted: '中断' }
export const RULE_NAMES: Record<string, string> = { empty_day: '全市场 0 行', bad_price: '价格非正或缺失', ohlc_order: '开高低收顺序错误', suspended_has_price: '停牌行带价格', count_jump: '在市证券数跳变', beyond_limit: '超出涨跌幅', vwap_outside: '均价越界' }
