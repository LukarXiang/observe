"""只读核验：外部分钟文件里缺失的少量证券，压缩包里是否真的没有，BaoStock 5 分钟线能否提供。不写数据表，不补数。
用法：python scripts/check_missing_minute_source.py --source "D:/projects/observe/data/A股分钟线" --instrument 601989.SH:2024-03-15 ... [--output check.json]
每个 证券:日期 查一次压缩包（成员名里是否有该代码）和一次 BaoStock 5 分钟线（登录、查询、登出）；总查询数应该很少，不是全市场补数。"""
import argparse, json
from datetime import datetime
from pathlib import Path

from observe.data.minute import archive_path, read_members


def bs_code(inst): code, mk = inst.split('.'); return f'{mk.lower()}.{code}'


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--source', required = True); ap.add_argument('--instrument', action = 'append', required = True, metavar = '代码.市场:日期'); ap.add_argument('--output'); ap.add_argument('--sevenzip'); a = ap.parse_args()
    checks = [(x.split(':')[0], datetime.strptime(x.split(':')[1], '%Y-%m-%d').date()) for x in a.instrument]
    if len(checks) > 12: raise SystemExit('核验的证券日太多：这个脚本只做少量抽查，不是补数')
    import baostock as bs
    lg = bs.login(); out = {'checked_at': datetime.now().isoformat(timespec = 'seconds'), 'baostock_login': lg.error_code == '0', 'checks': []}
    try:
        for inst, day in checks:
            path = archive_path(a.source, day); row = {'instrument': inst, 'date': str(day), 'archive': None if path is None else Path(path).name}
            if path is not None:
                keep = {(inst.split('.')[1].lower(), inst.split('.')[0]): inst}; row['in_archive'] = inst in read_members(path, keep, a.sevenzip)
            if lg.error_code == '0':
                rs = bs.query_history_k_data_plus(bs_code(inst), 'time,open,close,volume,amount', start_date = str(day), end_date = str(day), frequency = '5', adjustflag = '3')
                n = 0; first = last = None
                while rs.error_code == '0' and rs.next():
                    r = rs.get_row_data(); n += 1; first = first or r[0]; last = r[0]
                row.update(baostock_error = rs.error_code if rs.error_code != '0' else None, baostock_bars = n, baostock_first = first, baostock_last = last)
            out['checks'].append(row)
    finally: bs.logout()
    text = json.dumps(out, ensure_ascii = False, indent = 1); print(text)
    if a.output: Path(a.output).write_text(text, encoding = 'utf-8')


if __name__ == '__main__': main()
