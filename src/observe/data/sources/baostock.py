"""BaoStock 适配器：全机单会话（持 baostock 锁），结果集逐行读取（0.9.4 的 get_data 与 pandas 2+ 不兼容，决策 3）。"""
import time
from contextlib import contextmanager

import pandas as pd

from .. import raw
from ..locks import BAOSTOCK, operation_lock


class BaoStockError(RuntimeError): pass


class BaoStock:
    name = upstream = 'baostock'

    def __init__(self, root, bs = None):
        self.root = root; self.bs = bs

    @contextmanager
    def session(self):
        if self.bs is None: import baostock as bs; self.bs = bs
        with operation_lock(self.root, BAOSTOCK):
            lg = self.bs.login()
            if lg.error_code != '0': raise BaoStockError(f'登录失败 {lg.error_code} {lg.error_msg}')
            try: yield self
            finally: self.bs.logout()

    def _rows(self, endpoint, params, call):
        t = time.perf_counter()
        try:
            rs = raw.retry(call)
            if rs.error_code != '0': raise BaoStockError(f'{endpoint} {rs.error_code} {rs.error_msg}')
            rows = []
            while (rs.error_code == '0') & rs.next(): rows.append(rs.get_row_data())
            if rs.error_code != '0': raise BaoStockError(f'{endpoint} 翻页中断 {rs.error_code} {rs.error_msg}')
        except Exception as e:
            raw.log(self.root, self.name, self.upstream, endpoint, params, 'failed', error = str(e)[:300], sec = round(time.perf_counter() - t, 2)); raise
        df = pd.DataFrame(rows, columns = rs.fields); raw.log(self.root, self.name, self.upstream, endpoint, params, 'success', len(df), sec = round(time.perf_counter() - t, 2))
        return df

    def calendar(self, start, end): return self._rows('query_trade_dates', {'start': start, 'end': end}, lambda: self.bs.query_trade_dates(str(start), str(end)))
    def stock_basic(self): return self._rows('query_stock_basic', {}, lambda: self.bs.query_stock_basic())
    def daily_market(self, day): return self._rows('query_daily_history_k_AStock', {'date': day}, lambda: self.bs.query_daily_history_k_AStock(str(day)))
    def adjust_factor_day(self, day): return self._rows('query_daily_adjust_factor', {'date': day}, lambda: self.bs.query_daily_adjust_factor(str(day)))
    def adjust_factor(self, code, start = '1990-01-01', end = '2099-12-31'):
        return self._rows('query_adjust_factor', {'code': code}, lambda: self.bs.query_adjust_factor(code = code, start_date = str(start), end_date = str(end)))
    def index_daily(self, code, start, end):
        f = 'date,code,open,high,low,close,preclose,volume,amount'
        return self._rows('query_history_k_data_plus', {'code': code, 'start': start, 'end': end}, lambda: self.bs.query_history_k_data_plus(code, f, start_date = str(start), end_date = str(end), frequency = 'd', adjustflag = '3'))
