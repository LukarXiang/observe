import pandas as pd

from ..raw import append_request
from ..standardize import standardize_daily


class BaoStockSource:
    name = "baostock"

    def __init__(self, data_root = "data"):
        self.data_root = data_root

    def _query(self, result):
        if result.error_code != "0": raise RuntimeError(f"{result.error_code}: {result.error_msg}")
        rows = []
        while result.next(): rows.append(result.get_row_data())
        return pd.DataFrame(rows, columns = result.fields)

    def daily(self, start, end, bs_module):
        fields = "date,code,open,high,low,close,preclose,volume,amount,turn,tradestatus,pctChg,isST"
        session = bs_module.login()
        if session.error_code != "0": raise RuntimeError(session.error_msg)
        try:
            frame = self._query(bs_module.query_history_k_data_plus("sh.600000", fields, start_date = str(start), end_date = str(end), frequency = "d", adjustflag = "3"))
            append_request(self.data_root, self.name, "BaoStock", "query_history_k_data_plus", {"start": str(start), "end": str(end)}, "success", len(frame))
            return standardize_daily(frame)
        finally: bs_module.logout()
