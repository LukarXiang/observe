"""通达信适配器：只用除权除息（K 线与实时行情接口在当前服务器上已失效，见实测报告）。不走系统代理（socket 直连）。"""
import logging, time

from .. import raw

HOSTS = [('123.125.108.14', 7709), ('124.70.199.56', 7709), ('121.36.225.169', 7709), ('124.71.187.72', 7709), ('119.97.185.59', 7709)]   # 2026-09-28 实测可用


class Tdx:
    name, upstream = 'mootdx', 'tdx'

    def __init__(self, root, hosts = HOSTS):
        self.root, self.hosts, self.client = root, hosts, None

    def _connect(self):
        from mootdx.quotes import Quotes
        logging.getLogger('mootdx').setLevel(logging.ERROR)
        for h in self.hosts:
            try: self.client = Quotes.factory(market = 'std', server = h, timeout = 8); return
            except Exception: continue   # noqa: BLE001  换下一台
        raise ConnectionError('通达信服务器全部不可用')

    def xdxr(self, code):
        """code: 600000；返回原始除权除息表（可能为空）"""
        if self.client is None: self._connect()
        t = time.perf_counter(); df = self.client.xdxr(symbol = code)
        n = 0 if df is None else len(df); raw.log(self.root, self.name, self.upstream, 'xdxr', {'code': code}, 'success', n, sec = round(time.perf_counter() - t, 2))
        return df

    def close(self):
        if self.client is not None: self.client.close(); self.client = None
