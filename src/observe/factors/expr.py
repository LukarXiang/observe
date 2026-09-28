"""因子表达式（模块 12、决策 7）：ast 白名单解析，不用 eval；推算回看长度；在「交易日 × 证券」宽表上向量化计算。"""
import ast, math
from dataclasses import dataclass

import numpy as np
import pandas as pd

FIELDS = {'open_adj', 'high_adj', 'low_adj', 'close_adj', 'ret', 'volume', 'amount', 'turnover'}
TS1 = {'ts_mean', 'ts_std', 'ts_sum', 'ts_min', 'ts_max', 'ts_rank', 'ts_slope', 'ts_decay_linear'}   # (x, w)
TS_LAG = {'ts_delay', 'ts_delta'}                                                                       # (x, d)
TS2 = {'ts_corr', 'ts_cov'}                                                                             # (x, y, w)
CS = {'cs_rank', 'cs_zscore', 'cs_demean', 'cs_winsorize'}                                              # (x)，只在当天候选上算
EL1, EL2 = {'abs', 'log', 'sign'}, {'max2', 'min2'}
MAX_WINDOW = 500


class ExprError(ValueError): pass


@dataclass(frozen = True)
class Parsed:
    expr: str
    tree: ast.AST
    lookback: int
    fields: frozenset


def _int(node, name):
    if not (isinstance(node, ast.Constant) and type(node.value) is int): raise ExprError(f'{name} 的窗口必须是整数常量')
    if not 1 <= node.value <= MAX_WINDOW: raise ExprError(f'{name} 的窗口 {node.value} 超出 1–{MAX_WINDOW}（禁止负延迟与非法窗口）')
    return node.value


def _check(n):
    """返回 (回看长度, 依赖字段)；不在白名单的语法一律报错"""
    if isinstance(n, ast.Expression): return _check(n.body)
    if isinstance(n, ast.Constant) and type(n.value) in (int, float): return 0, set()
    if isinstance(n, ast.Name):
        if n.id not in FIELDS: raise ExprError(f'未知字段 {n.id}')
        return 0, {n.id}
    if isinstance(n, ast.UnaryOp) and isinstance(n.op, (ast.USub, ast.UAdd)): return _check(n.operand)
    if isinstance(n, ast.BinOp) and isinstance(n.op, (ast.Add, ast.Sub, ast.Mult, ast.Div)):
        (a, fa), (b, fb) = _check(n.left), _check(n.right); return max(a, b), fa | fb
    if isinstance(n, ast.Compare) and len(n.ops) == 1 and isinstance(n.ops[0], (ast.Gt, ast.Lt, ast.GtE, ast.LtE)):
        (a, fa), (b, fb) = _check(n.left), _check(n.comparators[0]); return max(a, b), fa | fb
    if isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and not n.keywords:
        f, args = n.func.id, n.args
        if f in TS_LAG | TS1 and len(args) == 2:
            lb, fs = _check(args[0]); w = _int(args[1], f); return lb + (w if f in TS_LAG else w - 1), fs
        if f in TS2 and len(args) == 3:
            (a, fa), (b, fb) = _check(args[0]), _check(args[1]); return max(a, b) + _int(args[2], f) - 1, fa | fb
        if f in CS | EL1 and len(args) == 1: return _check(args[0])
        if f in EL2 and len(args) == 2:
            (a, fa), (b, fb) = _check(args[0]), _check(args[1]); return max(a, b), fa | fb
        if f == 'where' and len(args) == 3:
            r = [_check(x) for x in args]; return max(x[0] for x in r), set().union(*(x[1] for x in r))
        raise ExprError(f'不支持的函数或参数个数：{f}/{len(args)}')
    raise ExprError(f'不允许的语法：{type(n).__name__}')


def parse(expr):
    try: tree = ast.parse(expr, mode = 'eval')
    except SyntaxError as e: raise ExprError(f'语法错误：{e.msg}') from e
    lb, fs = _check(tree); return Parsed(expr, tree, lb, frozenset(fs))


# 计算 ---------------------------------------------------------------------------------------------------
def _div(a, b):
    b = b.where(b != 0) if isinstance(b, pd.DataFrame) else (np.nan if b == 0 else b)   # 分母为 0 输出缺失，不输出无穷大
    r = a / b
    return r.where(np.isfinite(r)) if isinstance(r, pd.DataFrame) else r


def _slope(x, w, mp):
    t = pd.DataFrame(np.arange(len(x), dtype = float)[:, None].repeat(x.shape[1], 1), index = x.index, columns = x.columns).where(x.notna())
    def r(s): return s.rolling(w, min_periods = mp).mean()
    return _div(r(x * t) - r(x) * r(t), r(t * t) - r(t) ** 2)


def _cross(f, x, mask):
    v = x.where(mask)
    if f == 'cs_rank': out = v.rank(axis = 1, pct = True)
    elif f == 'cs_demean': out = v.sub(v.mean(axis = 1), axis = 0)
    elif f == 'cs_zscore':
        sd = v.std(axis = 1, ddof = 1); out = v.sub(v.mean(axis = 1), axis = 0).div(sd.where(sd > 0), axis = 0)
    else:
        med = v.median(axis = 1); mad = v.sub(med, axis = 0).abs().median(axis = 1); out = v.clip(med - 5 * mad, med + 5 * mad, axis = 0)
    return out.where(mask)


def compute(expr, panel, eligible = None, min_obs_ratio = 1.0):
    """panel: {字段: 宽表(交易日 × 证券)}，停牌日为缺失；eligible: 同形状布尔表，横截面算子只在 True 上算。
    窗口内有效值少于 ceil(w × min_obs_ratio) 时输出缺失"""
    p = parse(expr) if isinstance(expr, str) else expr
    ref = panel[next(iter(p.fields))] if p.fields else next(iter(panel.values()))
    mask = eligible if eligible is not None else pd.DataFrame(True, index = ref.index, columns = ref.columns)
    def frame(v):
        if isinstance(v, pd.DataFrame): return v
        return pd.DataFrame(v, index = ref.index, columns = ref.columns, dtype = float)
    def mp(w): return max(1, math.ceil(w * min_obs_ratio))

    def ev(n):
        if isinstance(n, ast.Expression): return ev(n.body)
        if isinstance(n, ast.Constant): return float(n.value)
        if isinstance(n, ast.Name): return panel[n.id].astype(float)
        if isinstance(n, ast.UnaryOp): v = ev(n.operand); return -v if isinstance(n.op, ast.USub) else v
        if isinstance(n, ast.BinOp):
            a, b = ev(n.left), ev(n.right); op = type(n.op)
            return a + b if op is ast.Add else a - b if op is ast.Sub else a * b if op is ast.Mult else _div(a, b)
        if isinstance(n, ast.Compare):
            a, b, op = ev(n.left), ev(n.comparators[0]), type(n.ops[0])
            r = a > b if op is ast.Gt else a < b if op is ast.Lt else a >= b if op is ast.GtE else a <= b
            return r.astype(float).where(a.notna()) if isinstance(a, pd.DataFrame) else r
        f, args = n.func.id, n.args
        if f in TS_LAG:
            x, d = ev(args[0]), args[1].value
            if not isinstance(x, pd.DataFrame): raise ExprError(f'{f} 的第一个参数必须是表格')
            return x.shift(d) if f == 'ts_delay' else x - x.shift(d)
        if f in TS1:
            x, w = ev(args[0]), args[1].value
            if not isinstance(x, pd.DataFrame): raise ExprError(f'{f} 的第一个参数必须是表格')
            r = x.rolling(w, min_periods = mp(w))
            if f == 'ts_slope': return _slope(x, w, mp(w))
            if f == 'ts_rank': return r.rank(pct = True)
            if f == 'ts_std': return r.std(ddof = 1)
            if f == 'ts_decay_linear':
                wt = np.arange(1, w + 1, dtype = float); return x.rolling(w, min_periods = w).apply(lambda a: a @ wt / wt.sum(), raw = True)
            return getattr(r, f[3:])()
        if f in TS2:
            x, y, w = ev(args[0]), ev(args[1]), args[2].value
            if not isinstance(x, pd.DataFrame) or not isinstance(y, pd.DataFrame): raise ExprError(f'{f} 的参数必须是表格')
            return getattr(x.rolling(w, min_periods = mp(w)), f[3:])(y)
        if f in CS: return _cross(f, ev(args[0]), mask)
        if f in EL1:
            x = ev(args[0])
            if not isinstance(x, pd.DataFrame): return float(np.abs(x) if f == 'abs' else np.sign(x) if f == 'sign' else np.log(x)) if (f != 'log' or x > 0) else np.nan
            return np.abs(x) if f == 'abs' else np.sign(x) if f == 'sign' else np.log(x.where(x > 0))
        if f in EL2:
            a, b = ev(args[0]), ev(args[1])
            if not isinstance(a, pd.DataFrame) and not isinstance(b, pd.DataFrame): return max(a, b) if f == 'max2' else min(a, b)
            a, b = frame(a), frame(b); fn = np.maximum if f == 'max2' else np.minimum
            return fn(a, b).where(a.notna() & b.notna())
        c, a, b = ev(args[0]), ev(args[1]), ev(args[2])
        if not any(isinstance(x, pd.DataFrame) for x in (c, a, b)): return a if c > 0 else b
        c, a, b = frame(c), frame(a), frame(b)
        return a.where(c > 0, b).where(c.notna())   # where(条件 > 0, a, b)

    out = ev(p.tree)
    return out if isinstance(out, pd.DataFrame) else pd.DataFrame(out, index = ref.index, columns = ref.columns)
