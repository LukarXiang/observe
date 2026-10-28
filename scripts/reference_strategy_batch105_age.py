"""Explicit pandas3 compatibility reference for the original252-natural-day counter, never NHNL."""
import ast
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd

from scripts import research_strategy_batch105 as original
from scripts.verify_strategy_batch3 import require


def function(basic,periods):
    node = next(n for n in original.tree(1).body if isinstance(n,ast.FunctionDef))
    source = ast.unparse(node)
    replacements = {
        'stock_ipo_list.astype(np.datetime64)': "stock_ipo_list.astype('datetime64[ns]')",
        'columns=codes': 'columns=pd.Index(codes,dtype=object)',
        '.sub(stock_ipo_list)': '.sub(stock_ipo_list.set_axis(pd.Index(stock_ipo_list.index,dtype=object)).rename_axis(None))',
        '.applymap(lambda x: x.days)': '.map(lambda x: x.days)',
        '.groupby(level=0, axis=1).sum()': '.T.groupby(level=0).sum().T',
    }
    for before,after in replacements.items():
        require(source.count(before) == 1,'Compatibility patch scope differs: '+before)
        source = source.replace(before,after,1)
    namespace = {'pd':pd,'np':np,'Dict':Dict,'List':List,'get_ts_stock_basic':lambda:basic,
        'get_trade_days':lambda *a:periods}
    exec(compile(source,'<source89-explicit-pandas3-compatibility-reference>','exec'),namespace)
    implementation = namespace['get_ind_overthreshold_num']
    def checked(sw_cons_name,*args,**kwargs):
        selected = basic.loc[basic['code'].isin(sw_cons_name)]
        if selected['code'].duplicated().any():
            raise ValueError('duplicate labels in selected stock_basic codes')
        return implementation(sw_cons_name,*args,**kwargs)
    return checked


def verify_environment():
    require(pd.__version__ == '3.0.6' and np.__version__ == '2.5.3','Compatibility environment differs')
    return {'pandas':pd.__version__,'numpy':np.__version__,'source_strategy_modified':False,
        'compatibility_code':str(Path(__file__).resolve()),'economic_variant_approved':False}
