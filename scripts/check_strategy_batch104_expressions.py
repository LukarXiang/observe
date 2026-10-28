"""Compare all saved MA flags to the original, unchanged condition AST expressions."""
import argparse
import ast
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from observe.runs import file_sha
from scripts import reference_strategy_batch104 as reference
from scripts.review_strategy_batch18 import save
from scripts.verify_strategy_batch3 import require


def check(output):
    path = reference.DIRECTORY/'compute/short-MA-reference.parquet'; frame = pd.read_parquet(path)
    price = np.stack(frame.close_means); volume = np.stack(frame.volume_means)
    operands = pd.DataFrame({
        'close_prev1_ma5':price[:,0], 'close_prev1_ma10':price[:,1],
        'close_prev1_ma20':price[:,2], 'close_prev1_ma30':price[:,3],
        'close_prev2_ma5':price[:,4], 'close_prev2_ma10':price[:,5],
        'close_prev2_ma20':price[:,6], 'close_prev2_ma30':price[:,7],
        'volume_prev1_ma5':volume[:,0], 'volume_prev1_ma10':volume[:,1],
        'volume_prev2_ma5':volume[:,2], 'volume_prev2_ma10':volume[:,3]})
    source = reference.study.tree(0)
    function = next(n for n in source.body if isinstance(n,ast.FunctionDef) and n.name == 'stock_filter1')
    expressions = {n.targets[0].id:n.value for n in function.body if isinstance(n,ast.Assign)
        and len(n.targets) == 1 and isinstance(n.targets[0],ast.Name) and n.targets[0].id in ('condition1','condition2','condition3')}
    require(set(expressions) == {'condition1','condition2','condition3'},'Original condition AST scope differs')
    values = []
    for name in ('condition1','condition2','condition3'):
        node = expressions[name]
        require(not any(isinstance(n,(ast.Call,ast.Attribute)) for n in ast.walk(node)), 'Condition has unreviewed executable calls')
        expression = ast.Expression(body=node)
        result = eval(compile(expression,'<unchanged-original-condition>','eval'),{'__builtins__':{}},{'stock_ma':operands})
        values.append(result.to_numpy())
    actual = np.stack(values,axis=1); saved = np.stack(frame.original_conditions)
    require(np.array_equal(actual,saved) and np.array_equal(actual.all(axis=1),frame.original_all_conditions.to_numpy()),
            'Saved reconstructed condition does not match actual original AST')
    result = {'status':'ok_component_only','rows':len(frame),'original_condition_cells_checked':int(actual.size),
        'original_AST_sha256':{name:hashlib.sha256(ast.dump(node,include_attributes=False).encode()).hexdigest() for name,node in expressions.items()},
        'evidence_sha256':{str(path):file_sha(path),str(reference.study.PREREAD/'0.source'):reference.study.SOURCE_SHA[0],
            'scripts/check_strategy_batch104_expressions.py':file_sha('scripts/check_strategy_batch104_expressions.py')},
        'reconstructed_condition_differences':0,'complete_original_pool_signals':0,'new_trading_backtests':0}
    output = Path(output); output.mkdir(parents=True,exist_ok=False); save(output/'checked.json',result)
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',required=True)
    print(json.dumps(check(parser.parse_args().output),ensure_ascii=False))
