"""Archive ETF6 mean boundaries without changing signals or applying another approval."""
import json
from decimal import Decimal, localcontext
import math
from pathlib import Path
import shutil

import pandas as pd

from observe.runs import file_sha
from scripts import reference_strategy_batch103 as reference
from scripts.recover_strategy_batch94_reference import network_blocked
from scripts.review_strategy_batch18 import save
from scripts.verify_strategy_batch3 import require

DIRECTORY = Path('data/staging/strategies-batch103/20261011-price-numeric-boundaries')


def run():
    reference.validate(); require(not DIRECTORY.exists(), 'Immutable numerical evidence exists')
    DIRECTORY.mkdir(parents=True)
    with network_blocked(), localcontext() as decimal_context:
        decimal_context.prec = 50
        input_path = reference.DIRECTORY/'compute/real-price-reference.parquet'
        frame = pd.read_parquet(input_path)
        differing = frame.loc[frame.price_eligible_reference.ne(frame.independent_eligible)]
        rows = []
        for item in differing.itertuples(index=False):
            native_path = reference.DIRECTORY/'inputs'/(item.instrument+'.parquet')
            native = pd.read_parquet(native_path)
            end = native.index[native.date.eq(item.asof_native_close)].item()
            prices = native.close.iloc[end-13:end].to_numpy(float); current = float(native.close.iloc[end])
            original = float(prices.mean()); stable = math.fsum(prices)/13
            exact_decimal = sum((Decimal(str(x)) for x in prices), Decimal(0))/Decimal(13)
            require(original == item.MA13 and stable == item.independent_MA13 and
                (current/original-1)*100 == item.deviation and (current/stable-1)*100 == item.independent_deviation,
                'Actual numeric boundary changed')
            rows.append({'instrument': item.instrument, 'date': item.asof_native_close,
                'window_first': item.window_first, 'history_prices': prices.tolist(), 'current_EOD_operand': current,
                'current_hex': current.hex(), 'original_mean': original, 'original_mean_hex': original.hex(),
                'fsum_mean': stable, 'fsum_mean_hex': stable.hex(), 'decimal_string_mean': str(exact_decimal),
                'original_deviation_percent': item.deviation, 'fsum_deviation_percent': item.independent_deviation,
                'original_price_eligible_reference': item.price_eligible_reference,
                'fsum_price_eligible_reference': item.independent_eligible, 'missing_open_count': item.missing_open_count,
                'native_input': str(native_path), 'native_input_sha256': file_sha(native_path)})
        script = Path('scripts/diagnose_strategy_batch103_numerics.py'); shutil.copyfile(script, DIRECTORY/script.name)
        doc = {'status': 'observed_reference_boundaries', 'count': len(rows), 'rows': rows,
            'reference_file': str(input_path), 'reference_sha256': file_sha(input_path),
            'implementation_sha256': file_sha(script), 'production_rules_changed': False,
            'existing_numeric_approval_not_extended': True, 'full_original_signals': 0, 'new_trading_backtests': 0}
        save(DIRECTORY/'checked.json', doc)
    return doc


if __name__ == '__main__': print(json.dumps(run(), ensure_ascii=False))
