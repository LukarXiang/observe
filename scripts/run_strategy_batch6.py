"""Use python -m scripts.run_strategy_batch6; source98 lagged SVM checks."""
import argparse
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.svm import SVC

from observe.data.prices import with_adjusted
from observe.data.store import Store
from observe.execution import sessions
from observe.runs import file_sha, write_json
from scripts.archive_strategy_progress import archive
from scripts.run_strategy_batch4 import fee_checks, protect, read
from scripts.run_strategy_batch5 import run_one, verify_one
from scripts.verify_strategy_batch3 import require


def signal_check(root, output):
    cfg = read(output / 'config.json')['config']; store = Store(root); state = store.state(cfg['snapshot'])
    filters = [('instrument', 'in', cfg['execution_instruments'])]
    view = with_adjusted(store.load_state(state, 'bars_1d', filters = filters), store.load_state(state, 'adj_factors', filters = filters),
                         store.load_state(state, 'adj_coverage', filters = filters))
    view['date'] = pd.to_datetime(view.date).dt.date; calendar = sessions(store.load_state(state, 'calendar'))
    prices = view.set_index('date').reindex(calendar)
    samples = pd.read_parquet(output / 'svm_samples.parquet'); models = read(output / 'svm_models.json')
    factors = pd.read_parquet(output / 'factors.parquet').set_index('date')
    targets = pd.read_parquet(output / 'targets.parquet')
    fields = ['close_mean', 'volume_mean', 'high_mean', 'low_mean', 'volume_first_ratio', 'close_first_ratio', 'close_std']
    predictions = {0: 0, 1: 0}; checks = []; samples_checked = 0
    for model in models:
        day = pd.Timestamp(model['decision_date']).date(); k = calendar.index(day)
        actual = samples[samples.decision_date.eq(day)].sort_values('feature_date')
        expected_dates = calendar[k - 251 + 22:k - 4]
        require(list(actual.feature_date) == expected_dates and len(actual) == 225, 'Original feature-date slice differs')
        require(list(actual.role) == ['train'] * 224 + ['prediction_only'], 'Holdout role differs')
        require(actual.iloc[-1].feature_date == calendar[k - 5] and actual.iloc[-2].label_end == calendar[k - 1], 'Lag or label maturity differs')
        for row in actual.itertuples():
            f = calendar.index(row.feature_date); window = prices.iloc[f - 21:f + 1]
            close = window.close_adj.tolist(); high = window.high_adj.tolist(); low = window.low_adj.tolist(); volume = window.volume.tolist()
            avg = lambda values: math.fsum(values) / 22
            mean = avg(close)
            expected = [close[-1] / mean, volume[-1] / avg(volume), high[-1] / avg(high), low[-1] / avg(low),
                        volume[-1] / volume[0], close[-1] / close[0], math.sqrt(math.fsum((p - mean) ** 2 for p in close) / 22)]
            require(all(math.isclose(getattr(row, column), value, rel_tol = 1e-10, abs_tol = 1e-8) for column, value in zip(fields, expected, strict = True)), f'{day}/{row.feature_date}: feature differs')
            require(row.feature_start == calendar[f - 21] and row.label_end == calendar[f + 5] and row.label == int(prices.iloc[f + 5].close_adj > close[-1]), 'Label or feature window differs')
            samples_checked += 1
        x = actual[fields].to_numpy(float); y = actual.label.to_numpy(int)
        clf = SVC(**cfg['parameters']['svc_parameters']).fit(x[:-1], y[:-1]); prediction = int(clf.predict(x[-1:])[0])
        require(prediction == model['prediction'] == factors.loc[day, 'prediction'], 'Independent SVC prediction differs')
        require(model['parameters'] == cfg['parameters']['svc_parameters'] and model['train_samples'] == 224, 'Model parameters differ')
        require(model['feature_date'] == str(calendar[k - 5]) and model['train_label_end'] == str(calendar[k - 1]) and model['prediction_label_end'] == str(day), 'Model timing trace differs')
        require(clf.support_.tolist() == model['support'] and np.allclose(clf.support_vectors_, model['support_vectors'], rtol = 0, atol = 1e-9), 'Fitted supports differ')
        predictions[prediction] += 1
        execution_day = calendar[k + 1] if k + 1 < len(calendar) else None
        if execution_day is not None:
            week = [d for d in calendar if d.isocalendar()[:2] == execution_day.isocalendar()[:2]]
            require(len(week) >= 3 and week[2] == execution_day, 'Execution is not the third session in the week')
        checks.append({'decision_date': str(day), 'feature_date': model['feature_date'], 'train_label_end': model['train_label_end'],
                       'execution_day': str(execution_day), 'prediction': prediction, 'train_samples': 224})
    require(int(factors.is_fit.sum()) == len(models), 'Fit-day coverage differs')
    held_weight = 0.
    for day, row in factors.iterrows():
        if row.is_fit: held_weight = float(row.prediction)
        require(row.weight == held_weight, 'Weight state differs')
        weights = targets[targets.decision_date.eq(day)].set_index('instrument')
        require(weights.loc[cfg['instrument'], 'weight'] == held_weight and weights.loc['CASH', 'weight'] == 1 - held_weight, 'Frozen target differs')
    fit_days = {m['decision_date'] for m in models}; order_checks = []
    for child in read(output / 'subruns.json')['backtests']:
        orders = read(Path(child['output']) / 'orders.json')
        require(all(o['decision_date'] in fit_days for o in orders), 'Off-schedule refill or rebalance')
        require(all(calendar[calendar.index(pd.Timestamp(o['decision_date']).date()) + 1] == pd.Timestamp(o['exec_date']).date() for o in orders), 'Not next-open execution')
        order_checks.append({'scenario': child['scenario'], 'orders_checked': len(orders)})
    return {'sessions_checked': len(factors), 'fits_checked': len(models), 'samples_checked': samples_checked,
            'prediction_counts': predictions, 'fits': checks, 'orders': order_checks,
            'method': 'Independent math.fsum feature arithmetic/date windows/labels, fresh sklearn SVC per archived sample set; original feature lag kept'}


def summarize(root, directory, output):
    output = Path(output)
    if output.exists(): raise FileExistsError(output)
    path = directory / 'svm_lagged_shape_batch6-verification.json'; doc = read(path)
    fees = read(directory / 'fee-hand-checks.json'); protection = read(directory / 'protection.json')
    require(doc['status'] == fees['status'] == protection['status'] == 'ok', 'Incomplete verification')
    report = doc['report']
    result = {'status': 'ok', 'run_id': doc['run']['run_id'], 'reproduction': doc['reproduction'],
              'source_sha256': doc['source_sha256'], 'freeze': read(directory / 'svm_lagged_shape_batch6-freeze.json'),
              'hand_check': doc['hand_check'], 'period': report['period'], 'results': report['results'], 'benchmark': report['benchmark'][0],
              'validation_file': str(path), 'validation_sha256': file_sha(path), 'protection': protection, 'fees': fees,
              'original_strategy_complete': False, 'progress': archive(root, '第六批SVM滞后形状修正验收完成；三成本、禁网复现及旧资产保护通过'),
              'limitations': report['strategy']['review']['differences'] + report['strategy']['review']['gaps']}
    write_json(output, result); return {'status': 'ok', 'output': str(output), 'progress': result['progress']['output']}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description = __doc__)
    parser.add_argument('action', choices = ['run', 'verify', 'fees', 'protect', 'summary'])
    parser.add_argument('--root', default = 'data'); parser.add_argument('--directory', default = 'data/staging/strategies-batch6/20261005-svm-lagged')
    parser.add_argument('--config', default = 'configs/strategies/svm_lagged_shape_batch6.yaml')
    parser.add_argument('--output', default = 'docs/handoff/2026-10-05-batch6-verification.json')
    args = parser.parse_args(); directory = Path(args.directory)
    if args.action == 'run': result = run_one(args.root, directory, args.config, batch_label = '第六批')
    elif args.action == 'verify': result = verify_one(args.root, directory, 'svm_lagged_shape_batch6', checker = signal_check, batch_label = '第六批')
    elif args.action == 'fees': result = fee_checks(directory)
    elif args.action == 'protect': result = protect(args.root, directory)
    else: result = summarize(args.root, directory, args.output)
    print(json.dumps(result, ensure_ascii = False))
