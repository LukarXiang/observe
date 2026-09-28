import pandas as pd


def audit_daily(frame):
    issues = []
    required = {"date", "instrument", "open", "high", "low", "close"}
    missing = required - set(frame.columns)
    if missing: issues.append({"kind": "missing_columns", "columns": sorted(missing)})
    if "close" in frame: issues.extend({"kind": "invalid_price", "row": int(index)} for index in frame.index[~pd.to_numeric(frame["close"], errors = "coerce").gt(0).fillna(False)])
    if frame.duplicated([c for c in ("date", "instrument") if c in frame]).any(): issues.append({"kind": "duplicate_key"})
    return pd.DataFrame(issues)
