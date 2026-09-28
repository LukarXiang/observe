import pandas as pd


def standardize_daily(frame):
    out = frame.copy(); out["date"] = pd.to_datetime(out["date"]).dt.date
    if "code" in out: out["instrument"] = out["code"].map(normalize_instrument)
    for column in ("open", "high", "low", "close", "preclose", "volume", "amount"):
        if column in out: out[column] = pd.to_numeric(out[column], errors = "coerce")
    return out


def normalize_instrument(code):
    code = str(code); parts = code.split(".")
    if len(parts) == 2: return f"{parts[1]}.{parts[0].upper()}"
    prefix = "SH" if code.startswith(("6", "68")) else "SZ"
    return f"{code.zfill(6)}.{prefix}"
