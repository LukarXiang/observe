def build_universe(daily, asof, min_listing_days = 60, exclude_st = True):
    frame = daily[daily["date"] == asof].copy()
    if exclude_st and "isST" in frame: frame = frame[frame["isST"].astype(str) != "1"]
    if "listed_days" in frame: frame = frame[frame["listed_days"] >= min_listing_days]
    if "instrument" in frame: frame = frame[frame["instrument"].str.match(r"^(60|00)\d{4}\.(SH|SZ)$")]
    return frame.assign(asof = asof, eligible = True)[[c for c in ("asof", "instrument", "eligible") if c in frame.columns or c == "eligible"]]
