def target_list(scores, positions, n = 20, buffer = 10, max_sell = 5, candidates = None):
    candidates = set(candidates if candidates is not None else scores)
    ranked = sorted(((i, scores[i]) for i in candidates if i in scores), key = lambda x: (-x[1], x[0]))
    rank = {i: n + 1 for i, _ in ranked}
    for k, (i, _) in enumerate(ranked, 1): rank[i] = k
    held = set(positions)
    forced = sorted(held - candidates)
    missing = sorted(i for i in held & candidates if i not in scores)
    kept = set(missing) | {i for i in held & candidates if i in scores and rank[i] <= n + buffer}
    ordinary = sorted((i for i in held & candidates - kept if i in scores), key = lambda i: (scores[i], i))
    allowed = len(ordinary) if max_sell is None else max(0, int(max_sell)); exits = forced + ordinary[:allowed]
    reserve = kept | (held - set(forced) - set(ordinary[:allowed]))
    buys = [i for i, _ in ranked if i not in held and i not in reserve][:max(0, n - len(reserve))]
    return {"target": sorted(reserve | set(buys)), "exits": exits, "buys": buys, "rank": rank, "missing": missing, "forced": forced}

def orders_for_targets(result, positions, target_amounts, refill = True):
    orders = [{"instrument": i, "side": "sell", "qty": "all", "reason": "exit_universe" if i in result["forced"] else "exit_rank"} for i in result["exits"]]
    if refill: orders += [{"instrument": i, "side": "buy", "amount": target_amounts[i], "reason": "enter_top"} for i in result["buys"]]
    return orders
