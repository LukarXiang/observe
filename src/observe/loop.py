from .ledger.book import Book
from .portfolio import orders_for_targets, target_list


def run_loop(dates, market, scores_by_date, initial_cash, rules, rebalance_every = 5, open_cash_policy = "sell_then_buy", slippage = 0.0, n = 20, buffer = 10, max_sell = 5, refill_between_rebalance = True, actions = None):
    if open_cash_policy not in {"sell_then_buy", "preopen_cash_only"}: raise ValueError("invalid open_cash_policy")
    book = Book(initial_cash); pending = []; target = set(); all_orders = []
    for index, on in enumerate(dates):
        quotes = market[on]; book.start_day(on, (actions or {}).get(on, []), quotes); pre_cash = book.cash
        available = pre_cash
        for order in pending:
            result = book.execute(order, quotes[order["instrument"]], on, rules, slippage, available if open_cash_policy == "preopen_cash_only" else None); all_orders.append(result)
            if open_cash_policy == "preopen_cash_only" and result.get("side") == "buy":
                available -= result.get("amount", 0) + result.get("total", 0)
        pending = []; book.mark_to_market(on, quotes)
        if index % rebalance_every == 0:
            scores = scores_by_date.get(on, {}); held = {i: p.qty for i, p in book.positions.items() if p.qty}
            result = target_list(scores, held, n, buffer, max_sell); target = set(result["target"])
            value = book.equity_rows[-1]["equity"] / max(1, n); amounts = {i: value for i in target}; pending = orders_for_targets(result, held, amounts, True)
        elif refill_between_rebalance:
            scores = scores_by_date.get(on, {})
            held = {i: p.qty for i, p in book.positions.items() if p.qty}
            result = target_list(scores, held, n, buffer, max_sell, candidates = set(scores) | target)
            value = book.equity_rows[-1]["equity"] / max(1, n)
            pending = orders_for_targets(result, held, {i: value for i in result["target"]}, True)
    return book, all_orders

def rerun_scenario(config, **overrides):
    """Run the full loop again with the same frozen inputs and changed costs."""
    options = dict(config); options.update(overrides)
    return run_loop(**options)
