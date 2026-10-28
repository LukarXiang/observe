"""Original state changes differ from actual fills; strict boundaries remain unchanged."""
from scripts import diagnose_strategy_batch102_transitions as study


def test_recovery_from_middle_mode_keeps_old_multiplier():
    first = study.handle(mode=0, times=-1, recovery=True)
    later = study.handle(mode=2, times=3, recovery=True)
    assert first['calls'] == [['value', '510300.XSHG', 40.]] and first['mode'] == 1 and first['times'] == 0
    assert later['calls'] == [] and later['mode'] == 1 and later['times'] == 3


def test_annual_entry_can_add_twice_inside_same_original_callback():
    result = study.handle(mode=0, times=-1, price=80., year_mean=100.)
    assert result['calls'] == [['value', '510300.XSHG', 40.], ['value', '510300.XSHG', 80.]]
    assert (result['mode'], result['times'], result['record']) == (2, 1, 80.)


def test_unfilled_intent_can_still_advance_multiplier_and_record():
    result = study.handle(price=97.9, cash=0.)
    assert result['calls'] == [] and result['times'] == 1 and result['record'] == 97.9


def test_full_cash_right_side_order_depends_on_injected_cash_feedback():
    delayed = study.handle(times=-1, right_side=True)
    immediate = study.handle(times=-1, right_side=True, immediate=True)
    assert delayed['calls'] == [['value', '510300.XSHG', 40.], ['value', '510300.XSHG', 1000.]]
    assert immediate['calls'] == [['value', '510300.XSHG', 40.], ['value', '510300.XSHG', 960.]]


def test_overheat_and_inventory_equalities_are_strict_and_latch_persists():
    assert study.exit_state()['calls'] == []
    latched = study.exit_state(score=80., latched=True, raw=50.)
    assert latched['calls'] == [['inventory_query']] and latched['mode'] == 1 and latched['over_hot']
    assert study.exit_state(score=100., raw=50.)['mode'] == 1


def test_exit_requires_price_above120_and_previous_MA_strictly_above():
    a = study.exit_state(score=100.)
    b = study.exit_state(score=100., index_price=101., prior_cross=False)
    assert a['mode'] == b['mode'] == 2 and a['calls'] == b['calls'] == [['inventory_query']]
