from datetime import date

import numpy as np
import pandas as pd

from observe.features import FIELDS, panel


def test_selected_fields_preserve_default_panel_values_and_pause_mask():
    days = [date(2024, 1, 2), date(2024, 1, 3), date(2024, 1, 4)]
    view = pd.DataFrame({'date': days[:2], 'instrument': ['600001.SH'] * 2,
                         'is_trading': [True, False], 'pb_mrq': [2., 3.], 'close_adj': [10., 11.]})
    original = panel(view, days, ['600002.SH', '600001.SH'])
    selected = panel(view, days, ['600002.SH', '600001.SH'], fields = ('pb_mrq', 'pe_ttm'))
    assert set(original) == set(FIELDS) and list(selected) == ['pb_mrq', 'pe_ttm']
    for name, values in selected.items(): pd.testing.assert_frame_equal(values, original[name])
    assert selected['pb_mrq'].at[days[0], '600001.SH'] == 2
    assert np.isnan(selected['pb_mrq'].at[days[1], '600001.SH'])
    assert selected['pb_mrq'].loc[days[2]].isna().all()
