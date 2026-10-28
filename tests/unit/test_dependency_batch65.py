import pandas as pd
import pytest

from scripts.probe_strategy_batch65 import pcf_audit, ratio_audit, shares_audit
from scripts.verify_strategy_batch65 import archive_keys


def test_share_dates_do_not_let_post_window_records_or_missing_announcements_pass():
    frame = pd.DataFrame({'变动日期': ['2024-01-01', '2024-06-30', '2024-01-01'],
                          '公告日期': ['2024-01-02', None, '2023-12-31'], '总股本': ['100', '0', None]})
    result = shares_audit(frame, 'cninfo')
    assert result['dates']['after_requested_end'] == 1
    assert result['announcements']['missing'] == 1
    assert result['announcement_before_change'] == result['announcement_after_change'] == 1
    assert result['duplicate_change_dates'] == 2
    assert result['missing_total'] == result['nonpositive_total'] == 1
    assert result['strict_usable'] is False


def test_ambiguous_share_events_cannot_contribute_a_unit_ratio():
    cninfo = pd.DataFrame({'变动日期': ['2024-01-01', '2024-01-01', '2024-02-01', '2024-03-01'],
                           '总股本': ['1', '2', '3', '0']})
    eastmoney = pd.DataFrame({'变更日期': ['2024-01-01', '2024-02-01', '2024-03-01'],
                              '总股本': ['10000', '30000', '1']})
    result = ratio_audit(cninfo, eastmoney)
    assert result['unambiguous_common_dates'] == 1
    assert result['observed_ratios'] == [10000.0]
    assert result['unit_conversion_authorized'] is False


def test_pcf_audit_keeps_negative_zero_and_missing_distinct():
    frame = pd.DataFrame({'date': ['2024-04-01'] * 5, 'code': ['sh.600837'] * 5,
                          'pcfNcfTTM': ['-2', '0', '5', '', 'not a number']})
    result = pcf_audit(frame)
    assert result['negative_pcf'] == result['zero_pcf'] == result['positive_pcf'] == 1
    assert result['missing_pcf'] == 2
    assert result['duplicate_code_dates'] == 5
    assert result['strict_usable'] is False


@pytest.mark.parametrize('day', ['../outside', '/tmp/other', '20240401', '2024-02-30',
                                 '2024-03-28', '2024-06-29', '2024-04-01/../outside'])
def test_external_pcf_dates_cannot_select_other_archive_paths(day):
    with pytest.raises(ValueError, match='PCF date'):
        archive_keys(pd.DataFrame({'date': [day], 'code': ['sh.600837']}))


def test_external_pcf_security_must_belong_to_the_fixed_request():
    with pytest.raises(ValueError, match='Unexpected PCF security'):
        archive_keys(pd.DataFrame({'date': ['2024-04-01'], 'code': ['sh.600519']}))
    assert archive_keys(pd.DataFrame({'date': ['2024-03-29', '2024-06-28'],
                                      'code': ['sh.600837', 'sz.002594']})) == (
                                          ['2024-03-29', '2024-06-28'], ['sh.600837', 'sz.002594'])
