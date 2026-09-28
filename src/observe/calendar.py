from datetime import date, timedelta


def sessions(start, end, holidays = None):
    excluded = set(holidays or ())
    current = start if isinstance(start, date) else date.fromisoformat(start)
    finish = end if isinstance(end, date) else date.fromisoformat(end)
    while current <= finish:
        if current.weekday() < 5 and current not in excluded: yield current
        current += timedelta(days = 1)
