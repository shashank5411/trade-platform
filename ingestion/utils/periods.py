from datetime import datetime, timezone
from typing import Optional


def current_period(granularity: str) -> str:
    now = datetime.now(timezone.utc)
    if granularity == "year":
        return str(now.year)
    return now.strftime("%Y-%m")


def add_periods(period: str, n: int, granularity: str) -> str:
    if granularity == "year":
        return str(int(period[:4]) + n)
    year, month = int(period[:4]), int(period[5:7])
    total = year * 12 + (month - 1) + n
    return f"{total // 12}-{(total % 12) + 1:02d}"


def subtract_periods(period: str, n: int, granularity: str) -> str:
    return add_periods(period, -n, granularity)


def period_range(start: str, end: str, granularity: str) -> list[str]:
    """Return all periods from start to end inclusive."""
    periods, current = [], start
    while current <= end:
        periods.append(current)
        current = add_periods(current, 1, granularity)
    return periods


def resolve_periods(
    args,
    granularity: str,
    overlap: int,
    watermark: Optional[dict],
    default_start: str,
) -> tuple[str, str]:
    """
    Resolve the fetch window.
      1. CLI --start-period / --end-period if both supplied.
      2. Watermark minus overlap window  →  now  (catches late-arriving data).
      3. Config default_start (first run)  →  now.
    """
    if getattr(args, "start_period", None) and getattr(args, "end_period", None):
        return args.start_period, args.end_period
    end = current_period(granularity)
    if watermark:
        start = subtract_periods(watermark["last_ingested_period"], overlap, granularity)
    else:
        start = default_start
    return start, end
