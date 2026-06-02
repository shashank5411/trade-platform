from datetime import date, datetime, timedelta
from typing import Optional


def current_date_str() -> str:
    return date.today().isoformat()


def subtract_days(date_str: str, days: int) -> str:
    d = datetime.strptime(date_str, "%Y-%m-%d").date()
    return (d - timedelta(days=days)).isoformat()


def resolve_dates(
    args,
    max_lookback_days: int,
    watermark: Optional[dict],
    default_start: str,
) -> tuple[str, str]:
    """
    Resolve fetch date range.
      1. CLI --start-date / --end-date if both provided.
      2. Watermark minus lookback days → today.
      3. Config default_start (first run) → today.
    """
    if getattr(args, "start_date", None) and getattr(args, "end_date", None):
        return args.start_date, args.end_date
    end = current_date_str()
    if watermark:
        start = subtract_days(watermark["last_ingested_period"], max_lookback_days)
    else:
        start = default_start
    return start, end
