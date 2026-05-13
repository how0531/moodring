"""
News sentiment persistence + score-blend helpers.

Daily aggregate sentiment (per market) is appended to data/news_history.json
so the moodring scorer can:
  1. apply today's sentiment as a small adjustment to compute_score
  2. accumulate history over time for IC validation once enough days exist

History schema (data/news_history.json):
    {
      "2026-03-26": {
        "us":     {"sentiment": 0.15, "n_items": 24, "avg_relevance": 6.2},
        "tw":     {"sentiment": -0.32, "n_items": 18, "avg_relevance": 5.4},
        "global": {"sentiment": 0.05, "n_items": 30, "avg_relevance": 5.9},
        "overall": 0.05,
        "updated_at": "2026-03-26T22:01:14"
      },
      ...
    }

This module purposely has no internet I/O — it reads news output produced by
src/news_fetcher.py and stores aggregates on disk. Designed to be importable
from daily_update.py without pulling network dependencies.
"""

from __future__ import annotations

import json
import os
from datetime import datetime
from pathlib import Path
from typing import Optional

ROOT = Path(__file__).resolve().parent.parent
HISTORY_PATH = ROOT / "data" / "news_history.json"

# Maximum adjustment, in score points, when news_weight == 1.0 and aggregate
# sentiment is at the extremes (±1). With default news_weight=0.1 the cap is
# ±1 point — deliberately small so news cannot override the technical core.
NEWS_ADJUSTMENT_SCALE = 10.0


def load_history() -> dict:
    if not HISTORY_PATH.exists():
        return {}
    try:
        with open(HISTORY_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError):
        return {}


def save_history(history: dict) -> None:
    HISTORY_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(HISTORY_PATH, "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2, ensure_ascii=False)


def record_news(news_payload: dict, date: Optional[str] = None) -> dict:
    """Persist today's per-market aggregate sentiment derived from fetch_news().

    Expects the structure returned by news_fetcher.fetch_news(): a dict with a
    top-level "news" key whose value contains "summary.by_market" and
    "summary.overall_sentiment". Returns the entry written.
    """
    summary = news_payload.get("news", {}).get("summary", {}) or {}
    by_mkt = summary.get("by_market", {}) or {}
    overall = summary.get("overall_sentiment", 0.0)
    today = date or datetime.now().strftime("%Y-%m-%d")

    entry = {
        "us": by_mkt.get("us", {"sentiment": 0.0, "n_items": 0, "avg_relevance": 0.0}),
        "tw": by_mkt.get("tw", {"sentiment": 0.0, "n_items": 0, "avg_relevance": 0.0}),
        "global": by_mkt.get("global", {"sentiment": 0.0, "n_items": 0, "avg_relevance": 0.0}),
        "overall": overall,
        "updated_at": datetime.now().isoformat(timespec="seconds"),
    }

    history = load_history()
    history[today] = entry
    save_history(history)
    return entry


def get_sentiment_for(market_key: str, date: Optional[str] = None,
                     fallback_days: int = 3) -> tuple[float, int]:
    """Return (sentiment, n_items) for the given market_key on `date` (today by
    default). When the requested date has 0 items, walk backwards up to
    fallback_days to find the most recent non-empty reading.

    Returns (0.0, 0) when nothing is available — caller should treat that as
    "no news adjustment today" rather than a bearish signal.
    """
    history = load_history()
    if not history:
        return 0.0, 0
    today = date or datetime.now().strftime("%Y-%m-%d")
    dates_desc = sorted([d for d in history.keys() if d <= today], reverse=True)
    checked = 0
    for d in dates_desc:
        if checked > fallback_days:
            break
        mkt_key = market_key if market_key in ("us", "tw", "global") else "global"
        entry = history[d].get(mkt_key, {})
        n = int(entry.get("n_items", 0) or 0)
        if n > 0:
            return float(entry.get("sentiment", 0.0) or 0.0), n
        checked += 1
    return 0.0, 0


def compute_news_adjustment(market_key: str, news_weight: float,
                            date: Optional[str] = None) -> float:
    """Return an additive adjustment in score points for compute_score().

    Positive sentiment (bullish news) raises the score (more greed). Magnitude
    is clamped to ±|news_weight| * NEWS_ADJUSTMENT_SCALE so news cannot swing
    the score by more than the configured cap.

    Returns 0.0 if news_weight <= 0, no history exists, or today's market has
    no news items.
    """
    if not news_weight or news_weight <= 0:
        return 0.0
    sentiment, n_items = get_sentiment_for(market_key, date=date)
    if n_items == 0:
        return 0.0
    return float(news_weight * sentiment * NEWS_ADJUSTMENT_SCALE)


if __name__ == "__main__":
    # CLI: dump today's per-market readings (or all when --all is passed)
    import sys
    history = load_history()
    if not history:
        print("No history yet.")
        sys.exit(0)
    if "--all" in sys.argv:
        print(json.dumps(history, indent=2, ensure_ascii=False))
    else:
        today = datetime.now().strftime("%Y-%m-%d")
        latest_key = sorted(history.keys())[-1]
        key = today if today in history else latest_key
        print(f"[{key}]  (latest in history)")
        print(json.dumps(history[key], indent=2, ensure_ascii=False))
