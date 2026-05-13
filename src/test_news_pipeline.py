"""
Offline test for the news sentiment + score blend pipeline.

No internet required. Exercises:
  - compute_sentiment_score() across US / TW / negation / mixed headlines
  - _build_item() market routing
  - news_history persistence (writes to a temp file, not the real one)
  - compute_news_adjustment() bounds + sign convention

Run: python src/test_news_pipeline.py

Each assertion prints ✓/✗ inline; non-zero exit on failure.
"""

from __future__ import annotations

import json
import os
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import news_history  # noqa: E402
from news_fetcher import (  # noqa: E402
    _aggregate_sentiment,
    _build_item,
    compute_sentiment_score,
)


_failures = 0


def check(label: str, ok: bool, detail: str = "") -> None:
    global _failures
    marker = "✓" if ok else "✗"
    msg = f"  {marker} {label}"
    if detail:
        msg += f"    [{detail}]"
    print(msg)
    if not ok:
        _failures += 1


def test_sentiment_polarity() -> None:
    print("\n[sentiment] polarity")
    s_bull = compute_sentiment_score("S&P 500 hits record high on strong earnings", "us")
    s_bear = compute_sentiment_score("Market crash deepens, recession fears mount", "us")
    s_neut = compute_sentiment_score("Federal Reserve meeting scheduled for next week", "us")
    check("bullish > 0", s_bull > 0, f"score={s_bull}")
    check("bearish < 0", s_bear < 0, f"score={s_bear}")
    check("neutral == 0", abs(s_neut) < 1e-9, f"score={s_neut}")

    print("\n[sentiment] TW lexicon")
    s_tw_bull = compute_sentiment_score("台股大漲突破歷史新高 外資買超", "tw")
    s_tw_bear = compute_sentiment_score("台股重挫 賣壓沉重 跌破支撐", "tw")
    s_tw_mix = compute_sentiment_score("台股上漲後拉回 走勢震盪", "tw")
    check("台股大漲 → bullish", s_tw_bull > 0.4, f"score={s_tw_bull}")
    check("台股重挫 → bearish", s_tw_bear < -0.4, f"score={s_tw_bear}")
    check("混合走勢 → small abs", abs(s_tw_mix) < 0.5, f"score={s_tw_mix}")


def test_sentiment_negation() -> None:
    print("\n[sentiment] negation handling")
    s_plain = compute_sentiment_score("台股大漲", "tw")
    s_neg = compute_sentiment_score("台股未大漲", "tw")
    check("'未大漲' flips to bearish", s_neg < 0 < s_plain, f"plain={s_plain} neg={s_neg}")

    s_en_plain = compute_sentiment_score("market surge after Fed cut", "us")
    s_en_neg = compute_sentiment_score("market did not surge after Fed cut", "us")
    check("'did not surge' weaker / flipped", s_en_neg < s_en_plain,
          f"plain={s_en_plain} neg={s_en_neg}")


def test_market_routing() -> None:
    print("\n[fetcher] market routing")
    yahoo = _build_item("2026-03-26T10:00:00", "yahoo", "S&P 500 hits record high")
    anue = _build_item("2026-03-26T10:00:00", "anue", "台股大漲突破新高")
    cna = _build_item("2026-03-26T10:00:00", "cna", "央行不升息 台股反彈")
    jin10 = _build_item("2026-03-26T10:00:00", "jin10", "OPEC announces production cut")
    check("yahoo → us", yahoo["market"] == "us", yahoo["market"])
    check("anue → tw", anue["market"] == "tw", anue["market"])
    check("cna → tw", cna["market"] == "tw", cna["market"])
    check("jin10 → global", jin10["market"] == "global", jin10["market"])
    check("sentiment_score in items", "sentiment_score" in yahoo,
          str(yahoo.get("sentiment_score")))


def test_aggregate() -> None:
    print("\n[fetcher] relevance-weighted aggregate")
    items = [
        {"sentiment_score": 0.8, "relevance_score": 9},
        {"sentiment_score": 0.0, "relevance_score": 3},
        {"sentiment_score": -0.5, "relevance_score": 2},
    ]
    agg = _aggregate_sentiment(items)
    # expected: (0.8*9 + 0 - 0.5*2) / 14 = (7.2 - 1.0) / 14 ≈ 0.443
    check("weighted mean close to 0.443", abs(agg - 0.443) < 0.01, f"agg={agg}")
    check("empty list → 0.0", _aggregate_sentiment([]) == 0.0)


def test_history_and_adjustment() -> None:
    print("\n[history] persistence + score adjustment")
    # Redirect HISTORY_PATH to a temp file so we don't pollute data/.
    with tempfile.TemporaryDirectory() as tmp:
        original = news_history.HISTORY_PATH
        news_history.HISTORY_PATH = Path(tmp) / "news_history.json"
        try:
            payload = {
                "news": {
                    "summary": {
                        "by_market": {
                            "us": {"sentiment": 0.4, "n_items": 10, "avg_relevance": 6.0},
                            "tw": {"sentiment": -0.6, "n_items": 5, "avg_relevance": 5.0},
                            "global": {"sentiment": 0.0, "n_items": 2, "avg_relevance": 4.0},
                        },
                        "overall_sentiment": 0.1,
                    }
                }
            }
            entry = news_history.record_news(payload, date="2026-03-26")
            check("record_news returns entry", isinstance(entry, dict))
            check("history file written", news_history.HISTORY_PATH.exists())

            us_sent, n_us = news_history.get_sentiment_for("us", date="2026-03-26")
            tw_sent, n_tw = news_history.get_sentiment_for("tw", date="2026-03-26")
            check("us sentiment readback", abs(us_sent - 0.4) < 1e-9 and n_us == 10,
                  f"us=({us_sent}, {n_us})")
            check("tw sentiment readback", abs(tw_sent + 0.6) < 1e-9 and n_tw == 5,
                  f"tw=({tw_sent}, {n_tw})")

            # news_weight=0 → adjustment must be 0
            check("news_weight=0 → adj=0",
                  news_history.compute_news_adjustment("us", 0.0, date="2026-03-26") == 0.0)
            # news_weight=0.1, sentiment=0.4 → adj = 0.1 * 0.4 * 10 = 0.4
            adj_us = news_history.compute_news_adjustment("us", 0.1, date="2026-03-26")
            check("us adj ≈ +0.4", abs(adj_us - 0.4) < 1e-9, f"adj={adj_us}")
            adj_tw = news_history.compute_news_adjustment("tw", 0.1, date="2026-03-26")
            check("tw adj ≈ -0.6", abs(adj_tw + 0.6) < 1e-9, f"adj={adj_tw}")

            # Cap: news_weight=1.0, sentiment=-0.6 → adj = -6.0 (no truncation needed)
            adj_cap = news_history.compute_news_adjustment("tw", 1.0, date="2026-03-26")
            check("max-weight adj scales linearly", abs(adj_cap + 6.0) < 1e-9, f"adj={adj_cap}")

            # Empty-market fallback: ask for a market not in history
            news_history.HISTORY_PATH = Path(tmp) / "missing.json"
            adj_none = news_history.compute_news_adjustment("tw", 0.5)
            check("missing history → adj=0", adj_none == 0.0, f"adj={adj_none}")
        finally:
            news_history.HISTORY_PATH = original


def main() -> int:
    print("=" * 60)
    print("news pipeline offline tests")
    print("=" * 60)
    test_sentiment_polarity()
    test_sentiment_negation()
    test_market_routing()
    test_aggregate()
    test_history_and_adjustment()
    print()
    print("=" * 60)
    if _failures:
        print(f"FAILED — {_failures} assertion(s) did not pass")
        return 1
    print("All assertions passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
