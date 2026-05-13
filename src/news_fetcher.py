"""
Moodring News Fetcher
=====================
Fetches financial news from Jin10, Yahoo Finance RSS, and CNBC RSS.
Scores each item for relevance, sentiment, impact, and hypothesis mapping.

Usage:
  python news_fetcher.py          # Print JSON to stdout
  from news_fetcher import fetch_news
"""

import json
import re
import logging
from datetime import datetime, timezone
from typing import Optional

logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Keyword definitions
# ---------------------------------------------------------------------------

# Geopolitical / Iran / War
GEO_KEYWORDS = [
    "iran", "伊朗", "war", "戰爭", "military", "軍事", "strike", "攻擊",
    "missile", "飛彈", "airstrike", "空襲", "conflict", "衝突",
    "sanction", "制裁", "hamas", "hezbollah", "中東", "middle east",
    "escalation", "升級", "nuclear", "核武", "drone", "無人機",
    "israel", "以色列", "ukraine", "烏克蘭", "russia", "俄羅斯",
    "north korea", "北韓", "taiwan strait", "台海",
]

# Fed / Monetary policy
FED_KEYWORDS = [
    "fed", "federal reserve", "fomc", "聯準會", "powell", "鮑威爾",
    "rate hike", "rate cut", "升息", "降息", "interest rate", "利率",
    "monetary policy", "貨幣政策", "quantitative", "QE", "QT",
    "tapering", "balance sheet", "資產負債表", "inflation", "通膨",
    "cpi", "pce", "核心通膨", "hawkish", "鷹派", "dovish", "鴿派",
    "dot plot", "fomc minutes", "회의록",
]

# AI / Tech bubble
AI_KEYWORDS = [
    "ai", "artificial intelligence", "人工智慧", "chatgpt", "openai",
    "nvidia", "nvda", "amd", "semiconductor", "半導體", "chip", "晶片",
    "tech bubble", "科技泡沫", "valuation", "估值", "overvalued", "高估",
    "microsoft", "google", "alphabet", "meta", "amazon", "apple",
    "mag7", "magnificent", "growth stock", "成長股", "nasdaq", "那斯達克",
    "hyperscaler", "data center", "資料中心", "llm", "large language",
]

# Oil / Energy
OIL_KEYWORDS = [
    "oil", "原油", "crude", "brent", "wti", "opec", "能源", "energy",
    "petroleum", "gasoline", "天然氣", "natural gas", "lng",
    "oil price", "油價", "refinery", "煉油", "barrel", "桶",
    "shale", "頁岩油", "pipeline", "輸油管", "supply cut", "減產",
    "oil shock", "油價衝擊",
]

# Recession / Jobs / Macro
RECESSION_KEYWORDS = [
    "recession", "衰退", "gdp", "growth", "經濟成長", "slowdown", "放緩",
    "unemployment", "失業", "jobs", "就業", "payroll", "非農", "nonfarm",
    "consumer confidence", "消費者信心", "retail sales", "零售",
    "manufacturing", "製造業", "pmi", "ism", "housing", "房市",
    "yield curve", "殖利率曲線", "inversion", "倒掛", "default", "違約",
    "credit", "信貸", "bank", "銀行", "financial stress", "金融壓力",
    "soft landing", "軟著陸", "hard landing", "硬著陸",
]

# Bullish signals (US / generic English + zh-CN macro). Used for global / US market.
BULLISH_KEYWORDS = [
    "surge", "rally", "gain", "rise", "soar", "jump", "beat", "exceed",
    "record high", "breakout", "upgrade", "buy", "bullish", "optimism",
    "上漲", "大漲", "突破", "創高", "買進", "樂觀", "強勁", "好於預期",
    "recovery", "rebound", "bounce", "positive", "upbeat",
]

# Bearish signals (US / generic English + zh-CN macro).
BEARISH_KEYWORDS = [
    "crash", "plunge", "fall", "drop", "tumble", "sink", "miss", "below",
    "downgrade", "sell", "bearish", "pessimism", "fear", "panic",
    "下跌", "暴跌", "崩盤", "賣出", "悲觀", "恐慌", "疲弱", "差於預期",
    "warning", "risk", "concern", "worry", "threat", "危機", "風險",
]

# ── TW-specific keyword lexicons (繁體中文) ──────────────────────────────────
# Weighted: STRONG terms count as 2, WEAK as 1, so aggregate sentiment
# reflects intensity rather than just direction. Used when market == "tw".
TW_BULLISH_STRONG = [
    "大漲", "暴漲", "飆漲", "飆升", "創新高", "創高", "突破高點",
    "強漲", "噴出", "漲停", "強彈", "多頭", "全面上漲", "拉抬",
    "全面收紅", "齊揚", "創歷史新高", "領漲",
]
TW_BULLISH_WEAK = [
    "上漲", "反彈", "回升", "走高", "走升", "看好", "樂觀",
    "利多", "買盤", "加碼", "翻紅", "上攻", "穩步上揚", "止跌",
    "外資買超", "三大法人買超", "資金流入",
]
TW_BEARISH_STRONG = [
    "大跌", "重挫", "崩跌", "崩盤", "暴跌", "跌停", "殺盤", "套牢",
    "慘跌", "失守", "跌破", "雪崩", "全面下跌", "賣壓沉重", "恐慌",
    "全面收黑", "齊跌", "創新低", "領跌",
]
TW_BEARISH_WEAK = [
    "下跌", "回檔", "拉回", "走低", "看壞", "悲觀", "利空", "賣盤",
    "減碼", "翻黑", "下殺", "疲弱", "走疲", "壓回", "外資賣超",
    "三大法人賣超", "資金流出",
]

# US / generic weighted lists — derived from the existing flat lists so the
# new scorer has a "strong" tier even for English news.
US_BULLISH_STRONG = [
    "surge", "soar", "skyrocket", "rally", "breakout", "record high",
    "all-time high", "blowout", "beat estimates",
]
US_BULLISH_WEAK = [
    "gain", "rise", "jump", "beat", "exceed", "upgrade", "buy", "bullish",
    "optimism", "recovery", "rebound", "bounce", "positive", "upbeat",
    "上漲", "大漲", "突破", "創高", "買進", "樂觀", "強勁", "好於預期",
]
US_BEARISH_STRONG = [
    "crash", "plunge", "collapse", "tumble", "rout", "meltdown",
    "selloff", "panic", "崩盤", "暴跌", "恐慌",
]
US_BEARISH_WEAK = [
    "fall", "drop", "sink", "miss", "below", "downgrade", "sell", "bearish",
    "pessimism", "fear", "warning", "risk", "concern", "worry", "threat",
    "下跌", "賣出", "悲觀", "疲弱", "差於預期", "危機", "風險",
]

# Negation markers — when one appears within a small window BEFORE a bull/bear
# keyword, the keyword's contribution is flipped.
NEGATION_MARKERS = [
    "不", "沒", "未", "無", "非", "並未", "並無", "沒有", "不會", "不再", "難以",
    "not ", "no ", "n't ", "without ", "never ", "hardly ", "scarcely ",
]
NEGATION_WINDOW = 6  # chars (works for both ASCII and CJK)

# High-relevance equity/options terms (boost score)
EQUITY_KEYWORDS = [
    "spy", "s&p 500", "標普", "nasdaq", "道瓊", "dow jones",
    "vix", "volatility", "波動率", "options", "選擇權",
    "futures", "期貨", "market", "股市", "stock", "equity",
    "earnings", "財報", "eps", "revenue", "營收",
]


# ---------------------------------------------------------------------------
# Scoring helpers
# ---------------------------------------------------------------------------

def _text_lower(headline: str, description: str = "") -> str:
    return (headline + " " + description).lower()


def compute_relevance_score(text: str) -> int:
    """Score 1–10 based on US equity/options keyword density."""
    score = 3  # baseline
    t = text.lower()

    hit_equity = sum(1 for kw in EQUITY_KEYWORDS if kw in t)
    hit_geo = sum(1 for kw in GEO_KEYWORDS if kw in t)
    hit_fed = sum(1 for kw in FED_KEYWORDS if kw in t)
    hit_ai = sum(1 for kw in AI_KEYWORDS if kw in t)
    hit_oil = sum(1 for kw in OIL_KEYWORDS if kw in t)
    hit_rec = sum(1 for kw in RECESSION_KEYWORDS if kw in t)

    score += min(hit_equity * 2, 4)
    score += min((hit_geo + hit_fed + hit_ai + hit_oil + hit_rec), 3)

    return min(max(score, 1), 10)


def _has_negation_before(t: str, kw_start: int) -> bool:
    """Return True if any NEGATION_MARKERS occurs in the NEGATION_WINDOW
    characters immediately before kw_start. Operates on lowercased text."""
    lo = max(0, kw_start - NEGATION_WINDOW)
    window = t[lo:kw_start]
    return any(m in window for m in NEGATION_MARKERS)


def _weighted_hits(t: str, strong_kws: list, weak_kws: list) -> tuple:
    """Return (plain_total, negated_total) of weighted keyword hits.

    STRONG keywords contribute 2.0 each; WEAK 1.0. A negation marker within
    NEGATION_WINDOW chars before the keyword routes that hit to the negated
    bucket (caller treats it as contributing to the opposite polarity)."""
    plain = 0.0
    negated = 0.0
    for kw, w in [(k, 2.0) for k in strong_kws] + [(k, 1.0) for k in weak_kws]:
        idx = 0
        while True:
            pos = t.find(kw, idx)
            if pos == -1:
                break
            if _has_negation_before(t, pos):
                negated += w
            else:
                plain += w
            idx = pos + len(kw)
    return plain, negated


def compute_sentiment_score(text: str, market: str = "us") -> float:
    """Return a continuous sentiment score in [-1, 1] for the given text.

    Positive = bullish (greedy), negative = bearish (fearful). The score is
    normalised so volume alone doesn't dominate:
        score = (bull - bear) / (bull + bear)

    Negated terms are routed to the opposite polarity bucket — e.g. "未上漲"
    (did not rise) contributes to the bear bucket, not the bull bucket.

    `market` selects the keyword lexicon. "tw" uses the Traditional Chinese
    weighted lexicons (with English/zh-CN macro terms at half weight);
    anything else falls back to US/generic English.
    """
    t = text.lower()
    if market == "tw":
        bull_plain, bull_neg = _weighted_hits(t, TW_BULLISH_STRONG, TW_BULLISH_WEAK)
        bear_plain, bear_neg = _weighted_hits(t, TW_BEARISH_STRONG, TW_BEARISH_WEAK)
        # Generic macro terms at half weight — TW headlines often borrow them
        # ("Fed 升息打擊台股") and we don't want to miss the polarity.
        us_bull_plain, us_bull_neg = _weighted_hits(t, US_BULLISH_STRONG, US_BULLISH_WEAK)
        us_bear_plain, us_bear_neg = _weighted_hits(t, US_BEARISH_STRONG, US_BEARISH_WEAK)
        bull_plain += 0.5 * us_bull_plain; bull_neg += 0.5 * us_bull_neg
        bear_plain += 0.5 * us_bear_plain; bear_neg += 0.5 * us_bear_neg
    else:
        bull_plain, bull_neg = _weighted_hits(t, US_BULLISH_STRONG, US_BULLISH_WEAK)
        bear_plain, bear_neg = _weighted_hits(t, US_BEARISH_STRONG, US_BEARISH_WEAK)

    # Negated bull terms count as bearish; negated bear terms count as bullish.
    bull = bull_plain + bear_neg
    bear = bear_plain + bull_neg
    denom = bull + bear
    if denom < 1e-9:
        return 0.0
    return float(max(-1.0, min(1.0, (bull - bear) / denom)))


def compute_sentiment(text: str, market: str = "us") -> str:
    """Categorical wrapper kept for backward compatibility with existing code."""
    s = compute_sentiment_score(text, market=market)
    if s > 0.2:
        return "bullish"
    if s < -0.2:
        return "bearish"
    return "neutral"


def compute_impact(text: str, sentiment: str) -> dict:
    """Estimate rough SPY % move and VIX point change."""
    t = text.lower()

    # Base magnitude from relevance
    rel = compute_relevance_score(text)
    base_spy = round(rel * 0.06, 2)   # up to ~0.6% per point
    base_vix = round(rel * 0.25, 2)   # up to ~2.5 pts

    # Amplify for high-impact themes
    if any(kw in t for kw in ["war", "戰爭", "strike", "攻擊", "nuclear", "核武", "missile"]):
        base_spy *= 1.8
        base_vix *= 2.0
    if any(kw in t for kw in ["recession", "衰退", "crash", "崩盤"]):
        base_spy *= 1.5
        base_vix *= 1.6
    if any(kw in t for kw in ["rate hike", "升息", "hawkish", "鷹派"]):
        base_spy *= 1.3
        base_vix *= 1.2

    spy_sign = -1 if sentiment == "bearish" else (1 if sentiment == "bullish" else 0)
    vix_sign = 1 if sentiment == "bearish" else (-1 if sentiment == "bullish" else 0)

    return {
        "spy": round(spy_sign * base_spy, 2),
        "vix": round(vix_sign * base_vix, 2),
    }


def compute_hypotheses(text: str) -> list:
    """Map to H1–H5 hypotheses."""
    t = text.lower()
    hyps = []
    if any(kw in t for kw in GEO_KEYWORDS):
        hyps.append("H1")
    if any(kw in t for kw in FED_KEYWORDS):
        hyps.append("H2")
    if any(kw in t for kw in AI_KEYWORDS):
        hyps.append("H3")
    if any(kw in t for kw in OIL_KEYWORDS):
        hyps.append("H4")
    if any(kw in t for kw in RECESSION_KEYWORDS):
        hyps.append("H5")
    return hyps


def compute_category(text: str) -> str:
    t = text.lower()
    geo_hits = sum(1 for kw in GEO_KEYWORDS if kw in t)
    fed_hits = sum(1 for kw in FED_KEYWORDS if kw in t)
    ai_hits = sum(1 for kw in AI_KEYWORDS if kw in t)
    oil_hits = sum(1 for kw in OIL_KEYWORDS if kw in t)
    rec_hits = sum(1 for kw in RECESSION_KEYWORDS if kw in t)

    # Earnings detection
    if any(kw in t for kw in ["earnings", "eps", "revenue", "財報", "q1", "q2", "q3", "q4", "quarterly"]):
        return "earnings"

    scores = {
        "geopolitical": geo_hits,
        "policy": fed_hits,
        "technical": ai_hits,
        "macro": rec_hits + oil_hits,
    }
    best = max(scores, key=scores.get)
    if scores[best] == 0:
        return "macro"
    return best


# Maps a news source to the market its headlines primarily concern.
SOURCE_MARKET = {
    "yahoo": "us",
    "cnbc": "us",
    "jin10": "global",
    "anue": "tw",
    "cna": "tw",
}


def _build_item(
    time_str: str,
    source: str,
    headline: str,
    description: str = "",
    market: Optional[str] = None,
) -> dict:
    text = headline + " " + description
    mkt = market or SOURCE_MARKET.get(source, "us")
    score = compute_sentiment_score(text, market=mkt)
    sentiment = "bullish" if score > 0.2 else ("bearish" if score < -0.2 else "neutral")
    return {
        "time": time_str,
        "source": source,
        "market": mkt,
        "headline": headline,
        "relevance_score": compute_relevance_score(text),
        "sentiment": sentiment,
        "sentiment_score": round(score, 3),
        "impact": compute_impact(text, sentiment),
        "related_hypotheses": compute_hypotheses(text),
        "category": compute_category(text),
    }


# ---------------------------------------------------------------------------
# Source: Yahoo Finance RSS
# ---------------------------------------------------------------------------

def _fetch_yahoo() -> list:
    try:
        import feedparser  # type: ignore
    except ImportError:
        logger.warning("feedparser not installed; skipping Yahoo Finance")
        return []

    url = "https://finance.yahoo.com/news/rssindex"
    try:
        feed = feedparser.parse(url)
        items = []
        for entry in feed.entries[:30]:
            headline = entry.get("title", "").strip()
            description = entry.get("summary", "")
            if not headline:
                continue
            # Parse publish time
            published = entry.get("published_parsed")
            if published:
                ts = datetime(*published[:6], tzinfo=timezone.utc)
                time_str = ts.strftime("%Y-%m-%dT%H:%M:%S")
            else:
                time_str = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S")
            items.append(_build_item(time_str, "yahoo", headline, description))
        return items
    except Exception as e:
        logger.warning(f"Yahoo Finance fetch failed: {e}")
        return []


# ---------------------------------------------------------------------------
# Source: CNBC RSS
# ---------------------------------------------------------------------------

def _fetch_cnbc() -> list:
    try:
        import feedparser  # type: ignore
    except ImportError:
        logger.warning("feedparser not installed; skipping CNBC")
        return []

    url = "https://www.cnbc.com/id/100003114/device/rss/rss.html"
    try:
        feed = feedparser.parse(url)
        items = []
        for entry in feed.entries[:30]:
            headline = entry.get("title", "").strip()
            description = entry.get("summary", "")
            if not headline:
                continue
            published = entry.get("published_parsed")
            if published:
                ts = datetime(*published[:6], tzinfo=timezone.utc)
                time_str = ts.strftime("%Y-%m-%dT%H:%M:%S")
            else:
                time_str = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S")
            items.append(_build_item(time_str, "cnbc", headline, description))
        return items
    except Exception as e:
        logger.warning(f"CNBC fetch failed: {e}")
        return []


# ---------------------------------------------------------------------------
# Source: Jin10 (金十數據)
# ---------------------------------------------------------------------------

def _fetch_jin10() -> list:
    """
    Attempt to scrape Jin10 flash news from datacenter.jin10.com.
    Falls back to HTML scraping of jin10.com/flash if API fails.
    Gracefully skips on any error.
    """
    try:
        import requests  # type: ignore
        from bs4 import BeautifulSoup  # type: ignore
    except ImportError:
        logger.warning("requests/beautifulsoup4 not installed; skipping Jin10")
        return []

    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/122.0.0.0 Safari/537.36"
        ),
        "Accept-Language": "zh-TW,zh;q=0.9,en;q=0.8",
        "Referer": "https://www.jin10.com/",
    }

    # --- Attempt 1: datacenter API ---
    try:
        api_url = "https://datacenter.jin10.com/flash_newest/0"
        resp = requests.get(api_url, headers=headers, timeout=8)
        resp.raise_for_status()
        data = resp.json()
        items = []
        entries = data if isinstance(data, list) else data.get("data", [])
        for entry in entries[:30]:
            headline = entry.get("title", "") or entry.get("content", "")
            headline = re.sub(r"<[^>]+>", "", headline).strip()
            if not headline:
                continue
            ts_raw = entry.get("time", entry.get("created_at", ""))
            try:
                ts = datetime.fromisoformat(ts_raw.replace("Z", "+00:00"))
                time_str = ts.strftime("%Y-%m-%dT%H:%M:%S")
            except Exception:
                time_str = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S")
            items.append(_build_item(time_str, "jin10", headline))
        if items:
            return items
    except Exception as e:
        logger.info(f"Jin10 datacenter API failed ({e}), trying HTML scrape")

    # --- Attempt 2: HTML scraping of jin10.com/flash ---
    try:
        html_url = "https://www.jin10.com/flash_newest.html"
        resp = requests.get(html_url, headers=headers, timeout=10)
        resp.raise_for_status()
        soup = BeautifulSoup(resp.text, "html.parser")

        items = []
        # Jin10 flash items are typically in <div class="jin-flash-item"> or similar
        for tag in soup.find_all(["div", "li"], class_=re.compile(r"flash|item|news")):
            text_el = tag.find(["p", "span", "a"])
            if not text_el:
                continue
            headline = text_el.get_text(strip=True)
            if len(headline) < 5:
                continue
            # Try to find timestamp
            time_el = tag.find(["time", "span"], class_=re.compile(r"time|date"))
            if time_el:
                raw_time = time_el.get_text(strip=True)
                try:
                    # Format: HH:MM or YYYY-MM-DD HH:MM
                    if re.match(r"^\d{2}:\d{2}$", raw_time):
                        today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
                        time_str = f"{today}T{raw_time}:00"
                    else:
                        ts = datetime.fromisoformat(raw_time)
                        time_str = ts.strftime("%Y-%m-%dT%H:%M:%S")
                except Exception:
                    time_str = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S")
            else:
                time_str = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S")

            items.append(_build_item(time_str, "jin10", headline))
            if len(items) >= 20:
                break

        return items
    except Exception as e:
        logger.warning(f"Jin10 HTML scrape failed: {e}")
        return []


# ---------------------------------------------------------------------------
# Source: Anue 鉅亨網 (TW)
# ---------------------------------------------------------------------------

def _fetch_anue() -> list:
    """Fetch Taiwan equity headlines from Anue (鉅亨網) RSS."""
    try:
        import feedparser  # type: ignore
    except ImportError:
        logger.warning("feedparser not installed; skipping Anue")
        return []

    # Anue exposes per-category RSS feeds. tw_stock = 台股, headline = 焦點新聞.
    urls = [
        "https://news.cnyes.com/rss/cat/tw_stock",
        "https://news.cnyes.com/rss/cat/headline",
    ]
    items: list = []
    for url in urls:
        try:
            feed = feedparser.parse(url)
            for entry in feed.entries[:25]:
                headline = entry.get("title", "").strip()
                description = entry.get("summary", "")
                if not headline:
                    continue
                published = entry.get("published_parsed")
                if published:
                    ts = datetime(*published[:6], tzinfo=timezone.utc)
                    time_str = ts.strftime("%Y-%m-%dT%H:%M:%S")
                else:
                    time_str = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S")
                items.append(_build_item(time_str, "anue", headline, description, market="tw"))
        except Exception as e:
            logger.warning(f"Anue fetch failed for {url}: {e}")
    return items


# ---------------------------------------------------------------------------
# Source: CNA 中央社 (TW)
# ---------------------------------------------------------------------------

def _fetch_cna() -> list:
    """Fetch Taiwan financial headlines from CNA (中央社) RSS."""
    try:
        import feedparser  # type: ignore
    except ImportError:
        logger.warning("feedparser not installed; skipping CNA")
        return []

    url = "https://feeds.feedburner.com/rsscna/finance"
    try:
        feed = feedparser.parse(url)
        items = []
        for entry in feed.entries[:30]:
            headline = entry.get("title", "").strip()
            description = entry.get("summary", "")
            if not headline:
                continue
            published = entry.get("published_parsed")
            if published:
                ts = datetime(*published[:6], tzinfo=timezone.utc)
                time_str = ts.strftime("%Y-%m-%dT%H:%M:%S")
            else:
                time_str = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S")
            items.append(_build_item(time_str, "cna", headline, description, market="tw"))
        return items
    except Exception as e:
        logger.warning(f"CNA fetch failed: {e}")
        return []


# ---------------------------------------------------------------------------
# Summary computation
# ---------------------------------------------------------------------------

def _compute_top_theme(items: list) -> str:
    """Pick top theme label in 繁體中文 based on hypothesis frequency."""
    from collections import Counter
    hyp_counts: Counter = Counter()
    for item in items:
        for h in item.get("related_hypotheses", []):
            hyp_counts[h] += 1

    if not hyp_counts:
        return "一般市場動態"

    top = hyp_counts.most_common(1)[0][0]
    mapping = {
        "H1": "地緣政治風險",
        "H2": "聯準會政策轉向",
        "H3": "AI科技泡沫",
        "H4": "油價衝擊",
        "H5": "經濟衰退疑慮",
    }
    return mapping.get(top, "一般市場動態")


def _aggregate_sentiment(items: list) -> float:
    """Relevance-weighted mean sentiment_score over items. Returns 0 if empty."""
    if not items:
        return 0.0
    num = sum(i.get("sentiment_score", 0.0) * max(i.get("relevance_score", 1), 1) for i in items)
    den = sum(max(i.get("relevance_score", 1), 1) for i in items)
    return round(num / den, 3) if den else 0.0


def _build_summary(items: list) -> dict:
    bullish = sum(1 for i in items if i["sentiment"] == "bullish")
    bearish = sum(1 for i in items if i["sentiment"] == "bearish")
    avg_rel = (
        round(sum(i["relevance_score"] for i in items) / len(items), 1)
        if items
        else 0.0
    )
    # Per-market aggregate sentiment so the score blender can pick the right
    # input for US vs TW dashboards. "global" covers Jin10-style macro items.
    by_market: dict = {}
    for mkt in ("us", "tw", "global"):
        mkt_items = [i for i in items if i.get("market") == mkt]
        by_market[mkt] = {
            "sentiment": _aggregate_sentiment(mkt_items),
            "n_items": len(mkt_items),
            "avg_relevance": (
                round(sum(i["relevance_score"] for i in mkt_items) / len(mkt_items), 1)
                if mkt_items else 0.0
            ),
        }
    return {
        "bullish_count": bullish,
        "bearish_count": bearish,
        "neutral_count": len(items) - bullish - bearish,
        "total_items": len(items),
        "avg_relevance": avg_rel,
        "top_theme": _compute_top_theme(items),
        "by_market": by_market,
        "overall_sentiment": _aggregate_sentiment(items),
    }


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def fetch_news() -> dict:
    """Fetch news from all sources (US + TW + global) and aggregate."""
    all_items = []

    jin10_items = _fetch_jin10()
    yahoo_items = _fetch_yahoo()
    cnbc_items = _fetch_cnbc()
    anue_items = _fetch_anue()
    cna_items = _fetch_cna()

    all_items.extend(jin10_items)
    all_items.extend(yahoo_items)
    all_items.extend(cnbc_items)
    all_items.extend(anue_items)
    all_items.extend(cna_items)

    # Sort by relevance descending, then by time descending
    all_items.sort(key=lambda x: (-x["relevance_score"], x["time"]), reverse=False)
    # Deduplicate by headline similarity (exact match only)
    seen_headlines: set = set()
    deduped = []
    for item in all_items:
        norm = re.sub(r"\s+", " ", item["headline"].lower().strip())
        if norm not in seen_headlines:
            seen_headlines.add(norm)
            deduped.append(item)

    updated_at = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S")

    return {
        "news": {
            "items": deduped,
            "summary": _build_summary(deduped),
            "source_counts": {
                "jin10": len(jin10_items),
                "yahoo": len(yahoo_items),
                "cnbc": len(cnbc_items),
                "anue": len(anue_items),
                "cna": len(cna_items),
            },
            "updated_at": updated_at,
        }
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import subprocess
    import sys

    # Auto-install dependencies if missing
    for pkg in ["feedparser", "beautifulsoup4", "requests"]:
        try:
            __import__(pkg.replace("-", "_").replace("beautifulsoup4", "bs4"))
        except ImportError:
            print(f"Installing {pkg}...", file=sys.stderr)
            subprocess.check_call([
                sys.executable, "-m", "pip", "install",
                "--break-system-packages", "--quiet", pkg,
            ])

    result = fetch_news()
    print(json.dumps(result, ensure_ascii=False, indent=2))
