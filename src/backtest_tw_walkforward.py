"""
Walk-Forward strategy backtest for TW sentiment score on TAIEX.

Strategy (long-only, state machine):
  - flat → long when score < entry_threshold
  - long → flat when score > exit_threshold
Signal is observed at T close; trade executes at T+1 open (proxied by next-day return).
Transaction cost charged on each round-trip (entry + exit).

Walk-forward:
  - Training window: 3 years (756 trading days)
  - Test window:    1 year (252 trading days)
  - Step:           1 year (rolling, non-overlapping OOS)
  - For each fold, grid-search (entry, exit) on training window by Sharpe,
    then apply chosen thresholds to OOS period.

Output: data/tw_walkforward_backtest.json
  {
    "params": { ... },
    "folds": [ {train_start, train_end, oos_start, oos_end, entry, exit,
                train_sharpe, oos_metrics, oos_dates, oos_strategy_equity,
                oos_bh_equity}, ... ],
    "aggregate": { oos_cagr, oos_sharpe, oos_maxdd, oos_hit_rate,
                   bh_cagr, bh_sharpe, bh_maxdd, n_trades, time_in_market_pct },
    "equity_curve": { dates, strategy, buy_and_hold }
  }

Run: python src/backtest_tw_walkforward.py
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, asdict
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
OVERLAY_PATH = ROOT / "data" / "overlay_data.json"
OUT_PATH = ROOT / "data" / "tw_walkforward_backtest.json"

# ── Backtest configuration ─────────────────────────────────────────────────────
TRAIN_DAYS = 756        # ~3 years of trading days
OOS_DAYS = 252          # ~1 year
STEP_DAYS = 252         # step by 1 year
ROUND_TRIP_COST = 0.005 # 0.5% — TW sell tax 0.3% + commission ~0.1425% rounded up
ENTRY_GRID = [15, 20, 25, 30, 35, 40]
EXIT_GRID = [55, 60, 65, 70, 75, 80, 85]
TRADING_DAYS_PER_YEAR = 252
RISK_FREE_DAILY = 0.0   # ignore RF — relative comparison vs BH

# Minimum training Sharpe needed to accept the grid-searched thresholds;
# otherwise fall back to canonical (30, 70). Prevents picking noise-driven
# corner solutions when the signal is sparse in the training window.
MIN_TRAIN_SHARPE = 0.3
FALLBACK_ENTRY = 30.0
FALLBACK_EXIT = 70.0


def load_aligned_data() -> pd.DataFrame:
    """Load TW score + TWII price into an aligned, sorted DataFrame."""
    with open(OVERLAY_PATH) as f:
        d = json.load(f)
    score_df = pd.DataFrame({"date": d["dates"], "score": d["tw_score"]})
    price_df = pd.DataFrame({"date": d["twii_dates"], "price": d["twii"]})
    score_df["date"] = pd.to_datetime(score_df["date"])
    price_df["date"] = pd.to_datetime(price_df["date"])
    df = score_df.merge(price_df, on="date", how="inner").sort_values("date").reset_index(drop=True)
    df = df.dropna(subset=["score", "price"]).reset_index(drop=True)
    df["ret"] = df["price"].pct_change().fillna(0.0)
    return df


def simulate(scores: np.ndarray, rets: np.ndarray, entry: float, exit_: float,
             cost: float = ROUND_TRIP_COST) -> dict:
    """Simulate long/flat strategy. Signal at T → position from T+1.

    Returns dict with daily_returns (strategy), n_trades, time_in_market.
    """
    n = len(scores)
    position = np.zeros(n, dtype=np.int8)  # position held during day t (applied to ret[t])
    pos = 0
    trades = 0
    for t in range(n):
        # position applied today was decided on yesterday's close
        position[t] = pos
        # decide tomorrow's position from today's signal
        sig = scores[t]
        if pos == 0 and sig < entry:
            pos = 1
        elif pos == 1 and sig > exit_:
            pos = 0
            trades += 1  # count complete round-trips on exit
    strat_ret = position * rets
    # apply round-trip cost on day after exit signal — approximate by subtracting
    # cost*0.5 on entry day and cost*0.5 on exit day. We detect changes in position.
    if n > 1:
        change = np.diff(position, prepend=position[0])
        # entries: change == 1, exits: change == -1
        # apply half-cost on entry, half on exit, so total round-trip cost stays at `cost`
        strat_ret = strat_ret - (np.abs(change) * (cost / 2.0))
    return {
        "daily_returns": strat_ret,
        "position": position,
        "n_trades": int(trades),
        "time_in_market": float(position.mean()),
    }


def metrics(daily_returns: np.ndarray) -> dict:
    """Compute annualized return, Sharpe, max drawdown, hit rate, total return."""
    if len(daily_returns) == 0:
        return dict(cagr=0.0, sharpe=0.0, maxdd=0.0, hit_rate=0.0, total_return=0.0)
    eq = np.cumprod(1.0 + daily_returns)
    total = eq[-1] - 1.0
    years = len(daily_returns) / TRADING_DAYS_PER_YEAR
    cagr = (eq[-1]) ** (1.0 / years) - 1.0 if years > 0 else 0.0
    mu = daily_returns.mean()
    sd = daily_returns.std(ddof=1) if len(daily_returns) > 1 else 0.0
    sharpe = (mu / sd) * np.sqrt(TRADING_DAYS_PER_YEAR) if sd > 1e-12 else 0.0
    peak = np.maximum.accumulate(eq)
    dd = (eq / peak) - 1.0
    maxdd = float(dd.min())
    # hit_rate: fraction of non-zero return days that are positive
    active = daily_returns[daily_returns != 0]
    hit_rate = float((active > 0).mean()) if len(active) > 0 else 0.0
    return dict(
        cagr=float(cagr),
        sharpe=float(sharpe),
        maxdd=float(maxdd),
        hit_rate=hit_rate,
        total_return=float(total),
    )


def grid_search(scores: np.ndarray, rets: np.ndarray) -> tuple[float, float, float, bool]:
    """Search (entry, exit) pair that maximizes training Sharpe.

    Returns (entry, exit, train_sharpe, used_fallback). Falls back to
    (FALLBACK_ENTRY, FALLBACK_EXIT) when no candidate clears MIN_TRAIN_SHARPE.
    """
    best = (-np.inf, ENTRY_GRID[0], EXIT_GRID[-1])
    for entry in ENTRY_GRID:
        for exit_ in EXIT_GRID:
            if exit_ <= entry + 10:
                continue
            sim = simulate(scores, rets, entry, exit_)
            m = metrics(sim["daily_returns"])
            if sim["n_trades"] < 3:
                continue
            score = m["sharpe"]
            if score > best[0]:
                best = (score, entry, exit_)
    train_sharpe, e, x = best
    if not np.isfinite(train_sharpe) or train_sharpe < MIN_TRAIN_SHARPE:
        return FALLBACK_ENTRY, FALLBACK_EXIT, float(train_sharpe if np.isfinite(train_sharpe) else 0.0), True
    return float(e), float(x), float(train_sharpe), False


@dataclass
class FoldResult:
    fold_idx: int
    train_start: str
    train_end: str
    oos_start: str
    oos_end: str
    entry: float
    exit: float
    train_sharpe: float
    used_fallback: bool
    oos_cagr: float
    oos_sharpe: float
    oos_maxdd: float
    oos_total_return: float
    oos_hit_rate: float
    oos_n_trades: int
    oos_time_in_market: float
    bh_cagr: float
    bh_sharpe: float
    bh_maxdd: float
    bh_total_return: float
    static_cagr: float
    static_sharpe: float
    static_maxdd: float
    static_n_trades: int


def walk_forward(df: pd.DataFrame) -> dict:
    n = len(df)
    scores_all = df["score"].to_numpy()
    rets_all = df["ret"].to_numpy()
    dates_all = df["date"].dt.strftime("%Y-%m-%d").to_numpy()

    folds: list[FoldResult] = []
    oos_dates: list[str] = []
    oos_strat_rets: list[float] = []
    oos_bh_rets: list[float] = []
    oos_static_rets: list[float] = []

    start = 0
    fold_idx = 0
    while start + TRAIN_DAYS + OOS_DAYS <= n:
        train_lo, train_hi = start, start + TRAIN_DAYS
        oos_lo, oos_hi = train_hi, min(train_hi + OOS_DAYS, n)
        e, x, train_sharpe, used_fallback = grid_search(
            scores_all[train_lo:train_hi], rets_all[train_lo:train_hi]
        )
        oos_scores = scores_all[oos_lo:oos_hi]
        oos_rets = rets_all[oos_lo:oos_hi]
        sim = simulate(oos_scores, oos_rets, e, x)
        static_sim = simulate(oos_scores, oos_rets, FALLBACK_ENTRY, FALLBACK_EXIT)
        m = metrics(sim["daily_returns"])
        static_m = metrics(static_sim["daily_returns"])
        bh = metrics(oos_rets)
        fold = FoldResult(
            fold_idx=fold_idx,
            train_start=str(dates_all[train_lo]),
            train_end=str(dates_all[train_hi - 1]),
            oos_start=str(dates_all[oos_lo]),
            oos_end=str(dates_all[oos_hi - 1]),
            entry=e,
            exit=x,
            train_sharpe=train_sharpe,
            used_fallback=used_fallback,
            oos_cagr=m["cagr"],
            oos_sharpe=m["sharpe"],
            oos_maxdd=m["maxdd"],
            oos_total_return=m["total_return"],
            oos_hit_rate=m["hit_rate"],
            oos_n_trades=sim["n_trades"],
            oos_time_in_market=sim["time_in_market"],
            bh_cagr=bh["cagr"],
            bh_sharpe=bh["sharpe"],
            bh_maxdd=bh["maxdd"],
            bh_total_return=bh["total_return"],
            static_cagr=static_m["cagr"],
            static_sharpe=static_m["sharpe"],
            static_maxdd=static_m["maxdd"],
            static_n_trades=static_sim["n_trades"],
        )
        folds.append(fold)
        oos_dates.extend(dates_all[oos_lo:oos_hi].tolist())
        oos_strat_rets.extend(sim["daily_returns"].tolist())
        oos_bh_rets.extend(oos_rets.tolist())
        oos_static_rets.extend(static_sim["daily_returns"].tolist())

        start += STEP_DAYS
        fold_idx += 1

    strat_arr = np.array(oos_strat_rets)
    bh_arr = np.array(oos_bh_rets)
    static_arr = np.array(oos_static_rets)
    agg_strat = metrics(strat_arr)
    agg_bh = metrics(bh_arr)
    agg_static = metrics(static_arr)
    total_trades = sum(f.oos_n_trades for f in folds)
    avg_tim = float(np.mean([f.oos_time_in_market for f in folds])) if folds else 0.0
    fallback_pct = float(np.mean([f.used_fallback for f in folds])) if folds else 0.0

    strat_eq = np.cumprod(1.0 + strat_arr).tolist() if len(strat_arr) else []
    bh_eq = np.cumprod(1.0 + bh_arr).tolist() if len(bh_arr) else []
    static_eq = np.cumprod(1.0 + static_arr).tolist() if len(static_arr) else []

    return {
        "params": {
            "train_days": TRAIN_DAYS,
            "oos_days": OOS_DAYS,
            "step_days": STEP_DAYS,
            "round_trip_cost": ROUND_TRIP_COST,
            "entry_grid": ENTRY_GRID,
            "exit_grid": EXIT_GRID,
            "min_train_sharpe": MIN_TRAIN_SHARPE,
            "fallback_entry": FALLBACK_ENTRY,
            "fallback_exit": FALLBACK_EXIT,
            "trading_days_per_year": TRADING_DAYS_PER_YEAR,
            "n_samples": n,
            "date_min": str(dates_all[0]),
            "date_max": str(dates_all[-1]),
        },
        "folds": [asdict(f) for f in folds],
        "aggregate": {
            "oos_cagr": agg_strat["cagr"],
            "oos_sharpe": agg_strat["sharpe"],
            "oos_maxdd": agg_strat["maxdd"],
            "oos_total_return": agg_strat["total_return"],
            "oos_hit_rate": agg_strat["hit_rate"],
            "bh_cagr": agg_bh["cagr"],
            "bh_sharpe": agg_bh["sharpe"],
            "bh_maxdd": agg_bh["maxdd"],
            "bh_total_return": agg_bh["total_return"],
            "static_cagr": agg_static["cagr"],
            "static_sharpe": agg_static["sharpe"],
            "static_maxdd": agg_static["maxdd"],
            "static_total_return": agg_static["total_return"],
            "n_trades_total": total_trades,
            "avg_time_in_market": avg_tim,
            "fallback_fold_pct": fallback_pct,
            "n_folds": len(folds),
            "oos_n_days": len(oos_strat_rets),
        },
        "equity_curve": {
            "dates": oos_dates,
            "strategy": strat_eq,
            "buy_and_hold": bh_eq,
            "static_30_70": static_eq,
        },
    }


def print_summary(result: dict) -> None:
    p = result["params"]
    a = result["aggregate"]
    print()
    print("=" * 72)
    print(f"TW Walk-Forward Backtest — long-only on TAIEX")
    print(f"Sample: {p['date_min']} → {p['date_max']}  ({p['n_samples']} aligned days)")
    print(f"WF: train={p['train_days']}d / OOS={p['oos_days']}d / step={p['step_days']}d")
    print(f"Cost: {p['round_trip_cost']*100:.2f}% round-trip")
    print("-" * 72)
    print(f"{'Fold':>4} {'OOS year':>12} {'Ent':>4} {'Ext':>4} {'FB':>3} "
          f"{'TrSh':>6} {'CAGR':>7} {'Shrp':>6} {'MaxDD':>7} {'#Tr':>4}")
    for f in result["folds"]:
        print(f"{f['fold_idx']:>4d} "
              f"{f['oos_start']:>12} "
              f"{f['entry']:>4.0f} {f['exit']:>4.0f} "
              f"{'Y' if f['used_fallback'] else '.':>3} "
              f"{f['train_sharpe']:>6.2f} "
              f"{f['oos_cagr']*100:>6.1f}% "
              f"{f['oos_sharpe']:>6.2f} "
              f"{f['oos_maxdd']*100:>6.1f}% "
              f"{f['oos_n_trades']:>4d}")
    print("-" * 72)
    print(f"Aggregate OOS ({a['n_folds']} folds, {a['oos_n_days']} days):")
    print(f"  WF strat :  CAGR {a['oos_cagr']*100:6.2f}%  Sharpe {a['oos_sharpe']:.2f}  "
          f"MaxDD {a['oos_maxdd']*100:6.1f}%  Total {a['oos_total_return']*100:7.1f}%")
    print(f"  Static 30/70: CAGR {a['static_cagr']*100:6.2f}%  Sharpe {a['static_sharpe']:.2f}  "
          f"MaxDD {a['static_maxdd']*100:6.1f}%  Total {a['static_total_return']*100:7.1f}%")
    print(f"  BuyHold  :  CAGR {a['bh_cagr']*100:6.2f}%  Sharpe {a['bh_sharpe']:.2f}  "
          f"MaxDD {a['bh_maxdd']*100:6.1f}%  Total {a['bh_total_return']*100:7.1f}%")
    print(f"  Trades total: {a['n_trades_total']}  Avg time-in-market: "
          f"{a['avg_time_in_market']*100:.1f}%  Fallback folds: "
          f"{a['fallback_fold_pct']*100:.0f}%")
    print("=" * 72)


def main() -> None:
    df = load_aligned_data()
    if len(df) < TRAIN_DAYS + OOS_DAYS:
        raise SystemExit(f"Not enough data: {len(df)} rows (need {TRAIN_DAYS + OOS_DAYS})")
    result = walk_forward(df)
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_PATH, "w") as f:
        json.dump(result, f, indent=2)
    print_summary(result)
    print(f"\nWrote {OUT_PATH.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
