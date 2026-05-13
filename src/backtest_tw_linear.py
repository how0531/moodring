"""
Walk-Forward backtest of TW sentiment score on TAIEX using CONTINUOUS linear
position sizing (no entry/exit thresholds).

Position mapping (long-only, applied with 1-day signal lag):
    target_position(score) = clip((upper - score) / (upper - lower), 0.0, 1.0)

  score <= lower  → position 1.0 (full long)
  score >= upper  → position 0.0 (flat)
  in between      → linear interpolation

Transaction cost: charged on absolute daily position change, scaled to a
round-trip equivalent. With ROUND_TRIP_COST = 0.5%, each unit of |Δpos|
costs 0.25% (one-way), so a full round-trip (0→1→0) costs 0.5%.

Walk-forward:
  - Training window: 3y (756d); Test: 1y (252d); Step: 1y
  - Grid-searched (lower, upper) on training Sharpe; min training Sharpe
    threshold falls back to canonical (30, 70)

Output: data/tw_walkforward_linear_backtest.json

Run: python src/backtest_tw_linear.py
"""

from __future__ import annotations

import json
from dataclasses import dataclass, asdict
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
OVERLAY_PATH = ROOT / "data" / "overlay_data.json"
OUT_PATH = ROOT / "data" / "tw_walkforward_linear_backtest.json"

TRAIN_DAYS = 756
OOS_DAYS = 252
STEP_DAYS = 252
ROUND_TRIP_COST = 0.005
TRADING_DAYS_PER_YEAR = 252

LOWER_GRID = [15, 20, 25, 30]
UPPER_GRID = [55, 60, 65, 70, 75, 80]
MIN_GAP = 25     # require upper - lower >= 25 for a meaningful sloped curve

# Smoothing window applied to the raw score before sizing. Reduces churn from
# day-to-day noise — TW score has ~1.5 pt daily std which causes heavy turnover
# at zero smoothing. 5 trading days is roughly one week.
SCORE_SMOOTH_WINDOW = 5

MIN_TRAIN_SHARPE = 0.3
FALLBACK_LOWER = 30.0
FALLBACK_UPPER = 70.0


def load_aligned_data() -> pd.DataFrame:
    with open(OVERLAY_PATH) as f:
        d = json.load(f)
    sdf = pd.DataFrame({"date": d["dates"], "score": d["tw_score"]})
    pdf_ = pd.DataFrame({"date": d["twii_dates"], "price": d["twii"]})
    sdf["date"] = pd.to_datetime(sdf["date"])
    pdf_["date"] = pd.to_datetime(pdf_["date"])
    df = sdf.merge(pdf_, on="date", how="inner").sort_values("date").reset_index(drop=True)
    df = df.dropna(subset=["score", "price"]).reset_index(drop=True)
    df["ret"] = df["price"].pct_change().fillna(0.0)
    if SCORE_SMOOTH_WINDOW > 1:
        df["score"] = df["score"].rolling(SCORE_SMOOTH_WINDOW, min_periods=1).mean()
    return df


def target_position(scores: np.ndarray, lower: float, upper: float) -> np.ndarray:
    return np.clip((upper - scores) / (upper - lower), 0.0, 1.0)


def simulate(scores: np.ndarray, rets: np.ndarray, lower: float, upper: float,
             cost: float = ROUND_TRIP_COST) -> dict:
    """Simulate continuous-sizing strategy with 1-day signal lag."""
    n = len(scores)
    target = target_position(scores, lower, upper)
    # position applied to ret[t] is set by signal at t-1 (lag 1)
    position = np.concatenate(([0.0], target[:-1]))
    # cost: |Δposition| * (cost / 2) so 0→1→0 totals one full round-trip cost
    delta = np.abs(np.diff(position, prepend=position[0]))
    strat_ret = position * rets - delta * (cost / 2.0)
    # n_trades proxy: full units of |Δposition| (0→1 counts as 1 entry, etc.)
    turnover = float(np.sum(delta))
    return {
        "daily_returns": strat_ret,
        "position": position,
        "turnover": turnover,
        "time_in_market": float((position > 0).mean()),
        "avg_position": float(position.mean()),
    }


def metrics(daily_returns: np.ndarray) -> dict:
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
    active = daily_returns[daily_returns != 0]
    hit_rate = float((active > 0).mean()) if len(active) > 0 else 0.0
    return dict(
        cagr=float(cagr),
        sharpe=float(sharpe),
        maxdd=float(dd.min()),
        hit_rate=hit_rate,
        total_return=float(total),
    )


def grid_search(scores: np.ndarray, rets: np.ndarray) -> tuple[float, float, float, bool]:
    best = (-np.inf, FALLBACK_LOWER, FALLBACK_UPPER)
    for lo in LOWER_GRID:
        for up in UPPER_GRID:
            if up - lo < MIN_GAP:
                continue
            sim = simulate(scores, rets, lo, up)
            # require non-trivial activity to avoid zero-turnover corner solutions
            if sim["turnover"] < 0.5:
                continue
            m = metrics(sim["daily_returns"])
            if m["sharpe"] > best[0]:
                best = (m["sharpe"], lo, up)
    sharpe, lo, up = best
    if not np.isfinite(sharpe) or sharpe < MIN_TRAIN_SHARPE:
        return FALLBACK_LOWER, FALLBACK_UPPER, float(sharpe if np.isfinite(sharpe) else 0.0), True
    return float(lo), float(up), float(sharpe), False


@dataclass
class FoldResult:
    fold_idx: int
    train_start: str
    train_end: str
    oos_start: str
    oos_end: str
    lower: float
    upper: float
    train_sharpe: float
    used_fallback: bool
    oos_cagr: float
    oos_sharpe: float
    oos_maxdd: float
    oos_total_return: float
    oos_hit_rate: float
    oos_turnover: float
    oos_time_in_market: float
    oos_avg_position: float
    bh_cagr: float
    bh_sharpe: float
    bh_maxdd: float
    bh_total_return: float
    static_cagr: float
    static_sharpe: float
    static_maxdd: float


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
        lo, up, train_sharpe, used_fallback = grid_search(
            scores_all[train_lo:train_hi], rets_all[train_lo:train_hi]
        )
        oos_scores = scores_all[oos_lo:oos_hi]
        oos_rets = rets_all[oos_lo:oos_hi]
        sim = simulate(oos_scores, oos_rets, lo, up)
        static_sim = simulate(oos_scores, oos_rets, FALLBACK_LOWER, FALLBACK_UPPER)
        m = metrics(sim["daily_returns"])
        static_m = metrics(static_sim["daily_returns"])
        bh = metrics(oos_rets)
        folds.append(FoldResult(
            fold_idx=fold_idx,
            train_start=str(dates_all[train_lo]),
            train_end=str(dates_all[train_hi - 1]),
            oos_start=str(dates_all[oos_lo]),
            oos_end=str(dates_all[oos_hi - 1]),
            lower=lo,
            upper=up,
            train_sharpe=train_sharpe,
            used_fallback=used_fallback,
            oos_cagr=m["cagr"],
            oos_sharpe=m["sharpe"],
            oos_maxdd=m["maxdd"],
            oos_total_return=m["total_return"],
            oos_hit_rate=m["hit_rate"],
            oos_turnover=sim["turnover"],
            oos_time_in_market=sim["time_in_market"],
            oos_avg_position=sim["avg_position"],
            bh_cagr=bh["cagr"],
            bh_sharpe=bh["sharpe"],
            bh_maxdd=bh["maxdd"],
            bh_total_return=bh["total_return"],
            static_cagr=static_m["cagr"],
            static_sharpe=static_m["sharpe"],
            static_maxdd=static_m["maxdd"],
        ))
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
    total_turnover = float(sum(f.oos_turnover for f in folds))
    avg_tim = float(np.mean([f.oos_time_in_market for f in folds])) if folds else 0.0
    avg_pos = float(np.mean([f.oos_avg_position for f in folds])) if folds else 0.0
    fallback_pct = float(np.mean([f.used_fallback for f in folds])) if folds else 0.0

    strat_eq = np.cumprod(1.0 + strat_arr).tolist() if len(strat_arr) else []
    bh_eq = np.cumprod(1.0 + bh_arr).tolist() if len(bh_arr) else []
    static_eq = np.cumprod(1.0 + static_arr).tolist() if len(static_arr) else []

    return {
        "params": {
            "sizing": "linear",
            "train_days": TRAIN_DAYS,
            "oos_days": OOS_DAYS,
            "step_days": STEP_DAYS,
            "round_trip_cost": ROUND_TRIP_COST,
            "lower_grid": LOWER_GRID,
            "upper_grid": UPPER_GRID,
            "min_gap": MIN_GAP,
            "score_smooth_window": SCORE_SMOOTH_WINDOW,
            "min_train_sharpe": MIN_TRAIN_SHARPE,
            "fallback_lower": FALLBACK_LOWER,
            "fallback_upper": FALLBACK_UPPER,
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
            "total_turnover": total_turnover,
            "avg_time_in_market": avg_tim,
            "avg_position": avg_pos,
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
    print("=" * 76)
    print(f"TW Walk-Forward Linear Backtest — continuous sizing on TAIEX")
    print(f"Sample: {p['date_min']} → {p['date_max']}  ({p['n_samples']} aligned days)")
    print(f"WF: train={p['train_days']}d / OOS={p['oos_days']}d / step={p['step_days']}d "
          f"| cost={p['round_trip_cost']*100:.2f}% RT")
    print("-" * 76)
    print(f"{'Fold':>4} {'OOS start':>11} {'Lo':>3} {'Up':>3} {'FB':>3} "
          f"{'TrSh':>6} {'CAGR':>7} {'Shrp':>6} {'MaxDD':>7} {'AvgPos':>7} {'Trn':>5}")
    for f in result["folds"]:
        print(f"{f['fold_idx']:>4d} "
              f"{f['oos_start']:>11} "
              f"{f['lower']:>3.0f} {f['upper']:>3.0f} "
              f"{'Y' if f['used_fallback'] else '.':>3} "
              f"{f['train_sharpe']:>6.2f} "
              f"{f['oos_cagr']*100:>6.1f}% "
              f"{f['oos_sharpe']:>6.2f} "
              f"{f['oos_maxdd']*100:>6.1f}% "
              f"{f['oos_avg_position']:>7.2f} "
              f"{f['oos_turnover']:>5.1f}")
    print("-" * 76)
    print(f"Aggregate OOS ({a['n_folds']} folds, {a['oos_n_days']} days):")
    print(f"  WF linear    : CAGR {a['oos_cagr']*100:6.2f}%  Sharpe {a['oos_sharpe']:.2f}  "
          f"MaxDD {a['oos_maxdd']*100:6.1f}%  Total {a['oos_total_return']*100:7.1f}%")
    print(f"  Static 30/70 : CAGR {a['static_cagr']*100:6.2f}%  Sharpe {a['static_sharpe']:.2f}  "
          f"MaxDD {a['static_maxdd']*100:6.1f}%  Total {a['static_total_return']*100:7.1f}%")
    print(f"  BuyHold      : CAGR {a['bh_cagr']*100:6.2f}%  Sharpe {a['bh_sharpe']:.2f}  "
          f"MaxDD {a['bh_maxdd']*100:6.1f}%  Total {a['bh_total_return']*100:7.1f}%")
    print(f"  Total turnover: {a['total_turnover']:.1f}  Avg position: "
          f"{a['avg_position']:.2f}  Fallback folds: {a['fallback_fold_pct']*100:.0f}%")
    print("=" * 76)


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
