"""
Purchase-anchored stop-loss / upside metrics, built on top of zscore_stats.

The base module (zscore_stats) anchors everything on a single global anchor
close. Here the anchor is *per position*: each ticker's purchase date and its
`Avg cost`. We reuse the base module's frozen-sigma and rolling-z machinery,
then add purchase-anchored metrics the base module doesn't have:

    - sigma (%)                          (from zscore_stats)
    - cumulative_return_from_purchase    close / avg_cost - 1
    - stop-loss @ 20%                    low  <= avg_cost * (1 - 0.20)   (intraday)
    - upside / "return high" @ 20%       high >= avg_cost * (1 + 0.20)   (intraday)
    - upward z 1.5 / 2.0                 z_close >= +threshold           (the upside
                                         mirror of the base module's z <= -threshold)

Inputs
------
price_df : OHLC in the base module's format -> columns: symbol, date, open, high, low, close
avg_df   : your `avg` frame -> columns: Ticker, Purchase date, Avg cost

Frozen sigma needs `volatility_returns + 1` closes *through each purchase date*,
so price history must extend well before the earliest purchase.
"""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pandas as pd

from zscore_stats import (
    ZScoreConfig,
    prepare_price_df,
    calculate_symbol_z_statistics,
)


def _snap_anchor(symbol_prices: pd.DataFrame, purchase_date: pd.Timestamp) -> pd.Timestamp | None:
    """Last trading day on/before the purchase date that exists for this symbol."""
    on_or_before = symbol_prices.loc[symbol_prices["date"] <= purchase_date, "date"]
    return None if on_or_before.empty else on_or_before.max()


def _first_true(frame: pd.DataFrame, flag_col: str) -> dict:
    """First row where flag_col is True, plus the next session's open (fill)."""
    hits = frame.loc[frame[flag_col]]
    if hits.empty:
        return {"hit": False, "first_date": pd.NaT, "fill_open": np.nan}
    i = int(hits.index[0])
    nxt = i + 1
    fill = float(frame.loc[nxt, "open"]) if nxt < len(frame) else np.nan
    return {"hit": True, "first_date": frame.loc[i, "date"], "fill_open": fill}


def augment_positions(
    price_df: pd.DataFrame,
    avg_df: pd.DataFrame,
    config: ZScoreConfig | None = None,
    *,
    stop_pct: float = 0.20,
    up_pct: float = 0.20,
    z_up_thresholds: tuple[float, ...] = (1.5, 2.0),
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Return (daily, errors).

    `daily` is the post-purchase daily series for every position, with the base
    module's columns plus the purchase-anchored columns listed in the module docstring.
    `errors` is one row per ticker that couldn't be processed (e.g. not enough history).
    """
    if config is None:
        config = ZScoreConfig()

    prices = prepare_price_df(price_df)

    positions = avg_df.rename(
        columns={"Ticker": "symbol", "Purchase date": "purchase_date", "Avg cost": "avg_cost"}
    ).copy()
    positions["symbol"] = positions["symbol"].astype(str).str.strip().str.upper()
    positions["purchase_date"] = pd.to_datetime(positions["purchase_date"]).dt.normalize()

    daily_frames: list[pd.DataFrame] = []
    errors: list[dict] = []

    for _, pos in positions.iterrows():
        sym, pdate, avg_cost = pos["symbol"], pos["purchase_date"], float(pos["avg_cost"])
        sp = prices.loc[prices["symbol"] == sym].sort_values("date").reset_index(drop=True)

        if sp.empty:
            errors.append({"symbol": sym, "error": "no price history"})
            continue

        anchor = _snap_anchor(sp, pdate)
        if anchor is None:
            errors.append({"symbol": sym, "error": f"no trading day on/before {pdate.date()}"})
            continue

        try:
            # Reuse the base module for sigma (%) + rolling z, anchored at the purchase day.
            z = calculate_symbol_z_statistics(sp, replace(config, anchor_date=str(anchor.date())))
        except Exception as exc:  # not enough history, missing anchor bar, etc.
            errors.append({"symbol": sym, "error": str(exc)})
            continue

        z = z.copy()
        z["purchase_date"] = anchor
        z["lot_id"] = f"{sym}@{anchor.date()}"   # one lot = symbol + purchase date
        z["avg_cost"] = avg_cost

        # --- purchase-anchored return (vs avg cost, not the anchor close) ---
        z["cumulative_return_from_purchase"] = (z["close"] / avg_cost - 1.0) * 100.0

        # --- stop-loss @ stop_pct (intraday low) ---
        z["stop_price"] = avg_cost * (1.0 - stop_pct)
        z["stop_trigger"] = z["low"] <= z["stop_price"]

        # --- upside / "return high" @ up_pct (intraday high) ---
        z["high_target"] = avg_cost * (1.0 + up_pct)
        z["high_trigger"] = z["high"] >= z["high_target"]

        # --- upward z triggers: mirror of the base module's z <= -N ---
        for t in z_up_thresholds:
            suffix = str(float(t)).replace(".", "_")
            z[f"z_up_trigger_price_{suffix}"] = z["reference_close"] * np.exp(t * z["z_denominator"])
            z[f"z_up_trigger_{suffix}"] = z["z_close"] >= t

        daily_frames.append(z)

    daily = pd.concat(daily_frames, ignore_index=True) if daily_frames else pd.DataFrame()
    return daily, pd.DataFrame(errors)


def summarize_positions(
    daily: pd.DataFrame,
    *,
    z_up_thresholds: tuple[float, ...] = (1.5, 2.0),
) -> pd.DataFrame:
    """One row per LOT (symbol + purchase date): sigma, current/peak returns, and
    first stop/high/z crossings. Same symbol bought on different dates -> separate rows."""
    if daily.empty:
        return pd.DataFrame()

    rows: list[dict] = []
    for (sym, pdate), g in daily.groupby(["symbol", "purchase_date"], sort=True):
        g = g.sort_values("date").reset_index(drop=True)
        g['peak_high_return (%)'] = (g['high'] / g['avg_cost'] - 1) * 100
        g['trail_stop_trigger'] = (g['peak_high_return (%)'] - g['cumulative_return_from_purchase']) >= 20.0

        last = g.iloc[-1]

        row: dict = {
            "symbol": sym,
            "purchase_date": pdate.strftime("%Y-%m-%d"),
            "avg_cost": round(float(g["avg_cost"].iloc[0]), 2),
            "sigma (%)": round(float(g["sigma (%)"].iloc[0]), 2),
            "last_date":  last["date"],
            "cum_return_from_purchase (%)": round(float(last["cumulative_return_from_purchase"]), 1),
            "peak_high_return (%)": round(float((g["high"] / g["avg_cost"] - 1).max() * 100), 1),
        }

        stop = _first_true(g, "stop_trigger")
        row["stop_price"] = round(float(g["stop_price"].iloc[0]), 2)
        row["stop_hit"] = stop["hit"]
        row["stop_first_date"] = stop["first_date"]

        high = _first_true(g, "high_trigger")
        row["high_target"] = round(float(g["high_target"].iloc[0]), 2)
        row["high_hit"] = high["hit"]
        row["high_first_date"] = high["first_date"]

        trail_stop = _first_true(g, "trail_stop_trigger")
        row["trail_stop_trigger"] = trail_stop["hit"]
        row["trail_stop_first_date"] = trail_stop["first_date"]

        for t in z_up_thresholds:
            suffix = str(float(t)).replace(".", "_")
            zt = _first_true(g, f"z_up_trigger_{suffix}")
            row[f"z_up_trigger_{suffix}"] = zt["hit"]
            row[f"z_up_{suffix}_first_date"] = zt["first_date"]

        rows.append(row)

    return pd.DataFrame(rows)


if __name__ == "__main__":
    # See test_position_stops.py for a runnable synthetic example.
    pass