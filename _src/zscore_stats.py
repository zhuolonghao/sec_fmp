from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd

Variant = Literal["close", "low"]


@dataclass(frozen=True)
class ZScoreConfig:
    """
    Configuration for z-score and simple anchor-decline statistics.

    Parameters
    ----------
    anchor_date:
        Date whose closing price is used as the anchor price. The same date
        is also the final observation in the frozen MAD-volatility window.
    volatility_returns:
        Number of daily close-to-close log returns used for frozen volatility.
        For example, 252 returns require 253 closing prices through the anchor.
    horizon:
        Z-score lookback measured in trading observations.
    thresholds:
        Positive z-score thresholds. A trigger occurs when z <= -N.
    decline_thresholds:
        Simple cumulative-decline thresholds expressed as decimal fractions.
        For example, 0.10 means a 10% decline from the anchor close.
    """

    anchor_date: str = "2026-05-01"
    volatility_returns: int = 252
    horizon: int = 20
    thresholds: tuple[float, ...] = (2.0, 2.5)
    decline_thresholds: tuple[float, ...] = (0.10, 0.15, 0.20)


def _threshold_suffix(threshold: float) -> str:
    """Convert a z threshold such as 2.0 to the column suffix '2_0'."""
    return str(float(threshold)).replace(".", "_")


def _decline_suffix(threshold: float) -> str:
    """
    Convert a decline fraction into a percentage suffix.

    Examples
    --------
    0.10 -> "10"
    0.15 -> "15"
    0.20 -> "20"
    0.125 -> "12_5"
    """
    percentage = float(threshold) * 100.0

    if percentage.is_integer():
        return str(int(percentage))

    return (
        f"{percentage:.10f}"
        .rstrip("0")
        .rstrip(".")
        .replace(".", "_")
    )


def _validate_config(config: ZScoreConfig) -> None:
    """Validate configuration values before calculations begin."""
    if config.volatility_returns <= 0:
        raise ValueError("volatility_returns must be positive.")

    if config.horizon <= 0:
        raise ValueError("horizon must be positive.")

    if not config.thresholds:
        raise ValueError("thresholds cannot be empty.")

    if any(not np.isfinite(x) or x <= 0 for x in config.thresholds):
        raise ValueError("Every z threshold must be finite and positive.")

    if not config.decline_thresholds:
        raise ValueError("decline_thresholds cannot be empty.")

    if any(
        not np.isfinite(x) or x <= 0 or x >= 1
        for x in config.decline_thresholds
    ):
        raise ValueError(
            "Every decline threshold must be finite and between 0 and 1."
        )


def prepare_price_df(price_df: pd.DataFrame) -> pd.DataFrame:
    """
    Validate and sort a multi-symbol OHLC DataFrame.

    Required columns
    ----------------
    symbol, date, open, high, low, close

    Extra columns such as volume, change, changePercent, and vwap are preserved.
    The input may be newest-first and may have duplicate DataFrame index values.
    Duplicate symbol/date observations are not allowed.
    """
    required = {"symbol", "date", "open", "high", "low", "close"}
    missing = required.difference(price_df.columns)

    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    df = price_df.copy().reset_index(drop=True)

    df["symbol"] = (
        df["symbol"]
        .astype(str)
        .str.strip()
        .str.upper()
    )

    df["date"] = pd.to_datetime(
        df["date"],
        errors="coerce",
    ).dt.normalize()

    price_cols = ["open", "high", "low", "close"]
    df[price_cols] = df[price_cols].apply(
        pd.to_numeric,
        errors="coerce",
    )

    if df["date"].isna().any():
        bad = df[df["date"].isna()]
        raise ValueError(f"Invalid date values found:\n{bad}")

    if df["symbol"].eq("").any():
        bad = df[df["symbol"].eq("")]
        raise ValueError(f"Blank symbols found:\n{bad}")

    if df[price_cols].isna().any().any():
        bad = df[df[price_cols].isna().any(axis=1)]
        raise ValueError(
            "Missing or invalid OHLC values found:\n"
            f"{bad[['symbol', 'date', *price_cols]]}"
        )

    if (df[price_cols] <= 0).any().any():
        bad = df[(df[price_cols] <= 0).any(axis=1)]
        raise ValueError(
            "OHLC prices must all be positive:\n"
            f"{bad[['symbol', 'date', *price_cols]]}"
        )

    duplicate_mask = df.duplicated(
        ["symbol", "date"],
        keep=False,
    )

    if duplicate_mask.any():
        duplicates = df.loc[
            duplicate_mask,
            ["symbol", "date", *price_cols],
        ].sort_values(["symbol", "date"])

        raise ValueError(
            "Duplicate symbol/date observations found:\n"
            f"{duplicates}"
        )

    invalid_high = (
        (df["high"] < df["open"])
        | (df["high"] < df["close"])
        | (df["high"] < df["low"])
    )

    invalid_low = (
        (df["low"] > df["open"])
        | (df["low"] > df["close"])
        | (df["low"] > df["high"])
    )

    if invalid_high.any() or invalid_low.any():
        bad = df[invalid_high | invalid_low]
        raise ValueError(
            "Invalid OHLC relationships found:\n"
            f"{bad[['symbol', 'date', *price_cols]]}"
        )

    return (
        df.sort_values(["symbol", "date"])
        .reset_index(drop=True)
    )


def calculate_frozen_sigma(
    symbol_df: pd.DataFrame,
    config: ZScoreConfig,
) -> float:
    """
    Calculate frozen MAD-based daily close volatility for one symbol.

    Formula
    -------
    r_t = ln(close_t / close_{t-1})

    sigma_daily =
        1.4826 * median(|r_t - median(r)|)

    The calibration window ends on the anchor date, inclusive.
    """
    _validate_config(config)

    if symbol_df.empty:
        raise ValueError("symbol_df is empty.")

    symbol_df = (
        symbol_df.sort_values("date")
        .reset_index(drop=True)
    )

    symbol = str(symbol_df["symbol"].iloc[0])
    anchor = pd.Timestamp(config.anchor_date).normalize()

    history = symbol_df.loc[
        symbol_df["date"] <= anchor
    ].copy()

    if not (history["date"] == anchor).any():
        raise ValueError(
            f"{symbol}: no price observation on {anchor.date()}."
        )

    required_closes = config.volatility_returns + 1

    if len(history) < required_closes:
        raise ValueError(
            f"{symbol}: need at least {required_closes} closes through "
            f"{anchor.date()} to calculate "
            f"{config.volatility_returns} returns; "
            f"only {len(history)} available."
        )

    calibration = history.iloc[-required_closes:].copy()

    log_returns = np.log(
        calibration["close"]
        / calibration["close"].shift(1)
    ).dropna()

    if len(log_returns) != config.volatility_returns:
        raise RuntimeError(
            f"{symbol}: expected "
            f"{config.volatility_returns} returns, "
            f"but calculated {len(log_returns)}."
        )

    median_return = float(log_returns.median())

    mad = float(
        np.median(
            np.abs(log_returns.to_numpy() - median_return)
        )
    )

    sigma_daily = 1.4826 * mad

    if not np.isfinite(sigma_daily) or sigma_daily <= 0:
        raise ValueError(
            f"{symbol}: invalid sigma_daily={sigma_daily}."
        )

    return sigma_daily


def calculate_symbol_z_statistics(
    symbol_df: pd.DataFrame,
    config: ZScoreConfig,
) -> pd.DataFrame:
    """
    Calculate z-score statistics and anchor-decline triggers for one symbol.

    Z-score variants
    ----------------
    z_close =
        ln(close_t / close_{t-h})
        / (sigma_daily * sqrt(h))

    z_low =
        ln(low_t / close_{t-h})
        / (sigma_daily * sqrt(h))

    Simple anchor-decline triggers
    ------------------------------
    anchor_close is the close on config.anchor_date.

    cumulative_return_from_anchor =
        close_t / anchor_close - 1

    cumulative_decline_from_anchor =
        1 - close_t / anchor_close

    For decline threshold D:

        decline_trigger_price_D = anchor_close * (1 - D)

        decline_trigger_D =
            close_t <= decline_trigger_price_D

    The simple decline flags are daily conditions, not sticky flags. If the
    stock later rebounds above a threshold, that day's flag becomes False.
    Use get_first_decline_trigger() to retrieve the first crossing date.
    """
    _validate_config(config)

    if symbol_df.empty:
        raise ValueError("symbol_df is empty.")

    result = (
        symbol_df.sort_values("date")
        .reset_index(drop=True)
        .copy()
    )

    symbol = str(result["symbol"].iloc[0])
    anchor = pd.Timestamp(config.anchor_date).normalize()

    anchor_rows = result.loc[
        result["date"] == anchor,
        "close",
    ]

    if anchor_rows.empty:
        raise ValueError(
            f"{symbol}: no price observation on {anchor.date()}."
        )

    anchor_close = float(anchor_rows.iloc[0])

    sigma_daily = calculate_frozen_sigma(
        result,
        config,
    )

    denominator = (
        sigma_daily
        * math.sqrt(config.horizon)
    )

    # Z-score reference is the close exactly h trading observations earlier.
    result["reference_date"] = result["date"].shift(
        config.horizon
    )

    result["reference_close"] = result["close"].shift(
        config.horizon
    )

    result["return_h_close_log"] = np.log(
        result["close"]
        / result["reference_close"]
    )

    result["return_h_low_log"] = np.log(
        result["low"]
        / result["reference_close"]
    )

    result["z_close"] = (
        result["return_h_close_log"]
        / denominator
    )

    result["z_low"] = (
        result["return_h_low_log"]
        / denominator
    )

    result["sigma_daily"] = sigma_daily
    result["sigma (%)"] = sigma_daily * 100
    result["z_denominator"] = denominator

    for threshold in config.thresholds:
        suffix = _threshold_suffix(threshold)

        result[f"trigger_price_n{suffix}"] = (
            result["reference_close"]
            * np.exp(-threshold * denominator)
        )

        result[f"close_trigger_n{suffix}"] = (
            result["z_close"] <= -threshold
        )

        result[f"low_trigger_n{suffix}"] = (
            result["z_low"] <= -threshold
        )

    # Simple cumulative decline from the anchor-date close.
    result["anchor_date"] = anchor
    result["anchor_close"] = anchor_close

    result["cumulative_return_from_anchor"] = (
        result["close"] / anchor_close - 1.0
    ) * 100

    result["cumulative_decline_from_anchor"] = (
        1.0 - result["close"] / anchor_close
    )

    for threshold in config.decline_thresholds:
        suffix = _decline_suffix(threshold)
        trigger_price = anchor_close * (1.0 - threshold)

        result[f"decline_trigger_price_{suffix}"] = trigger_price

        result[f"decline_trigger_{suffix}"] = (
            result["low"] <= trigger_price
        )

    # Signal evaluation begins after the anchor date.
    # reference_close should exist because frozen sigma requires substantial
    # pre-anchor history, but the condition is kept explicit for safety.
    result = result.loc[
        (result["date"] > anchor)
        & result["reference_close"].notna()
    ].copy()

    return result.reset_index(drop=True)


def calculate_all_z_statistics(
    price_df: pd.DataFrame,
    config: ZScoreConfig | None = None,
    *,
    stop_on_error: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Calculate all statistics for every symbol in price_df.

    Returns
    -------
    z_statistics:
        Full post-anchor daily time series for successfully processed symbols.
        Despite its historical name, this DataFrame now also contains the
        simple anchor-decline trigger columns.
    errors:
        One row per symbol that could not be processed.
    """
    if config is None:
        config = ZScoreConfig()

    _validate_config(config)
    prepared = prepare_price_df(price_df)

    results: list[pd.DataFrame] = []
    errors: list[dict[str, str]] = []

    for symbol, symbol_df in prepared.groupby(
        "symbol",
        sort=True,
    ):
        try:
            results.append(
                calculate_symbol_z_statistics(
                    symbol_df,
                    config,
                )
            )
        except Exception as exc:
            errors.append(
                {
                    "symbol": symbol,
                    "error": str(exc),
                }
            )

            if stop_on_error:
                raise ValueError(
                    "DATA UNAVAILABLE — cannot calculate "
                    f"{symbol}: {exc}"
                ) from exc

    z_statistics = (
        pd.concat(results, ignore_index=True)
        if results
        else pd.DataFrame()
    )

    return z_statistics, pd.DataFrame(errors)


def get_first_trigger(
    z_df: pd.DataFrame,
    symbol: str,
    threshold: float,
    variant: Variant,
) -> dict[str, object]:
    """
    Return the first z-score trigger and next-session opening fill.

    variant="close" corresponds to the close-based trigger.
    variant="low" corresponds to the intraday-low trigger.
    """
    symbol = symbol.upper().strip()

    if variant not in {"close", "low"}:
        raise ValueError(
            "variant must be 'close' or 'low'."
        )

    suffix = _threshold_suffix(threshold)
    trigger_col = f"{variant}_trigger_n{suffix}"
    z_col = f"z_{variant}"

    required_columns = {
        "symbol",
        "date",
        "open",
        "close",
        "low",
        "reference_date",
        "reference_close",
        "sigma_daily",
        trigger_col,
        z_col,
        f"trigger_price_n{suffix}",
    }

    missing_columns = required_columns.difference(z_df.columns)

    if missing_columns:
        raise ValueError(
            "z_df is missing required columns: "
            f"{sorted(missing_columns)}"
        )

    symbol_data = (
        z_df.loc[z_df["symbol"] == symbol]
        .sort_values("date")
        .reset_index(drop=True)
        .copy()
    )

    if symbol_data.empty:
        return {
            "symbol": symbol,
            "status": "No calculated data",
        }

    triggered = symbol_data.loc[
        symbol_data[trigger_col]
    ]

    if triggered.empty:
        return {
            "symbol": symbol,
            "threshold": threshold,
            "variant": variant,
            "status": "Not triggered — no position",
        }

    trigger_index = int(triggered.index[0])
    row = symbol_data.loc[trigger_index]
    observed_col = (
        "close"
        if variant == "close"
        else "low"
    )

    output: dict[str, object] = {
        "symbol": symbol,
        "threshold": threshold,
        "variant": variant,
        "status": "Triggered",
        "trigger_date": row["date"],
        "observed_trigger_price": float(
            row[observed_col]
        ),
        "threshold_price": float(
            row[f"trigger_price_n{suffix}"]
        ),
        "reference_date": row["reference_date"],
        "reference_close": float(
            row["reference_close"]
        ),
        "sigma_daily": float(
            row["sigma_daily"]
        ),
        "z_at_trigger": float(
            row[z_col]
        ),
        "buy_date": None,
        "buy_open": None,
    }

    next_index = trigger_index + 1

    if next_index < len(symbol_data):
        fill = symbol_data.loc[next_index]
        output["buy_date"] = fill["date"]
        output["buy_open"] = float(
            fill["open"]
        )
    else:
        output["status"] = (
            "Triggered — no subsequent session available "
            "for next-day open fill"
        )

    return output


def get_first_decline_trigger(
    z_df: pd.DataFrame,
    symbol: str,
    decline_threshold: float,
) -> dict[str, object]:
    """
    Return the first close-based anchor-decline trigger and next-open fill.

    Parameters
    ----------
    z_df:
        Output from calculate_all_z_statistics().
    symbol:
        Ticker symbol.
    decline_threshold:
        Decimal decline fraction, such as 0.10, 0.15, or 0.20.
    """
    symbol = symbol.upper().strip()

    if (
        not np.isfinite(decline_threshold)
        or decline_threshold <= 0
        or decline_threshold >= 1
    ):
        raise ValueError(
            "decline_threshold must be between 0 and 1."
        )

    suffix = _decline_suffix(decline_threshold)
    trigger_col = f"decline_trigger_{suffix}"
    trigger_price_col = (
        f"decline_trigger_price_{suffix}"
    )

    required_columns = {
        "symbol",
        "date",
        "open",
        "close",
        "anchor_date",
        "anchor_close",
        "cumulative_return_from_anchor",
        "cumulative_decline_from_anchor",
        trigger_col,
        trigger_price_col,
    }

    missing_columns = required_columns.difference(
        z_df.columns
    )

    if missing_columns:
        raise ValueError(
            "z_df is missing required columns: "
            f"{sorted(missing_columns)}"
        )

    symbol_data = (
        z_df.loc[z_df["symbol"] == symbol]
        .sort_values("date")
        .reset_index(drop=True)
        .copy()
    )

    if symbol_data.empty:
        return {
            "symbol": symbol,
            "status": "No calculated data",
        }

    triggered = symbol_data.loc[
        symbol_data[trigger_col]
    ]

    if triggered.empty:
        return {
            "symbol": symbol,
            "decline_threshold": decline_threshold,
            "decline_threshold_percent": (
                decline_threshold * 100.0
            ),
            "status": "Not triggered — no position",
        }

    trigger_index = int(triggered.index[0])
    row = symbol_data.loc[trigger_index]

    output: dict[str, object] = {
        "symbol": symbol,
        "decline_threshold": decline_threshold,
        "decline_threshold_percent": (
            decline_threshold * 100.0
        ),
        "status": "Triggered",
        "anchor_date": row["anchor_date"],
        "anchor_close": float(
            row["anchor_close"]
        ),
        "trigger_date": row["date"],
        "observed_close": float(
            row["close"]
        ),
        "threshold_price": float(
            row[trigger_price_col]
        ),
        "cumulative_return_at_trigger": float(
            row["cumulative_return_from_anchor"]
        ),
        "cumulative_decline_at_trigger": float(
            row["cumulative_decline_from_anchor"]
        ),
        "buy_date": None,
        "buy_open": None,
    }

    next_index = trigger_index + 1

    if next_index < len(symbol_data):
        fill = symbol_data.loc[next_index]
        output["buy_date"] = fill["date"]
        output["buy_open"] = float(
            fill["open"]
        )
    else:
        output["status"] = (
            "Triggered — no subsequent session available "
            "for next-day open fill"
        )

    return output


def get_trigger_summary(
    z_df: pd.DataFrame,
    *,
    thresholds: tuple[float, ...] = (2.0, 2.5),
) -> pd.DataFrame:
    """Build a first z-trigger summary for every symbol and variant."""
    if z_df.empty:
        return pd.DataFrame()

    rows: list[dict[str, object]] = []

    for symbol in sorted(
        z_df["symbol"].dropna().unique()
    ):
        for threshold in thresholds:
            for variant in ("close", "low"):
                rows.append(
                    get_first_trigger(
                        z_df,
                        symbol,
                        threshold,
                        variant,
                    )
                )

    return pd.DataFrame(rows)


def get_decline_trigger_summary(
    z_df: pd.DataFrame,
    *,
    decline_thresholds: tuple[float, ...] = (
        0.10,
        0.15,
        0.20,
    ),
) -> pd.DataFrame:
    """
    Build a first simple-decline trigger summary for every symbol.
    """
    if z_df.empty:
        return pd.DataFrame()

    rows: list[dict[str, object]] = []

    for symbol in sorted(
        z_df["symbol"].dropna().unique()
    ):
        for threshold in decline_thresholds:
            rows.append(
                get_first_decline_trigger(
                    z_df,
                    symbol,
                    threshold,
                )
            )

    return pd.DataFrame(rows)

def format_z_statistics(
    z_df: pd.DataFrame,
    decimals: int = 2,
) -> pd.DataFrame:
    """
    Return a rounded presentation copy.

    Trigger calculations should be completed before calling this function.
    """
    result = z_df.copy()

    numeric_cols = result.select_dtypes(
        include=["number"]
    ).columns

    result[numeric_cols] = result[numeric_cols].round(decimals)

    return result

__all__ = [
    "ZScoreConfig",
    "prepare_price_df",
    "calculate_frozen_sigma",
    "calculate_symbol_z_statistics",
    "calculate_all_z_statistics",
    "get_first_trigger",
    "get_first_decline_trigger",
    "get_trigger_summary",
    "get_decline_trigger_summary",
    "format_z_statistics",
]
