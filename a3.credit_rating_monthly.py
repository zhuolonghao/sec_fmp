import os
import sys
import pandas as pd
import numpy as np
from datetime import date
from pathlib import Path

# --- Absolute Path Fix for GitHub Actions ---
current_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.join(current_dir, "_src")
sys.path.insert(0, src_dir)

from financial_tools import FinancialAnalyzer, FMPClient

pd.set_option('display.max_columns', None)
pd.set_option('display.max_colwidth', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.expand_frame_repr', False)

# --- Handle Inputs: Cloud UI ---
env_tickers = os.environ.get("TARGET_TICKERS")
env_anchor_date = os.environ.get("TARGET_ANCHOR_DATE")
env_anchor_date = pd.to_datetime(
        env_anchor_date,
        errors="coerce",
    ).dt.strftime('%Y-%m-%d')
print(f"env_anchor_date: {env_anchor_date}")

# Exit early if inputs are null, empty, or just whitespace
if not env_tickers or not env_tickers.strip():
    print("⚠️ No tickers provided in the input variables. Exiting early.")
    sys.exit(0)

print(f"☁️ Cloud mode detected. Analyzing inputs: {env_tickers}")
clean_tickers = [t.strip().upper() for t in env_tickers.split(',') if t.strip()]

if not clean_tickers:
    print("⚠️ Ticker list is empty after cleaning. Exiting early.")
    sys.exit(0)

# Simplified: Just a flat list [string, list]
ticker_list = [env_anchor_date, clean_tickers]

# --- Run the Logic ---
# Unpack the list directly instead of looping
category, tickers = ticker_list

base_dir = Path("bqr") / f"{env_anchor_date}_{tickers[0]}"
output_dir = base_dir 
output_dir.mkdir(parents=True, exist_ok=True)



# ------------------------------------------------------------------
# 1. Pulling Price Data for Each Ticker
# ------------------------------------------------------------------

client1 = FMPClient()
all_tickers_price = []
for symbol in np.unique(tickers):
    print(f"\n--- Processing {symbol} ---")

    price_data = client1.get_data('historical-price-eod', symbol)
    if price_data:
        price = pd.DataFrame(price_data)
        all_tickers_price.append(price)
        
# --- Combine and Export ---
if all_tickers_price:
    price_df = pd.concat(all_tickers_price, axis=0, ignore_index=False)
    
    
# ------------------------------------------------------------------
# 2. Calculate Z-Scores and Decline Triggers
# ------------------------------------------------------------------
print(f"\n--- Calculating Z-scores and Decline Trigger ---")

from zscore_stats import (
    ZScoreConfig,
    calculate_all_z_statistics,
    get_decline_trigger_summary,
    get_first_decline_trigger,
    get_first_trigger,
    get_trigger_summary,
    format_z_statistics,
)

config = ZScoreConfig(
    anchor_date=env_anchor_date,
    volatility_returns=252,
    horizon=20,
    thresholds=(1.5, 2.0, 2.5),
    decline_thresholds=(0.10, 0.15, 0.20),
)

z_df, error_df = calculate_all_z_statistics(
    price_df=price_df,
    config=config,
    stop_on_error=True,
)

z_df_display = format_z_statistics(
    z_df,
    decimals=2,
)


wulf_columns = [    
    "symbol","anchor_date","anchor_close","sigma (%)",
    "date","open","low","close","cumulative_return_from_anchor",

    'decline_trigger_price_10', 'decline_trigger_10',
    'decline_trigger_price_15', 'decline_trigger_15',
    'decline_trigger_price_20', 'decline_trigger_20',
    "trigger_price_n1_5",'low_trigger_n1_5',
    "trigger_price_n2_0",'low_trigger_n2_0',
    "trigger_price_n2_5",'low_trigger_n2_5'
]

column_rename_map = {
    "anchor_date": "anchor date",
    "anchor_close": "anchor close",
    "cumulative_return_from_anchor": "ret_itd (%)",
    
    "decline_trigger_price_10": "px dd10",
    "decline_trigger_10": "trig dd10",

    "decline_trigger_price_15": "px dd15",
    "decline_trigger_15": "trig dd15",

    "decline_trigger_price_20": "px dd20",
    "decline_trigger_20": "trig dd20",

    "trigger_price_n1_5": "px z1.5",
    "low_trigger_n1_5": "trig z1.5",

    "trigger_price_n2_0": "px z2.0",
    "low_trigger_n2_0": "trig z2.0",

    "trigger_price_n2_5": "px z2.5",
    "low_trigger_n2_5": "trig z2.5",
}

output_df = (
    z_df_display.loc[:, wulf_columns]
    .rename(columns=column_rename_map)
    .sort_values(['symbol', 'date'])
    .reset_index(drop=True)
)

# ------------------------------------------------------------------
# 3. Save to CSV
# ------------------------------------------------------------------
print(f"\n--- Writing to BQR folder ---")

for symbol in np.unique(tickers):
    symbol_df = output_df[output_df['symbol'] == symbol]
    if not symbol_df.empty:
        symbol_file = os.path.join(output_dir, f"{symbol}.csv")
        symbol_df.to_csv(symbol_file, index=False)
