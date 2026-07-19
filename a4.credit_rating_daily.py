import os
import sys
import pandas as pd
import numpy as np
from datetime import date, datetime
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


from zscore_stats import (
    ZScoreConfig,
    calculate_all_z_statistics,
    get_decline_trigger_summary,
    get_first_decline_trigger,
    get_first_trigger,
    get_trigger_summary,
    format_z_statistics,
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


# ------------------------------------------------------------------
# 1. Search for all tickers: 
# return [achor_date, ticker] for the latest date of each ticker
# ------------------------------------------------------------------

output_dir = Path("bqr")
csv_files = list(output_dir.rglob("*.csv"))

latest_by_ticker = {}

for file_path in csv_files:
    ticker = file_path.stem

    # Example parent folder: 2026-07-01_SPY
    date_text = file_path.parent.name.split("_")[0]

    try:
        file_date = datetime.strptime(date_text, "%Y-%m-%d").date()
    except ValueError:
        # Skip folders that do not begin with YYYY-MM-DD
        continue

    if (
        ticker not in latest_by_ticker
        or file_date > latest_by_ticker[ticker]
    ):
        latest_by_ticker[ticker] = file_date

result = [
    [date.isoformat(), ticker]
    for ticker, date in latest_by_ticker.items()
]

# ------------------------------------------------------------------
# 2. Process each ticker to calculate z-score statistics and decline triggers
#    and save to _triggered.csv 
# ------------------------------------------------------------------

client1 = FMPClient()
all_tickers_price = []

for env_anchor_date, symbol in result:
    print(f"\n--- Processing {symbol} ---")
    
    price_data = client1.get_data('historical-price-eod', symbol)
    
    config = ZScoreConfig(
        anchor_date=env_anchor_date,
        volatility_returns=252,
        horizon=20,
        thresholds=(1.5, 2.0, 2.5),
        decline_thresholds=(0.10, 0.15, 0.20),
    )
    
    z_df, error_df = calculate_all_z_statistics(
        price_df=pd.DataFrame(price_data),
        config=config,
        stop_on_error=True,
    )

    z_df_display = format_z_statistics(
        z_df,
        decimals=2,
    )
    
    output_df = (
        z_df_display.loc[:, wulf_columns]
        .rename(columns=column_rename_map)
        .sort_values(['symbol', 'date'])
        .reset_index(drop=True)
    )
    
    output_df['first_trigger'] = output_df.groupby(['trig dd10', 'trig dd15', 'trig dd20', 'trig z1.5', 'trig z2.0', 'trig z2.5']).cumcount() + 1
    true_trigger = output_df[['trig dd10', 'trig dd15', 'trig dd20', 'trig z1.5', 'trig z2.0', 'trig z2.5']].any(axis=1)
    first_trigger = output_df['first_trigger']  == 1
    triggered_df = output_df[true_trigger & first_trigger].sort_values(['date', 'anchor date'], ascending=[False, True])

    if triggered_df is not None and not triggered_df.empty:
        all_tickers_price.append(triggered_df.drop(columns=["first_trigger"]))


if all_tickers_price:
    price_df = pd.concat(all_tickers_price, axis=0, ignore_index=False)
    price_df['date'] = price_df['date'].dt.strftime('%Y-%m-%d')
    price_df['anchor date'] = price_df['anchor date'].dt.strftime('%Y-%m-%d')

    price_file = os.path.join(output_dir, "_triggered.csv")
    price_df.replace(False, np.nan).to_csv(price_file, index=False)
    print(f"Saved {price_file}")