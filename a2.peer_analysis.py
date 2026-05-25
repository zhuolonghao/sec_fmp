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
env_category = os.environ.get("TARGET_CATEGORY", "AdHoc_Run")

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
ticker_list = [env_category, clean_tickers]

# --- Run the Logic ---
# Unpack the list directly instead of looping
category, tickers = ticker_list

base_dir = Path("outputs") / f"{category}_{tickers[0]}"
output_dir = base_dir / date.today().isoformat()
output_dir.mkdir(parents=True, exist_ok=True)

client1 = FMPClient()
analyzer = FinancialAnalyzer()

all_tickers_data = []
all_tickers_price = []
all_tickers_news = {}
all_tickers_sec_filings = {}

for symbol in np.unique(tickers):
    print(f"\n--- Processing {symbol} ---")
    inc = client1.get_data('income-statement', symbol)
    bs  = client1.get_data('balance-sheet-statement', symbol)
    cf  = client1.get_data('cash-flow-statement', symbol)
    ev = client1.get_data('enterprise-values', symbol)
    news = client1.get_data('news', symbol)
    sec_filings = client1.get_data('sec-filings-search', symbol)
    
    price_data = client1.get_data('historical-price-eod', symbol)
    if price_data:
        price = pd.DataFrame(price_data)
        all_tickers_price.append(price)
        
    all_tickers_news[symbol] = news
    all_tickers_sec_filings[symbol] = sec_filings

    rev_bus_seg = client1.get_data('revenue-product-segmentation', symbol)
    rev_geo_seg = client1.get_data('revenue-geographic-segmentation', symbol)
    raw_data = rev_bus_seg if rev_bus_seg else rev_geo_seg
    
    if raw_data:
        rev_seg = pd.DataFrame(raw_data)
        rev_seg = rev_seg.apply(analyzer.process_segments, axis=1)
    else:
        rev_seg = pd.DataFrame(columns=analyzer.segment_vars + analyzer.id_vars)
    rev_seg = rev_seg if rev_seg is not None else pd.DataFrame()

    if inc and bs and cf and ev:
        df_merged = analyzer.build_merged_dataframe(inc, bs, cf, ev, rev_seg)
        output = analyzer.process_ltm_data(df_merged)
        output = analyzer.add_category(output)

        if not output.empty:
            print(output.map(analyzer.format_numbers).iloc[:, :5])
            file_path = os.path.join(output_dir, f"{symbol}_{output.iloc[2, 0]}.csv")
            output.to_csv(file_path, index=True)
            print(f"Saved {file_path}")

            output = output.replace('-', np.nan).bfill(axis=1)
            latest_col = output.iloc[:, [0]].fillna('-')
            latest_col.columns = [symbol]
            all_tickers_data.append(latest_col)
    else:
        print(f"Skipping {symbol} due to missing data.")

# --- Combine and Export ---
if all_tickers_data:
    final_df = pd.concat(all_tickers_data, axis=1)
    csv_file = os.path.join(output_dir, "_peer_analysis.csv")
    final_df.to_csv(csv_file, index=True)
    print(f"\nSuccess! Combined dataset saved to {csv_file}")
    
    if all_tickers_price:
        price_df = pd.concat(all_tickers_price, axis=0, ignore_index=False)
        price_file = os.path.join(output_dir, "_price.xlsx")
        price_df.to_excel(price_file, index=False)
        print(f"\nSuccess! daily price dataset saved to {price_file}")
        
    news_file = base_dir / "_news.xlsx"
    writer_kwargs = {'engine': 'openpyxl', 'mode': 'w'}
    if os.path.exists(news_file):
        writer_kwargs['mode'] = 'a'
        writer_kwargs['if_sheet_exists'] = 'replace'
    with pd.ExcelWriter(news_file, **writer_kwargs) as writer:   
        for k, v in all_tickers_news.items():
            if v: pd.DataFrame(v).to_excel(writer, sheet_name=k, index=False)
    print(f"\nSuccess! news dataset saved to {news_file}")

    sec_filings_file = base_dir / "_sec_filings.xlsx"
    with pd.ExcelWriter(sec_filings_file, **writer_kwargs) as writer:   
        for k, v in all_tickers_sec_filings.items():
            if v: pd.DataFrame(v).to_excel(writer, sheet_name=k, index=False)       
    print(f"\nSuccess! SEC filings dataset saved to {sec_filings_file}")
else:
    print("\nNo data was collected to export.")