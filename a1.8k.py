import sys
import os
import time
import re
import requests
import pandas as pd
from datetime import datetime, timedelta
from pathlib import Path
from bs4 import BeautifulSoup
from urllib.parse import urljoin

sys.path.insert(0, "_src")
from financial_tools import FMPClient, FMPError, MARKET_TZ

pd.set_option('display.max_columns', None)
pd.set_option('display.max_colwidth', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.expand_frame_repr', False)

# --- Configuration ---
# Anchor everything to market time. On a GitHub runner the clock is UTC, so
# after 8pm ET the machine already thinks it is tomorrow.
RUN_DATE = datetime.now(MARKET_TZ).strftime("%Y-%m-%d")
# 0 = today only. Raise it (LOOKBACK_DAYS=3) to backfill after a missed run.
LOOKBACK_DAYS = int(os.getenv("LOOKBACK_DAYS", "0"))
# Requests to sec.gov must carry a real contact address.
SEC_USER_AGENT = os.getenv("SEC_USER_AGENT", "Dave Zhuo zhuo.longhao@gmail.com")
# Politeness delay for sec.gov. Runner IPs are shared and get throttled harder
# than your home connection, so this is deliberately slower than 0.2s.
SEC_DELAY = float(os.getenv("SEC_DELAY", "0.5"))
ROOT_DIR = Path("sec_filings_8k")
output_dir = ROOT_DIR / RUN_DATE
output_dir.mkdir(parents=True, exist_ok=True)

# Filings already scanned on a previous run. Only useful if this directory
# survives between runs (committed back to the repo, or restored from cache).
SEEN_PATH = ROOT_DIR / "_seen_filings.txt"

# --- Instantiate the Classes ---
client1 = FMPClient()

# raise_on_error=True: a bad key, an exhausted quota or a blocked IP now raises
# FMPError and fails the job, instead of returning [] and looking like a quiet
# day. An empty list from here is a genuine "nothing was filed".
filings_8k = client1.get_data(
    'sec-filings-8k', "ALL", raise_on_error=True, lookback_days=LOOKBACK_DAYS
)

if not filings_8k:
    print(f"No 8-K filings for {RUN_DATE} (ET). Weekend or holiday. Nothing to do.")
    sys.exit(0)

pd.DataFrame(filings_8k).to_csv(output_dir / "filings_8k.csv", index=False)

# --- Skip anything already processed on an earlier run ---
seen = set()
if SEEN_PATH.exists():
    seen = {line.strip() for line in SEEN_PATH.read_text().splitlines() if line.strip()}

pending = [f for f in filings_8k if f.get("link") not in seen]
skipped = len(filings_8k) - len(pending)
print(f"{len(filings_8k)} filings returned | {skipped} already seen | {len(pending)} to scan")

if not pending:
    print("Everything in this window has already been processed.")
    sys.exit(0)

SEC_BASE = "https://www.sec.gov"

session = requests.Session()
session.headers.update({
    "User-Agent": SEC_USER_AGENT,
    "Accept-Encoding": "gzip, deflate",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
})


def download_sec_html(url, timeout=(10, 30)):
    """Fetch a page from sec.gov with exponential backoff on throttling."""
    delay = 5
    for attempt in range(4):
        r = session.get(url, timeout=timeout)
        if r.status_code in (429, 403):
            print(f"   Throttled by SEC ({r.status_code}). Sleeping {delay}s...")
            time.sleep(delay)
            delay *= 2
            continue
        r.raise_for_status()
        return r.text
    raise Exception(f"Failed to fetch {url} after 4 attempts.")


def html_to_clean_text(html):
    soup = BeautifulSoup(html, "html.parser")
    for tag in soup(["script", "style", "noscript", "nav", "header", "footer"]):
        tag.decompose()
    text = soup.get_text(" ", strip=True)
    text = re.sub(r"\s+", " ", text)
    return text


def extract_sec_documents(filing_detail_url, filing_detail_html):
    soup = BeautifulSoup(filing_detail_html, "html.parser")
    docs = []
    table = soup.find("table", class_="tableFile", summary="Document Format Files")
    if not table:
        return docs
    rows = table.find_all("tr")[1:]
    for row in rows:
        cells = row.find_all("td")
        if len(cells) < 5:
            continue
        seq = cells[0].get_text(" ", strip=True)
        description = cells[1].get_text(" ", strip=True)
        document_cell = cells[2]
        doc_type = cells[3].get_text(" ", strip=True)
        size = cells[4].get_text(" ", strip=True)
        a = document_cell.find("a")
        if not a:
            continue
        href = a.get("href")
        document_name = a.get_text(" ", strip=True)
        if href.startswith("/ix?doc="):
            href = href.replace("/ix?doc=", "")
        full_url = urljoin(SEC_BASE, href)
        docs.append({
            "seq": seq, "description": description, "document": document_name,
            "type": doc_type, "size": size, "url": full_url,
        })
    return docs


def has_keywords_in_text(text):
    text = text.lower()
    keywords = [
        "credit agreement", "amended and restated credit agreement",
        "loan agreement", "revolving credit facility", "term loan",
        "entry into a material definitive agreement", "item 1.01",
    ]
    return any(keyword in text for keyword in keywords)


def search_actual_sec_documents(filing_detail_url):
    index_html = download_sec_html(filing_detail_url)
    docs = extract_sec_documents(filing_detail_url, index_html)
    results = []
    for doc in docs:
        doc_type = doc["type"].upper()
        if doc_type not in ["8-K", "EX-10.1", "EX-10.2", "EX-99.1"]:
            continue
        try:
            html = download_sec_html(doc["url"])
            text = html_to_clean_text(html)
            matched = has_keywords_in_text(text)
            results.append({
                "type": doc["type"], "description": doc["description"],
                "document": doc["document"], "url": doc["url"],
                "matched": matched, "text": text,
            })
            time.sleep(SEC_DELAY)
        except Exception as e:
            results.append({
                "type": doc["type"], "description": doc["description"],
                "document": doc["document"], "url": doc["url"],
                "matched": False, "text": "", "error": str(e),
            })
    return results


all_results = []
processed_urls = []
failed_filings = 0

for i, filing in enumerate(pending, start=1):
    symbol = filing.get("symbol")
    filing_date = filing.get("filingDate")
    filing_detail_url = filing.get("link")
    print(f"[{i}/{len(pending)}] Assessing: {symbol} | {filing_date}")
    try:
        doc_results = search_actual_sec_documents(filing_detail_url)
        matched_docs = [r for r in doc_results if r["matched"]]
        for r in doc_results:
            all_results.append({
                "symbol": symbol, "filingDate": filing_date,
                "filing_detail_url": filing_detail_url, "document_type": r.get("type"),
                "document_description": r.get("description"), "document_url": r.get("url"),
                "matched": r.get("matched"), "error": r.get("error"),
            })
        # Only mark a filing as seen once its index page was actually parsed.
        processed_urls.append(filing_detail_url)
        if matched_docs:
            print(f"  MATCH: {len(matched_docs)} document(s)")
        else:
            print("  No match")
    except Exception as e:
        failed_filings += 1
        print(f"  ERROR: {e}")
        all_results.append({
            "symbol": symbol, "filingDate": filing_date, "filing_detail_url": filing_detail_url,
            "document_type": None, "document_description": None, "document_url": None,
            "matched": False, "error": str(e),
        })
    time.sleep(SEC_DELAY)

if processed_urls:
    with SEEN_PATH.open("a") as fh:
        for url in processed_urls:
            fh.write(f"{url}\n")

if not all_results:
    print("No SEC documents were retrieved for any filing.")
    sys.exit(1)

results_df = pd.DataFrame(all_results)
results_df.to_csv(output_dir / "filings_8k_assessment.csv", index=False)

matched_df = results_df[
    (results_df["matched"] == True) & (results_df["document_type"] == "8-K")
]

if not matched_df.empty:
    matched_df = matched_df[["symbol", "filingDate", "filing_detail_url", "document_type"]]
    matched_df.to_csv(output_dir / "filings_8k_assessment_matched.csv", index=False)
    print(f"Exported {len(matched_df)} matched 8-K filings. to {output_dir}")
else:
    print("No matched 8-K documents in this window.")

print(
    f"Summary | filings scanned: {len(processed_urls)} | "
    f"documents assessed: {len(results_df)} | filings failed: {failed_filings}"
)

# Fail the run if the SEC side was broadly unreachable rather than just noisy.
if failed_filings and failed_filings == len(pending):
    print("Every filing failed to download. Treating this as a failed run.")
    sys.exit(1)