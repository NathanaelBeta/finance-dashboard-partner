"""
fetch_prices.py  (v3)
---------------------
Data-driven rewrite: all personal holdings data is read from holdings.json.
No hardcoded transaction lists, balances, or constants — this file is generic
and works for any repo that has a holdings.json in the expected schema.

Run:
  pip install yfinance pandas
  python scripts/fetch_prices.py
"""

import json
import math
from datetime import date, timedelta
from pathlib import Path

import pandas as pd
import yfinance as yf

# ─────────────────────────────────────────────────────────────
# LOAD HOLDINGS.JSON
# ─────────────────────────────────────────────────────────────

_repo_root = Path(__file__).parent.parent
H = json.loads((_repo_root / "holdings.json").read_text())

HISTORY_START = H.get("history_start", "2025-06-01")

# Trade history
IDX_TRANSACTIONS = H["trades"]["idx"]     # [[date, ticker, lot_delta], ...]
IDX_BASELINE     = H.get("idx_baseline", {})
ETF_TRANSACTIONS = H["trades"]["etf"]     # [[date, ticker, qty], ...]

# Holdings snapshot at anchor_date (for validation)
_h = H.get("holdings", {})
ANCHOR_IDX    = _h.get("idx", {})
ANCHOR_ETF    = _h.get("etf", {})
ANCHOR_CRYPTO = _h.get("crypto", {})

# Balances at anchor_date
_b = H.get("balances", {})
STOCKBIT_RDN_ANCHOR  = _b.get("stockbit_rdn", 0)
PLUANG_CASH_ANCHOR   = _b.get("pluang_cash_usd", 0)
PAYPAL_USD_ANCHOR    = _b.get("paypal_usd", 0)

# Constant holdings (non-traded)
GOLD_GRAMS   = _h.get("gold_grams", 0)
FUTURES_USD  = _h.get("futures_usd", 0)
BONDS_IDR    = sum(_h.get("bonds", {}).values())

# Cash flow series for backwards reconstruction (optional — absent → use anchor as constant)
_cf = H.get("cash_flows", {})
STOCKBIT_CASH_FLOWS = _cf.get("stockbit_idr", {})
PLUANG_CASH_FLOWS   = _cf.get("pluang_usd", {})

# BCA liquid checkpoints (replaces monthly_liquid)
CASH_CHECKPOINTS = H.get("cash_checkpoints", {})

# ─────────────────────────────────────────────────────────────
# TICKERS — auto-derived from trade history
# ─────────────────────────────────────────────────────────────

_idx_tickers = set(IDX_BASELINE.keys()) | {row[1] for row in IDX_TRANSACTIONS}
_etf_tickers = {row[1] for row in ETF_TRANSACTIONS}
_crypto_tickers = set(ANCHOR_CRYPTO.keys()) - {"USDC", "USDT"}  # stables priced at 1 USD

TICKERS = {
    "GOLD_USD": "GC=F",
    "USD_IDR":  "IDR=X",
    **{t: t + "-USD" for t in sorted(_crypto_tickers)},
    **{t: t         for t in sorted(_etf_tickers)},
    **{t: t + ".JK" for t in sorted(_idx_tickers)},
}

# ─────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────

def daterange(start: str, end: str):
    d = date.fromisoformat(start)
    e = date.fromisoformat(end)
    while d <= e:
        yield d.isoformat()
        d += timedelta(days=1)


def fetch_ticker(yahoo_symbol: str, start: str, end: str) -> dict:
    print(f"  Fetching {yahoo_symbol} ...")
    ticker = yf.Ticker(yahoo_symbol)
    df = ticker.history(start=start, end=end, interval="1d", auto_adjust=True)
    if df.empty:
        print(f"  WARNING: no data for {yahoo_symbol}")
        return {}
    result = {}
    for ts, row in df.iterrows():
        d = ts.date().isoformat()
        close = row["Close"]
        if not math.isnan(close):
            result[d] = round(float(close), 6)
    return result


def forward_fill(prices: dict, start: str, end: str) -> dict:
    filled = {}
    last = None
    for d in daterange(start, end):
        if d in prices:
            last = prices[d]
        if last is not None:
            filled[d] = last
    return filled


def compute_holdings_series(transactions, baseline, start, end):
    """Replay trade history forward from baseline to produce daily lot/share counts."""
    txns = sorted(transactions, key=lambda x: x[0])
    positions = dict(baseline)
    all_tickers = set(positions.keys()) | {row[1] for row in txns}
    snapshots = {t: {start: positions.get(t, 0)} for t in all_tickers}

    for txn_date, ticker, delta in txns:
        all_tickers.add(ticker)
        if ticker not in snapshots:
            snapshots[ticker] = {start: 0}
        if txn_date < start:
            positions[ticker] = positions.get(ticker, 0) + delta
            snapshots[ticker][start] = positions[ticker]
            continue
        positions[ticker] = positions.get(ticker, 0) + delta
        snapshots[ticker][txn_date] = round(positions[ticker], 8)

    result = {}
    for ticker in all_tickers:
        daily = {}
        last = 0.0
        for d in daterange(start, end):
            if d in snapshots.get(ticker, {}):
                last = snapshots[ticker][d]
            daily[d] = round(last, 8)
        result[ticker] = daily
    return result


def reconstruct_cash_backwards(flows: dict, anchor_value: float,
                                anchor_date: str, start: str, end: str) -> dict:
    """
    Reconstruct a daily cash balance series using an anchor value on anchor_date.
    Walk backwards from anchor_date to start (undoing flows), and forwards to end
    (applying flows). If flows is empty, anchor_value is used as a constant.
    """
    series = {}
    # Backwards from anchor to start
    balance = anchor_value
    for d in sorted(daterange(start, anchor_date), reverse=True):
        series[d] = round(balance, 2)
        balance -= flows.get(d, 0)
    # Forwards from anchor+1 to end
    balance = anchor_value
    prev = anchor_date
    for d in daterange(anchor_date, end):
        if d == anchor_date:
            series[d] = round(anchor_value, 2)
            continue
        balance += flows.get(d, 0)
        series[d] = round(balance, 2)
    return series


# ─────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────

def main():
    today = date.today().isoformat()
    anchor_date = H.get("anchor_date", today)

    print("=== Fetching prices ===")
    raw_prices = {}
    for name, symbol in TICKERS.items():
        raw_prices[name] = fetch_ticker(symbol, HISTORY_START, today)

    print("\n=== Forward-filling ===")
    daily_prices = {}
    for name, prices in raw_prices.items():
        daily_prices[name] = forward_fill(prices, HISTORY_START, today)
        print(f"  {name}: {len(daily_prices[name])} days")

    # Gold in IDR per gram
    gold_idr = {}
    for d in daterange(HISTORY_START, today):
        g = daily_prices.get("GOLD_USD", {}).get(d)
        r = daily_prices.get("USD_IDR", {}).get(d)
        if g and r:
            gold_idr[d] = round(g / 31.1035 * r, 0)
    daily_prices["GOLD_IDR_GRAM"] = gold_idr

    print("\n=== Computing holdings series ===")
    idx_holdings  = compute_holdings_series(IDX_TRANSACTIONS, IDX_BASELINE, HISTORY_START, today)
    etf_holdings  = compute_holdings_series(ETF_TRANSACTIONS, {}, HISTORY_START, today)

    # Validate against anchor snapshot
    for ticker, expected in ANCHOR_IDX.items():
        actual = idx_holdings.get(ticker, {}).get(today, 0)
        status = "OK" if abs(actual - expected) < 0.01 else "MISMATCH"
        print(f"  [{status}] IDX {ticker}: {actual} (expected {expected})")
    for ticker, expected in ANCHOR_ETF.items():
        actual = etf_holdings.get(ticker, {}).get(today, 0)
        status = "OK" if abs(actual - expected) < 0.0001 else "MISMATCH"
        print(f"  [{status}] ETF {ticker}: {actual:.6f} (expected {expected:.6f})")

    print("\n=== Reconstructing cash balances ===")
    stockbit_rdn = reconstruct_cash_backwards(
        STOCKBIT_CASH_FLOWS, STOCKBIT_RDN_ANCHOR, anchor_date, HISTORY_START, today)
    pluang_cash  = reconstruct_cash_backwards(
        PLUANG_CASH_FLOWS, PLUANG_CASH_ANCHOR, anchor_date, HISTORY_START, today)

    # Floor RDN at 0 for early dates where deposit history is incomplete
    for d in list(stockbit_rdn.keys()):
        if stockbit_rdn[d] < 0:
            stockbit_rdn[d] = 0

    print(f"  Stockbit RDN @ {anchor_date}: Rp {stockbit_rdn.get(anchor_date, 0):,.0f} (expected {STOCKBIT_RDN_ANCHOR:,.0f})")
    print(f"  Pluang cash  @ {anchor_date}: ${pluang_cash.get(anchor_date, 0):.2f} (expected {PLUANG_CASH_ANCHOR:.2f})")

    print("\n=== Computing daily net worth ===")
    daily_networth = {}

    # BCA liquid: snap to known month-end checkpoints, forward-fill between them
    bca_checkpoints = {d: v["bca"] for d, v in CASH_CHECKPOINTS.items() if "bca" in v}
    last_bca_liquid = 0

    for d in daterange(HISTORY_START, today):
        if d in bca_checkpoints:
            last_bca_liquid = bca_checkpoints[d]

        usd_idr = daily_prices.get("USD_IDR", {}).get(d)
        if not usd_idr:
            continue

        nw = 0.0

        # Gold
        nw += GOLD_GRAMS * daily_prices.get("GOLD_IDR_GRAM", {}).get(d, 0)

        # Crypto (priced in USD)
        for coin, units in ANCHOR_CRYPTO.items():
            if coin in ("USDC", "USDT"):
                nw += units * usd_idr  # stablecoins: 1 USD each
            else:
                nw += units * daily_prices.get(coin, {}).get(d, 0) * usd_idr

        # Futures + PayPal (USD, treated as constant at anchor value)
        nw += (FUTURES_USD + PAYPAL_USD_ANCHOR) * usd_idr

        # IDX stocks (lots × 100 × price in IDR)
        for ticker, series in idx_holdings.items():
            lots = series.get(d, 0)
            if lots > 0:
                nw += lots * 100 * daily_prices.get(ticker, {}).get(d, 0)

        # Stockbit RDN cash
        nw += stockbit_rdn.get(d, 0)

        # Pluang ETFs + Pluang cash (USD)
        for ticker, series in etf_holdings.items():
            shares = series.get(d, 0)
            if shares > 0:
                nw += shares * daily_prices.get(ticker, {}).get(d, 0) * usd_idr
        nw += pluang_cash.get(d, 0) * usd_idr

        # Bonds (fixed IDR)
        nw += BONDS_IDR

        # BCA liquid (forward-filled from monthly checkpoints)
        nw += last_bca_liquid

        daily_networth[d] = round(nw, 0)

    # Latest prices snapshot
    latest = {}
    for name, series in daily_prices.items():
        if series:
            ld = max(series.keys())
            latest[name] = {"price": series[ld], "date": ld}

    # Build constant holdings block for prices.json (backward compat)
    constants = {
        "GOLD_GRAMS":  GOLD_GRAMS,
        "FUTURES_USD": FUTURES_USD,
        "PAYPAL_USD":  PAYPAL_USD_ANCHOR,
        **ANCHOR_CRYPTO,
    }

    output = {
        "generated": today,
        "latest": latest,
        "daily": {
            "prices": daily_prices,
            "holdings": {
                "idx":          idx_holdings,
                "pluang":       {t: etf_holdings[t] for t in etf_holdings if t in _etf_tickers},
                "stockbit_rdn": stockbit_rdn,
                "pluang_cash":  pluang_cash,
            },
            "networth": daily_networth,
        },
        "constants": constants,
        "monthly_liquid": bca_checkpoints,
    }

    out_path = _repo_root / "prices.json"
    with open(out_path, "w") as f:
        json.dump(output, f, separators=(",", ":"))

    size_kb = out_path.stat().st_size / 1024
    print(f"\nDone. Written to {out_path} ({size_kb:.1f} KB)")
    print(f"  Net worth entries: {len(daily_networth)}")
    if daily_networth:
        nw_vals = list(daily_networth.values())
        print(f"  NW range: {min(nw_vals)/1e9:.3f}B - {max(nw_vals)/1e9:.3f}B IDR")


if __name__ == "__main__":
    main()
