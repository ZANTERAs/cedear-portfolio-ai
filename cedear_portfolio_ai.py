#!/usr/bin/env python3
"""
CEDEAR Heat Finder
==================
5-factor scoring model to identify "hot" CEDEARs for mid-short term.

Scoring weights (total 100 pts):
  1. Valuation vs Growth  (30 pts) -- PEG ratio, Forward P/E
  2. Analyst Signals      (25 pts) -- consensus rating + target price upside
  3. Earnings Momentum    (20 pts) -- revenue growth, EPS growth, margins
  4. Price Momentum       (15 pts) -- 1m / 3m / 6m returns (percentile ranked)
  5. Technical Setup      (10 pts) -- RSI, 50/200 MA position

Output: PDF report + JSON with top 20 portfolio.

Usage:
    python cedear_portfolio_ai.py
"""

import io
import json
import os
import sys
import math
import warnings
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor, as_completed

warnings.filterwarnings("ignore")

if sys.platform == "win32":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

import numpy as np
import pandas as pd
import yfinance as yf

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.backends.backend_pdf import PdfPages


# ── UNIVERSE (individual stocks only — no ETFs) ───────────────────────────────
UNIVERSE = [
    # (byma_ticker, yf_ticker, name, exchange, sector)
    # ── TECNOLOGIA ──
    ("AAPL",  "AAPL",    "Apple",              "NASDAQ", "Tecnologia"),
    ("MSFT",  "MSFT",    "Microsoft",          "NASDAQ", "Tecnologia"),
    ("NVDA",  "NVDA",    "Nvidia",             "NASDAQ", "Tecnologia"),
    ("GOOGL", "GOOGL",   "Alphabet",           "NASDAQ", "Tecnologia"),
    ("META",  "META",    "Meta Platforms",     "NASDAQ", "Tecnologia"),
    ("AMZN",  "AMZN",    "Amazon",             "NASDAQ", "Tecnologia"),
    ("AVGO",  "AVGO",    "Broadcom",           "NASDAQ", "Tecnologia"),
    ("ASML",  "ASML",    "ASML Holding",       "NASDAQ", "Tecnologia"),
    ("NOW",   "NOW",     "ServiceNow",         "NYSE",   "Tecnologia"),
    ("ANET",  "ANET",    "Arista Networks",    "NYSE",   "Tecnologia"),
    ("ADBE",  "ADBE",    "Adobe",              "NASDAQ", "Tecnologia"),
    ("CRM",   "CRM",     "Salesforce",         "NYSE",   "Tecnologia"),
    ("ORCL",  "ORCL",    "Oracle",             "NYSE",   "Tecnologia"),
    ("ARM",   "ARM",     "ARM Holdings",       "NASDAQ", "Tecnologia"),
    ("TSM",   "TSM",     "TSMC",               "NYSE",   "Tecnologia"),
    ("LRCX",  "LRCX",    "Lam Research",       "NASDAQ", "Tecnologia"),
    ("AMAT",  "AMAT",    "Applied Materials",  "NASDAQ", "Tecnologia"),
    ("QCOM",  "QCOM",    "Qualcomm",           "NASDAQ", "Tecnologia"),
    ("MU",    "MU",      "Micron Technology",  "NASDAQ", "Tecnologia"),
    ("TXN",   "TXN",     "Texas Instruments",  "NASDAQ", "Tecnologia"),
    ("INTC",  "INTC",    "Intel",              "NASDAQ", "Tecnologia"),
    ("CSCO",  "CSCO",    "Cisco",              "NASDAQ", "Tecnologia"),
    ("IBM",   "IBM",     "IBM",                "NYSE",   "Tecnologia"),
    ("MRVL",  "MRVL",    "Marvell Technology", "NASDAQ", "Tecnologia"),
    ("ISRG",  "ISRG",    "Intuitive Surgical", "NASDAQ", "Tecnologia"),
    ("TEAM",  "TEAM",    "Atlassian",          "NASDAQ", "Tecnologia"),
    ("CRWD",  "CRWD",    "CrowdStrike",        "NASDAQ", "Tecnologia"),
    ("SHOP",  "SHOP",    "Shopify",            "NYSE",   "Tecnologia"),
    ("SAP",   "SAP",     "SAP SE",             "NYSE",   "Tecnologia"),
    ("AMD",   "AMD",     "AMD",                "NASDAQ", "Tecnologia"),
    ("ADI",   "ADI",     "Analog Devices",     "NASDAQ", "Tecnologia"),
    ("ACN",   "ACN",     "Accenture",          "NYSE",   "Tecnologia"),
    # ── IA / SEMIS ──
    ("PLTR",  "PLTR",    "Palantir",           "NYSE",   "IA/Semis"),
    ("MSTR",  "MSTR",    "MicroStrategy",      "NASDAQ", "IA/Semis"),
    ("AI",    "AI",      "C3.AI",              "NYSE",   "IA/Semis"),
    ("NBIS",  "NBIS",    "Nebius Group",       "NASDAQ", "IA/Semis"),
    # ── FINANZAS ──
    ("JPM",   "JPM",     "JPMorgan Chase",     "NYSE",   "Finanzas"),
    ("V",     "V",       "Visa",               "NYSE",   "Finanzas"),
    ("MA",    "MA",      "Mastercard",         "NYSE",   "Finanzas"),
    ("GS",    "GS",      "Goldman Sachs",      "NYSE",   "Finanzas"),
    ("BA.C",  "BAC",     "Bank of America",    "NYSE",   "Finanzas"),
    ("BX",    "BX",      "Blackstone",         "NYSE",   "Finanzas"),
    ("AXP",   "AXP",     "American Express",   "NYSE",   "Finanzas"),
    ("C",     "C",       "Citigroup",          "NYSE",   "Finanzas"),
    ("WFC",   "WFC",     "Wells Fargo",        "NYSE",   "Finanzas"),
    ("SCHW",  "SCHW",    "Charles Schwab",     "NYSE",   "Finanzas"),
    ("BK",    "BK",      "BNY Mellon",         "NYSE",   "Finanzas"),
    ("COIN",  "COIN",    "Coinbase",           "NASDAQ", "Finanzas"),
    ("PYPL",  "PYPL",    "PayPal",             "NASDAQ", "Finanzas"),
    ("BKNG",  "BKNG",    "Booking Holdings",   "NASDAQ", "Finanzas"),
    # ── SALUD ──
    ("LLY",   "LLY",     "Eli Lilly",          "NYSE",   "Salud"),
    ("JNJ",   "JNJ",     "Johnson & Johnson",  "NYSE",   "Salud"),
    ("UNH",   "UNH",     "UnitedHealth",       "NYSE",   "Salud"),
    ("ABBV",  "ABBV",    "AbbVie",             "NYSE",   "Salud"),
    ("MRK",   "MRK",     "Merck",              "NYSE",   "Salud"),
    ("TMO",   "TMO",     "Thermo Fisher",      "NYSE",   "Salud"),
    ("AMGN",  "AMGN",    "Amgen",              "NASDAQ", "Salud"),
    ("GILD",  "GILD",    "Gilead Sciences",    "NASDAQ", "Salud"),
    ("PFE",   "PFE",     "Pfizer",             "NYSE",   "Salud"),
    ("ABT",   "ABT",     "Abbott Labs",        "NYSE",   "Salud"),
    ("DHR",   "DHR",     "Danaher",            "NYSE",   "Salud"),
    ("MDT",   "MDT",     "Medtronic",          "NYSE",   "Salud"),
    ("MRNA",  "MRNA",    "Moderna",            "NASDAQ", "Salud"),
    ("NVS",   "NVS",     "Novartis",           "NYSE",   "Salud"),
    ("VRTX",  "VRTX",    "Vertex Pharma",      "NYSE",   "Salud"),
    ("AZN",   "AZN",     "AstraZeneca",        "NYSE",   "Salud"),
    # ── CONSUMO ──
    ("COST",  "COST",    "Costco",             "NASDAQ", "Consumo"),
    ("WMT",   "WMT",     "Walmart",            "NYSE",   "Consumo"),
    ("HD",    "HD",      "Home Depot",         "NYSE",   "Consumo"),
    ("MCD",   "MCD",     "McDonald's",         "NYSE",   "Consumo"),
    ("NFLX",  "NFLX",    "Netflix",            "NASDAQ", "Consumo"),
    ("SBUX",  "SBUX",    "Starbucks",          "NASDAQ", "Consumo"),
    ("KO",    "KO",      "Coca-Cola",          "NYSE",   "Consumo"),
    ("PEP",   "PEP",     "PepsiCo",            "NASDAQ", "Consumo"),
    ("PG",    "PG",      "Procter & Gamble",   "NYSE",   "Consumo"),
    ("NKE",   "NKE",     "Nike",               "NYSE",   "Consumo"),
    ("TGT",   "TGT",     "Target",             "NYSE",   "Consumo"),
    ("SPOT",  "SPOT",    "Spotify",            "NYSE",   "Consumo"),
    ("UBER",  "UBER",    "Uber",               "NYSE",   "Consumo"),
    ("DISN",  "DIS",     "Walt Disney",        "NYSE",   "Consumo"),
    ("JD",    "JD",      "JD.com",             "NASDAQ", "Consumo"),
    ("PDD",   "PDD",     "PDD Holdings",       "NASDAQ", "Consumo"),
    ("BABA",  "BABA",    "Alibaba",            "NYSE",   "Consumo"),
    ("TSLA",  "TSLA",    "Tesla",              "NASDAQ", "Consumo"),
    ("PM",    "PM",      "Philip Morris",      "NYSE",   "Consumo"),
    ("MO",    "MO",      "Altria Group",       "NYSE",   "Consumo"),
    # ── ENERGIA ──
    ("XOM",   "XOM",     "Exxon Mobil",        "NYSE",   "Energia"),
    ("CVX",   "CVX",     "Chevron",            "NYSE",   "Energia"),
    ("COP",   "COP",     "ConocoPhillips",     "NYSE",   "Energia"),
    ("SLB",   "SLB",     "Schlumberger",       "NYSE",   "Energia"),
    ("BP",    "BP",      "BP",                 "NYSE",   "Energia"),
    ("TTE",   "TTE",     "TotalEnergies",      "NYSE",   "Energia"),
    ("OXY",   "OXY",     "Occidental",         "NYSE",   "Energia"),
    ("HAL",   "HAL",     "Halliburton",        "NYSE",   "Energia"),
    ("EQNR",  "EQNR",    "Equinor",            "NYSE",   "Energia"),
    ("PETR3", "PETR3.SA","Petrobras (B3)",      "B3",     "Energia"),
    ("PBR",   "PBR",     "Petrobras (ADR)",    "NYSE",   "Energia"),
    # ── MINERIA ──
    ("NEM",   "NEM",     "Newmont",            "NYSE",   "Mineria"),
    ("B",     "GOLD",    "Barrick Gold",       "NYSE",   "Mineria"),
    ("AEM",   "AEM",     "Agnico Eagle",       "NYSE",   "Mineria"),
    ("VALE",  "VALE",    "Vale",               "NYSE",   "Mineria"),
    ("BHP",   "BHP",     "BHP Group",          "NYSE",   "Mineria"),
    ("FCX",   "FCX",     "Freeport-McMoRan",   "NYSE",   "Mineria"),
    ("RIO",   "RIO",     "Rio Tinto",          "NYSE",   "Mineria"),
    ("KGC",   "KGC",     "Kinross Gold",       "NYSE",   "Mineria"),
    ("GFI",   "GFI",     "Gold Fields",        "NYSE",   "Mineria"),
    ("PAAS",  "PAAS",    "Pan American Silver", "NASDAQ", "Mineria"),
    ("VALE3", "VALE3.SA","Vale (B3)",           "B3",     "Mineria"),
    ("CCJ",   "CCJ",     "Cameco",             "NYSE",   "Mineria"),
    # ── INDUSTRIA ──
    ("CAT",   "CAT",     "Caterpillar",        "NYSE",   "Industria"),
    ("DE",    "DE",      "Deere & Co",         "NYSE",   "Industria"),
    ("HON",   "HON",     "Honeywell",          "NYSE",   "Industria"),
    ("GE",    "GE",      "GE Aerospace",       "NYSE",   "Industria"),
    ("LMT",   "LMT",     "Lockheed Martin",    "NYSE",   "Industria"),
    ("RTX",   "RTX",     "RTX Corp",           "NYSE",   "Industria"),
    ("UNP",   "UNP",     "Union Pacific",      "NYSE",   "Industria"),
    ("BA",    "BA",      "Boeing",             "NYSE",   "Industria"),
    ("HWM",   "HWM",     "Howmet Aerospace",   "NYSE",   "Industria"),
    ("RKLB",  "RKLB",    "Rocket Lab",         "NASDAQ", "Industria"),
    ("ASTS",  "ASTS",    "AST SpaceMobile",    "NASDAQ", "Industria"),
    # ── LATAM ──
    ("MELI",  "MELI",    "MercadoLibre",       "NASDAQ", "LatAm"),
    ("GLOB",  "GLOB",    "Globant",            "NYSE",   "LatAm"),
    ("VIST",  "VIST",    "Vista Energy",       "NYSE",   "LatAm"),
    ("ARCO",  "ARCO",    "Arcos Dorados",      "NYSE",   "LatAm"),
    ("CAAP",  "CAAP",    "Corp. America AP",   "NYSE",   "LatAm"),
    ("BIOX",  "BIOX",    "Bioceres Crop",      "NYSE",   "LatAm"),
    # ── BRASIL B3 ──
    ("ITUB3", "ITUB3.SA","Itau Unibanco (B3)", "B3",     "LatAm"),
    ("BBAS3", "BBAS3.SA","Banco do Brasil (B3)","B3",    "LatAm"),
    ("WEGE3", "WEGE3.SA","WEG (B3)",           "B3",     "LatAm"),
    ("SUZB3", "SUZB3.SA","Suzano (B3)",        "B3",     "LatAm"),
    # ── EUROPA ──
    ("ADS",   "ADS.DE",  "Adidas",             "XETRA",  "Europa"),
    ("BAS",   "BAS.DE",  "BASF",               "XETRA",  "Europa"),
    ("BAYN",  "BAYN.DE", "Bayer",              "XETRA",  "Europa"),
    ("MBG",   "MBG.DE",  "Mercedes-Benz",      "XETRA",  "Europa"),
    ("GSK",   "GSK",     "GSK",                "NYSE",   "Europa"),
    ("SHEL",  "SHEL",    "Shell",              "NYSE",   "Europa"),
    # ── UTILITIES ──
    ("NEE",   "NEE",     "NextEra Energy",     "NYSE",   "Utilities"),
    ("CEG",   "CEG",     "Constellation Engy", "NASDAQ", "Utilities"),
]

SECTOR_COLORS = {
    "Tecnologia": "#4472C4", "IA/Semis":  "#7030A0",
    "Finanzas":   "#2E75B6", "Salud":     "#70AD47",
    "Consumo":    "#FFC000", "Energia":   "#E05050",
    "Mineria":    "#808080", "Industria": "#ED7D31",
    "LatAm":      "#00B0F0", "Europa":    "#9E480E",
    "Utilities":  "#92D050",
}
BG         = "#F5F7FA"
HDR        = "#1F3864"
ALT        = "#DDE8F8"
SCORE_CLR  = {
    "Valuacion":  "#2E75B6",
    "Analistas":  "#70AD47",
    "Earnings":   "#FFC000",
    "Momentum":   "#ED7D31",
    "Tecnico":    "#7030A0",
}
N_PICKS = 20


# ── PRICE DATA ────────────────────────────────────────────────────────────────
def fetch_prices(tickers: list, lookback_years: int = 1) -> pd.DataFrame:
    start = (datetime.now() - timedelta(days=lookback_years * 365)).strftime("%Y-%m-%d")
    print(f"  Descargando precios de {len(tickers)} tickers ({lookback_years} ano)...")
    data   = yf.download(tickers, start=start, auto_adjust=True, progress=False, threads=True)
    prices = data["Close"] if "Close" in data.columns else data
    if isinstance(prices, pd.Series):
        prices = prices.to_frame(name=tickers[0])
    return prices


def compute_price_metrics(prices: pd.DataFrame) -> dict:
    """RSI, momentum, moving averages for every ticker."""
    out = {}
    for t in prices.columns:
        col = prices[t].dropna()
        if len(col) < 30:
            continue

        def mom(n):
            return (col.iloc[-1] / col.iloc[-n] - 1) if len(col) >= n else None

        delta    = col.diff()
        gain     = delta.clip(lower=0).rolling(14).mean()
        loss     = (-delta.clip(upper=0)).rolling(14).mean()
        rs       = gain / loss
        rsi_val  = (100 - 100 / (1 + rs)).iloc[-1]

        ma50     = col.rolling(50).mean().iloc[-1]  if len(col) >= 50  else None
        ma200    = col.rolling(200).mean().iloc[-1] if len(col) >= 200 else None
        cur      = float(col.iloc[-1])
        rsi_f    = float(rsi_val) if not np.isnan(rsi_val) else None

        # 52-week high (1yr of data == 52 weeks) and distance below it
        high_52w = float(col.max())
        pct_below_high = (cur / high_52w - 1.0) * 100 if high_52w > 0 else None  # <=0; -8.3 == 8.3% below high

        out[t] = {
            "mom_1m":       mom(21),
            "mom_3m":       mom(63),
            "mom_6m":       mom(126),
            "rsi":          rsi_f,
            "above_50ma":   (cur > float(ma50))  if ma50  is not None else None,
            "above_200ma":  (cur > float(ma200)) if ma200 is not None else None,
            "golden_cross": (float(ma50) > float(ma200)) if (ma50 is not None and ma200 is not None) else None,
            "price":        cur,
            "high_52w":     high_52w,
            "pct_below_high": pct_below_high,
            "overbought":   (rsi_f is not None and rsi_f >= 70),
        }
    return out


# ── FUNDAMENTALS ──────────────────────────────────────────────────────────────
def _fetch_one(ticker: str) -> tuple:
    try:
        info = yf.Ticker(ticker).info
        return ticker, info
    except Exception:
        return ticker, {}


def fetch_fundamentals(tickers: list) -> dict:
    print(f"  Descargando fundamentals de {len(tickers)} tickers (puede tardar 2-3 min)...")
    results = {}
    done    = 0
    with ThreadPoolExecutor(max_workers=12) as pool:
        futures = {pool.submit(_fetch_one, t): t for t in tickers}
        for fut in as_completed(futures):
            t, info = fut.result()
            results[t] = info
            done += 1
            if done % 20 == 0 or done == len(tickers):
                print(f"    {done}/{len(tickers)} completados...")
    return results


# ── SCORING ───────────────────────────────────────────────────────────────────
# Max points per factor:
MAX_VALUACION = 30
MAX_ANALISTAS = 25
MAX_EARNINGS  = 20
MAX_MOMENTUM  = 15
MAX_TECNICO   = 10


def _clamp(v, lo=0.0, hi=1.0):
    return max(lo, min(hi, v))


def score_valuacion(info: dict) -> float:
    """PEG ratio and Forward P/E.  Returns 0–30."""
    peg = info.get("pegRatio", None)
    fpe = info.get("forwardPE", None)
    eps_growth = info.get("earningsGrowth", None)

    # PEG (primary signal)
    if peg is not None and 0 < peg < 50:
        if   peg < 0.8:  s = 1.00
        elif peg < 1.2:  s = 0.88
        elif peg < 1.7:  s = 0.72
        elif peg < 2.5:  s = 0.50
        elif peg < 4.0:  s = 0.25
        else:            s = 0.05
        return round(MAX_VALUACION * s, 2)

    # Fallback: forward P/E adjusted by growth
    if fpe is not None and fpe > 0:
        if eps_growth and eps_growth > 0:
            implied_peg = fpe / (eps_growth * 100)
            if   implied_peg < 1.0: s = 0.90
            elif implied_peg < 2.0: s = 0.65
            elif implied_peg < 3.5: s = 0.40
            else:                   s = 0.15
        else:
            # No growth info — score by P/E alone
            if   fpe < 12:  s = 0.90
            elif fpe < 18:  s = 0.75
            elif fpe < 25:  s = 0.55
            elif fpe < 35:  s = 0.30
            elif fpe < 50:  s = 0.12
            else:           s = 0.03
        return round(MAX_VALUACION * s, 2)

    return round(MAX_VALUACION * 0.40, 2)   # neutral if no data


def score_analistas(info: dict) -> float:
    """Analyst consensus + target price upside.  Returns 0–25."""
    rec    = info.get("recommendationMean", None)   # 1=Strong Buy … 5=Strong Sell
    target = info.get("targetMeanPrice", None)
    cur    = info.get("currentPrice", None) or info.get("regularMarketPrice", None)
    n      = info.get("numberOfAnalystOpinions", 0) or 0

    if n < 3:
        return round(MAX_ANALISTAS * 0.40, 2)   # not enough coverage → neutral

    # Consensus score (0–1)
    rec_s = _clamp((5 - rec) / 4) if rec is not None else 0.45

    # Target upside (0–1)
    if target and cur and cur > 0:
        upside = (target - cur) / cur
        if   upside >  0.40: up_s = 1.00
        elif upside >  0.25: up_s = 0.82
        elif upside >  0.15: up_s = 0.65
        elif upside >  0.05: up_s = 0.45
        elif upside > -0.05: up_s = 0.25
        else:                up_s = 0.05
    else:
        up_s = 0.45

    combined = 0.45 * rec_s + 0.55 * up_s
    return round(MAX_ANALISTAS * combined, 2)


def score_earnings(info: dict) -> float:
    """Revenue growth, EPS growth, profit margin.  Returns 0–20."""
    rev_g    = info.get("revenueGrowth",    None)   # YoY quarterly
    eps_g    = info.get("earningsGrowth",   None)   # YoY quarterly
    margin   = info.get("profitMargins",    None)

    def growth_score(g):
        if g is None: return 0.40
        if   g >  0.35: return 1.00
        elif g >  0.20: return 0.82
        elif g >  0.10: return 0.62
        elif g >  0.0:  return 0.38
        elif g > -0.10: return 0.18
        else:           return 0.02

    def margin_score(m):
        if m is None: return 0.40
        if   m > 0.25: return 1.00
        elif m > 0.15: return 0.75
        elif m > 0.05: return 0.45
        elif m > 0:    return 0.20
        else:          return 0.00

    combined = 0.40 * growth_score(rev_g) + 0.40 * growth_score(eps_g) + 0.20 * margin_score(margin)
    return round(MAX_EARNINGS * combined, 2)


def score_momentum_all(price_metrics: dict) -> dict:
    """Percentile-rank momentum within universe.  Returns {ticker: 0–15}."""
    def collect(field):
        return {t: m[field] for t, m in price_metrics.items() if m.get(field) is not None}

    m1 = collect("mom_1m")
    m3 = collect("mom_3m")
    m6 = collect("mom_6m")

    def pct_rank(d, ticker):
        if ticker not in d or not d:
            return 0.50
        vals = list(d.values())
        return sum(v <= d[ticker] for v in vals) / len(vals)

    scores = {}
    for t in price_metrics:
        p1 = pct_rank(m1, t)
        p3 = pct_rank(m3, t)
        p6 = pct_rank(m6, t)
        combined = 0.20 * p1 + 0.50 * p3 + 0.30 * p6
        scores[t] = round(MAX_MOMENTUM * combined, 2)
    return scores


def score_tecnico(pm: dict) -> float:
    """RSI + MA position.  Returns 0–10."""
    rsi     = pm.get("rsi", None)
    ab50    = pm.get("above_50ma", None)
    ab200   = pm.get("above_200ma", None)
    golden  = pm.get("golden_cross", None)

    # RSI: sweet spot 48–68 (momentum without being overbought)
    if rsi is not None:
        if   48 <= rsi <= 68:  rsi_s = 1.00
        elif 40 <= rsi < 75:   rsi_s = 0.65
        elif rsi >= 75:         rsi_s = 0.20   # overbought
        else:                   rsi_s = 0.30   # oversold
    else:
        rsi_s = 0.50

    # Moving averages
    ma_s = 0.0
    if ab200 is True:  ma_s += 0.35
    if ab50  is True:  ma_s += 0.35
    if golden is True: ma_s += 0.30
    if ab200 is None and ab50 is None:
        ma_s = 0.50

    combined = 0.45 * rsi_s + 0.55 * ma_s
    return round(MAX_TECNICO * combined, 2)


def compute_all_scores(fundamentals: dict, price_metrics: dict) -> pd.DataFrame:
    """Build full score DataFrame for every ticker."""
    mom_scores = score_momentum_all(price_metrics)
    rows = []
    for t, info in fundamentals.items():
        pm = price_metrics.get(t, {})
        s_val = score_valuacion(info)
        s_ana = score_analistas(info)
        s_ear = score_earnings(info)
        s_mom = mom_scores.get(t, MAX_MOMENTUM * 0.40)
        s_tec = score_tecnico(pm)
        total = s_val + s_ana + s_ear + s_mom + s_tec
        rows.append({
            "ticker":    t,
            "total":     round(total, 2),
            "Valuacion": s_val,
            "Analistas": s_ana,
            "Earnings":  s_ear,
            "Momentum":  s_mom,
            "Tecnico":   s_tec,
            # Raw fundamentals for the table
            "fwd_pe":    info.get("forwardPE", None),
            "peg":       info.get("pegRatio",  None),
            "rev_growth":info.get("revenueGrowth", None),
            "eps_growth":info.get("earningsGrowth", None),
            "roe":       info.get("returnOnEquity", None),
            "rec_mean":  info.get("recommendationMean", None),
            "n_analysts":info.get("numberOfAnalystOpinions", None),
            "target":    info.get("targetMeanPrice", None),
            "price":     pm.get("price", info.get("currentPrice", None)),
            "rsi":       pm.get("rsi", None),
            "mom_3m":    pm.get("mom_3m", None),
            "pct_below_high": pm.get("pct_below_high", None),
            "overbought":     pm.get("overbought", False),
        })
    df = pd.DataFrame(rows).set_index("ticker").sort_values("total", ascending=False)
    return df


# ── PORTFOLIO WEIGHTS ─────────────────────────────────────────────────────────
def build_portfolio(scores_df: pd.DataFrame, n: int = N_PICKS,
                    min_w: float = 0.02, max_w: float = 0.15) -> dict:
    """Score-proportional weights with min/max constraints."""
    top = scores_df.head(n)
    raw = {t: math.exp(row["total"] * 0.06) for t, row in top.iterrows()}
    total = sum(raw.values())
    weights = {t: v / total for t, v in raw.items()}

    # Clip and renormalize (iterative)
    for _ in range(20):
        clipped = {t: max(min_w, min(max_w, w)) for t, w in weights.items()}
        total_c = sum(clipped.values())
        weights = {t: v / total_c for t, v in clipped.items()}
        if all(min_w - 1e-6 <= w <= max_w + 1e-6 for w in weights.values()):
            break

    return {t: round(w, 4) for t, w in sorted(weights.items(), key=lambda x: -x[1])}


# ── PDF REPORT ────────────────────────────────────────────────────────────────
def generate_pdf_report(scores_df: pd.DataFrame, weights: dict,
                        universe_df: pd.DataFrame, output_path: str) -> None:
    print("  Generando reporte PDF...")
    idx = universe_df.set_index("yf_ticker")

    def byma(t):   return idx.loc[t, "byma_ticker"] if t in idx.index else t
    def uname(t):  return idx.loc[t, "name"]        if t in idx.index else t
    def sector(t): return idx.loc[t, "sector"]      if t in idx.index else "Otro"
    def clr(t):    return SECTOR_COLORS.get(sector(t), "#999999")

    top20  = scores_df.head(N_PICKS)
    factors = ["Valuacion", "Analistas", "Earnings", "Momentum", "Tecnico"]

    with PdfPages(output_path) as pdf:

        # ── PAGE 1: COVER ─────────────────────────────────────────────────────
        fig = plt.figure(figsize=(11.69, 8.27))
        fig.patch.set_facecolor("#0D1F3C")

        fig.text(0.5, 0.83, "CEDEAR Heat Finder",
                 fontsize=38, ha="center", color="white", fontweight="bold")
        fig.text(0.5, 0.73, "Analisis Fundamental + Momentum  |  Seleccion Mid-Short Term",
                 fontsize=16, ha="center", color="#7EB3FF")
        fig.text(0.5, 0.64,
                 f"Generado el {datetime.now().strftime('%d/%m/%Y a las %H:%M')}",
                 fontsize=12, ha="center", color="#8899AA")

        # Factor cards
        factor_info = [
            ("Valuacion",  f"{MAX_VALUACION} pts", "PEG Ratio / Forward P/E vs Crecimiento"),
            ("Analistas",  f"{MAX_ANALISTAS} pts", "Consenso + Upside vs Precio Objetivo"),
            ("Earnings",   f"{MAX_EARNINGS} pts",  "Crecimiento de Revenue y EPS"),
            ("Momentum",   f"{MAX_MOMENTUM} pts",  "Retorno 1m / 3m / 6m (percentil)"),
            ("Tecnico",    f"{MAX_TECNICO} pts",   "RSI + Medias Moviles 50/200"),
        ]
        for i, (fname, pts, desc) in enumerate(factor_info):
            ax = fig.add_axes([0.03 + i * 0.195, 0.27, 0.18, 0.27])
            ax.set_facecolor(SCORE_CLR[fname] + "33")
            ax.set_xticks([]); ax.set_yticks([])
            for sp in ax.spines.values():
                sp.set_edgecolor(SCORE_CLR[fname]); sp.set_linewidth(2)
            ax.text(0.5, 0.75, pts, ha="center", va="center",
                    color="white", fontsize=20, fontweight="bold",
                    transform=ax.transAxes)
            ax.text(0.5, 0.52, fname, ha="center", va="center",
                    color=SCORE_CLR[fname], fontsize=11, fontweight="bold",
                    transform=ax.transAxes)
            ax.text(0.5, 0.20, desc, ha="center", va="center",
                    color="#AABBCC", fontsize=7.5, transform=ax.transAxes,
                    wrap=True)

        fig.text(0.5, 0.11,
                 "Universo: ~130 CEDEARs individuales (acciones, sin ETFs)  |  Datos: Yahoo Finance",
                 ha="center", color="#445566", fontsize=9)
        fig.text(0.5, 0.06,
                 "(*) No constituye asesoramiento financiero. Solo para fines informativos.",
                 ha="center", color="#334455", fontsize=8, style="italic")

        pdf.savefig(fig, facecolor=fig.get_facecolor())
        plt.close(fig)

        # ── PAGE 2: HEAT SCORE LEADERBOARD (stacked bars) ─────────────────────
        fig, ax = plt.subplots(figsize=(11.69, 8.27))
        fig.patch.set_facecolor(BG)
        ax.set_facecolor(BG)

        labels  = [f"{byma(t)}  {uname(t)[:22]}" for t in top20.index]
        y_pos   = np.arange(len(labels))
        bar_h   = 0.72
        lefts   = np.zeros(len(labels))

        for factor in factors:
            vals = top20[factor].values
            bars = ax.barh(y_pos, vals, left=lefts, height=bar_h,
                           color=SCORE_CLR[factor], label=factor,
                           edgecolor="white", linewidth=0.4)
            lefts += vals

        # Total score label at end
        for i, (t, row) in enumerate(top20.iterrows()):
            ax.text(row["total"] + 0.3, i, f"{row['total']:.1f}",
                    va="center", ha="left", fontsize=8, fontweight="bold", color="#333333")

        ax.set_yticks(y_pos)
        ax.set_yticklabels(labels, fontsize=8)
        ax.set_xlabel("Heat Score (0-100)", fontsize=11)
        ax.set_title("Leaderboard  -  Top 20 CEDEARs por Puntaje Compuesto",
                     fontsize=14, fontweight="bold", color=HDR, pad=12)
        ax.legend(loc="lower right", fontsize=9, framealpha=0.9,
                  title="Factor", title_fontsize=9)
        ax.set_xlim(0, 105)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.invert_yaxis()

        plt.tight_layout()
        pdf.savefig(fig, facecolor=BG)
        plt.close(fig)

        # ── PAGE 3: BUBBLE CHART — Valuacion vs Analistas ─────────────────────
        fig, ax = plt.subplots(figsize=(11.69, 8.27))
        fig.patch.set_facecolor(BG)
        ax.set_facecolor("#FAFBFD")

        for t, row in top20.iterrows():
            sz = (row["total"] / 100) * 1200 + 100
            ax.scatter(row["Valuacion"], row["Analistas"],
                       s=sz, c=clr(t), alpha=0.80,
                       edgecolors="white", linewidth=1.2, zorder=3)
            ax.annotate(byma(t), (row["Valuacion"], row["Analistas"]),
                        xytext=(5, 4), textcoords="offset points",
                        fontsize=7.5, color="#111111", zorder=4)

        # Padded axis limits from the data so bubbles + labels never clip
        xs = top20["Valuacion"].astype(float)
        ys = top20["Analistas"].astype(float)
        xpad = max(2.5, (xs.max() - xs.min()) * 0.20)
        ypad = max(2.0, (ys.max() - ys.min()) * 0.20)
        ax.set_xlim(xs.min() - xpad, xs.max() + xpad)
        ax.set_ylim(ys.min() - ypad, ys.max() + ypad)
        x0, x1 = ax.get_xlim()
        y0, y1 = ax.get_ylim()

        # Reference lines at the median of the picks (splits the cloud)
        ax.axvline(xs.median(), color="#BBBBBB", lw=0.8, linestyle="--", alpha=0.6)
        ax.axhline(ys.median(), color="#BBBBBB", lw=0.8, linestyle="--", alpha=0.6)

        # Corner labels anchored to the visible box (never clip / never shift layout)
        ax.text(x1 - (x1 - x0) * 0.02, y1 - (y1 - y0) * 0.03, "MEJOR",
                ha="right", va="top", fontsize=10, color="#27AE60",
                fontweight="bold", alpha=0.55, clip_on=True)
        ax.text(x0 + (x1 - x0) * 0.02, y0 + (y1 - y0) * 0.03, "Menor conviccion",
                ha="left", va="bottom", fontsize=9, color="#E74C3C",
                alpha=0.5, clip_on=True)

        seen = sorted(set(sector(t) for t in top20.index))
        handles = [mpatches.Patch(color=SECTOR_COLORS.get(s, "#999"), label=s) for s in seen]
        ax.legend(handles=handles, loc="upper left", fontsize=8,
                  title="Sector", title_fontsize=8.5, framealpha=0.9)

        ax.set_xlabel(f"Score Valuacion vs Crecimiento  (max {MAX_VALUACION} pts)", fontsize=11)
        ax.set_ylabel(f"Score Analistas  (max {MAX_ANALISTAS} pts)", fontsize=11)
        ax.set_title("Valuacion vs Confianza de Analistas  (tamano = heat score total)",
                     fontsize=13, fontweight="bold", color=HDR, pad=12)
        ax.grid(True, alpha=0.20, linestyle="--")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        plt.tight_layout()
        pdf.savefig(fig, facecolor=BG)
        plt.close(fig)

        # ── PAGE 4: FUNDAMENTALS TABLE ────────────────────────────────────────
        fig, ax = plt.subplots(figsize=(11.69, 8.27))
        fig.patch.set_facecolor(BG)
        ax.axis("off")
        ax.set_title("Datos Fundamentales  -  Top 20 CEDEARs",
                     fontsize=14, fontweight="bold", color=HDR, pad=16)
        ax.text(0.5, 0.965,
                "'% < Max' = caida desde el maximo de 52 semanas  "
                "(verde 5-25% = pullback sano  |  naranja = pegado al maximo)      "
                "RSI en rojo = sobrecomprado (>70)",
                transform=ax.transAxes, ha="center", va="bottom",
                fontsize=7.0, color="#666666")

        def fmt(v, fmt_str, suffix=""):
            if v is None or (isinstance(v, float) and np.isnan(v)):
                return "N/D"
            try:
                return f"{fmt_str.format(v)}{suffix}"
            except Exception:
                return "N/D"

        def rec_label(v):
            if v is None: return "N/D"
            if v <= 1.5: return "Comp. Fuerte"
            if v <= 2.5: return "Comprar"
            if v <= 3.5: return "Mantener"
            return "Vender"

        col_labels = ["BYMA", "Empresa", "Sector", "Score",
                      "PEG", "F-P/E", "Rev Gr%", "EPS Gr%",
                      "ROE%", "Consenso", "Upside%", "% < Max", "RSI"]
        col_w = [0.05, 0.135, 0.085, 0.05,
                 0.05, 0.05, 0.058, 0.058,
                 0.05, 0.085, 0.058, 0.062, 0.045]

        cell_text   = []
        cell_colors = []
        for i, (t, row) in enumerate(top20.iterrows()):
            upside = None
            if row["target"] and row["price"] and row["price"] > 0:
                upside = (row["target"] - row["price"]) / row["price"]

            pbh   = row.get("pct_below_high", None)
            rsi_v = row.get("rsi", None)
            has_pbh = pbh   is not None and not (isinstance(pbh, float)   and np.isnan(pbh))
            has_rsi = rsi_v is not None and not (isinstance(rsi_v, float) and np.isnan(rsi_v))
            pct_high_str = f"{abs(pbh):.1f}%" if has_pbh else "N/D"   # magnitude below 52w high, e.g. 8.3%
            rsi_str      = f"{rsi_v:.0f}"  if has_rsi else "N/D"

            cell_text.append([
                byma(t), uname(t)[:20], sector(t)[:12],
                f"{row['total']:.1f}",
                fmt(row["peg"],        "{:.2f}"),
                fmt(row["fwd_pe"],     "{:.1f}"),
                fmt(row["rev_growth"], "{:.1%}"),
                fmt(row["eps_growth"], "{:.1%}"),
                fmt(row["roe"],        "{:.1%}"),
                rec_label(row["rec_mean"]),
                fmt(upside,            "{:.1%}"),
                pct_high_str,
                rsi_str,
            ])
            row_bg = ALT if i % 2 == 0 else "white"
            row_c  = [row_bg] * len(col_labels)
            sc = SECTOR_COLORS.get(sector(t), "#999")
            row_c[2] = sc + "28"
            # Highlight score
            if row["total"] >= 70: row_c[3] = "#FFE066"
            elif row["total"] >= 60: row_c[3] = "#D6EAF8"
            # Distance-from-high cue: green = healthy pullback, orange = pinned to the high
            if has_pbh:
                mag = -pbh
                if mag <= 3:          row_c[11] = "#F8D9C0"   # within 3% of 52w high -> extended
                elif 5 <= mag <= 25:  row_c[11] = "#D5F5E3"   # pullback sweet spot
            # RSI overbought flag (> 70)
            if has_rsi and rsi_v >= 70:
                row_c[12] = "#F5B7B1"
            cell_colors.append(row_c)

        tbl = ax.table(
            cellText=cell_text, colLabels=col_labels,
            cellColours=cell_colors, cellLoc="center",
            loc="center", bbox=[0, 0, 1, 0.93],
        )
        tbl.auto_set_font_size(False)
        tbl.set_fontsize(7.5)
        for j in range(len(col_labels)):
            tbl[(0, j)].set_facecolor(HDR)
            tbl[(0, j)].set_text_props(color="white", fontweight="bold")
        for j, cw in enumerate(col_w):
            for i in range(len(top20) + 1):
                tbl[(i, j)].set_width(cw)

        plt.tight_layout()
        pdf.savefig(fig, facecolor=BG)
        plt.close(fig)

        # ── PAGE 5: PORTFOLIO ALLOCATION ─────────────────────────────────────
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.69, 8.27))
        fig.patch.set_facecolor(BG)
        fig.suptitle("Portfolio Recomendado  -  Top 20 por Heat Score",
                     fontsize=15, fontweight="bold", y=0.97, color=HDR)

        # Donut
        sorted_w = sorted(weights.items(), key=lambda x: -x[1])
        sizes_d  = [w for _, w in sorted_w]
        labels_d = [f"{byma(t)}\n{w:.1%}" for t, w in sorted_w]
        colors_d = [clr(t) for t, _ in sorted_w]

        ax1.pie(sizes_d, labels=labels_d, colors=colors_d, startangle=90,
                wedgeprops=dict(width=0.52, edgecolor="white", linewidth=1.4),
                textprops={"fontsize": 7.0})
        ax1.text(0, 0, f"{len(weights)}\nposic.", ha="center", va="center",
                 fontsize=13, fontweight="bold", color=HDR)
        ax1.set_title("Pesos por Posicion", fontsize=12,
                      fontweight="bold", pad=12, color=HDR)

        # Weight table (right side)
        ax2.axis("off")
        ax2.set_title("Detalle de Pesos", fontsize=12,
                      fontweight="bold", pad=12, color=HDR)

        tbl_data = []
        for i, (t, w) in enumerate(sorted_w):
            tbl_data.append([str(i+1), byma(t), uname(t)[:22],
                             f"{scores_df.loc[t,'total']:.1f}" if t in scores_df.index else "N/D",
                             f"{w:.1%}"])

        t2 = ax2.table(
            cellText=tbl_data,
            colLabels=["#", "BYMA", "Empresa", "Score", "Peso"],
            cellLoc="center", loc="center",
            bbox=[0, 0, 1, 0.97],
        )
        t2.auto_set_font_size(False)
        t2.set_fontsize(8)
        for j in range(5):
            t2[(0, j)].set_facecolor(HDR)
            t2[(0, j)].set_text_props(color="white", fontweight="bold")
        for i in range(1, len(tbl_data) + 1):
            bg = ALT if i % 2 == 0 else "white"
            for j in range(5):
                t2[(i, j)].set_facecolor(bg)
            t2[(i, 4)].set_text_props(fontweight="bold")
        for j, cw in zip(range(5), [0.06, 0.12, 0.48, 0.13, 0.12]):
            for i in range(len(tbl_data) + 1):
                t2[(i, j)].set_width(cw)

        plt.tight_layout(rect=[0, 0, 1, 0.95])
        pdf.savefig(fig, facecolor=BG)
        plt.close(fig)

        d = pdf.infodict()
        d["Title"]        = "CEDEAR Heat Finder Report"
        d["Author"]       = "CEDEAR Portfolio AI"
        d["CreationDate"] = datetime.now()

    print(f"  PDF guardado: {output_path}")


# ── HTML DASHBOARD (self-contained: no server, no CDN, works offline) ──────────
_DASHBOARD_TEMPLATE = r"""<!doctype html>
<html lang="es">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>CEDEAR Heat Finder - Dashboard</title>
<style>
:root{
  --page:#f2f4f8; --surface:#ffffff; --surface-2:#f6f8fc; --surface-3:#eef2f9;
  --ink:#0b0b0b; --ink-2:#52514e; --muted:#8a8a8a;
  --grid:#e6e9f1; --border:rgba(11,11,11,.10);
  --accent:#1F3864; --accent-2:#2E75B6;
  --good-bg:#d8f3d8; --good-ink:#0a7d0a; --warn-bg:#fbe6cf; --warn-ink:#a05a08;
  --bad-bg:#f7d6d6; --bad-ink:#b02525; --star:#e8a91e;
  --shadow:0 1px 2px rgba(11,11,11,.05),0 6px 20px rgba(11,11,11,.06);
}
.dark{
  --page:#0d0f13; --surface:#171a20; --surface-2:#1d2129; --surface-3:#222732;
  --ink:#f4f5f7; --ink-2:#c3c2b7; --muted:#8a8a8a;
  --grid:#282d38; --border:rgba(255,255,255,.12);
  --accent:#8fb3ec; --accent-2:#5b9be0;
  --good-bg:#123a1a; --good-ink:#4fd06a; --warn-bg:#3d2a12; --warn-ink:#e6a24b;
  --bad-bg:#3d1a1a; --bad-ink:#f07070; --star:#f0bd45;
  --shadow:0 1px 2px rgba(0,0,0,.4),0 6px 20px rgba(0,0,0,.45);
}
@media (prefers-color-scheme:dark){
  :root:not(.light){
    --page:#0d0f13; --surface:#171a20; --surface-2:#1d2129; --surface-3:#222732;
    --ink:#f4f5f7; --ink-2:#c3c2b7; --muted:#8a8a8a;
    --grid:#282d38; --border:rgba(255,255,255,.12);
    --accent:#8fb3ec; --accent-2:#5b9be0;
    --good-bg:#123a1a; --good-ink:#4fd06a; --warn-bg:#3d2a12; --warn-ink:#e6a24b;
    --bad-bg:#3d1a1a; --bad-ink:#f07070; --star:#f0bd45;
    --shadow:0 1px 2px rgba(0,0,0,.4),0 6px 20px rgba(0,0,0,.45);
  }
}
*{box-sizing:border-box}
html,body{margin:0}
body{background:var(--page);color:var(--ink);
  font-family:system-ui,-apple-system,"Segoe UI",Roboto,sans-serif;font-size:14px;line-height:1.45}
.wrap{max-width:1160px;margin:0 auto;padding:22px 20px 60px}
.num{font-variant-numeric:tabular-nums}
a{color:var(--accent-2)}

header.top{display:flex;align-items:flex-start;justify-content:space-between;gap:16px;flex-wrap:wrap;margin-bottom:18px}
.title h1{margin:0;font-size:24px;letter-spacing:-.02em}
.title .sub{color:var(--ink-2);margin-top:4px}
.title .meta{color:var(--muted);font-size:12px;margin-top:6px}
.theme-btn{background:var(--surface);border:1px solid var(--border);color:var(--ink);
  border-radius:9px;padding:8px 12px;cursor:pointer;box-shadow:var(--shadow);font-size:13px}
.theme-btn:hover{background:var(--surface-2)}

.stats{display:grid;grid-template-columns:repeat(auto-fit,minmax(150px,1fr));gap:12px;margin-bottom:20px}
.stat{background:var(--surface);border:1px solid var(--border);border-radius:12px;padding:14px 16px;box-shadow:var(--shadow)}
.stat .k{color:var(--muted);font-size:11.5px;text-transform:uppercase;letter-spacing:.04em}
.stat .v{font-size:26px;font-weight:650;margin-top:4px}
.stat .v small{font-size:13px;font-weight:500;color:var(--ink-2)}

.tabs{display:flex;gap:6px;flex-wrap:wrap;border-bottom:1px solid var(--border);margin-bottom:18px}
.tab{background:none;border:0;border-bottom:2px solid transparent;color:var(--ink-2);
  padding:10px 14px;cursor:pointer;font-size:14px;font-weight:550;border-radius:6px 6px 0 0}
.tab:hover{background:var(--surface-2);color:var(--ink)}
.tab.active{color:var(--accent-2);border-bottom-color:var(--accent-2)}
.panel[hidden]{display:none}

.card{background:var(--surface);border:1px solid var(--border);border-radius:14px;
  padding:18px 18px 20px;box-shadow:var(--shadow);margin-bottom:18px}
.card h2{margin:0 0 4px;font-size:17px}
.card .hint{color:var(--muted);font-size:12.5px;margin:0 0 16px}

.legend{display:flex;gap:14px;flex-wrap:wrap;margin:2px 0 16px}
.legend .li{display:flex;align-items:center;gap:6px;font-size:12.5px;color:var(--ink-2)}
.dot{width:11px;height:11px;border-radius:3px;flex:0 0 auto}
.dot.round{border-radius:50%}

/* leaderboard */
.lb-row{display:grid;grid-template-columns:26px 66px 1fr 190px 46px;align-items:center;gap:10px;
  padding:7px 4px;border-bottom:1px solid var(--grid)}
.lb-row:last-child{border-bottom:0}
.lb-rank{color:var(--muted);font-size:12px;text-align:right}
.lb-tk{font-weight:650}
.lb-nm{color:var(--ink-2);font-size:12.5px;white-space:nowrap;overflow:hidden;text-overflow:ellipsis;display:flex;align-items:center;gap:7px}
.lb-bar{display:flex;height:16px;border-radius:5px;overflow:hidden;background:var(--surface-3);gap:2px}
.lb-seg{height:100%}
.lb-total{font-weight:700;text-align:right}

/* portfolio */
.pf-grid{display:grid;grid-template-columns:280px 1fr;gap:22px;align-items:start}
@media(max-width:720px){.pf-grid{grid-template-columns:1fr}}
.donut-wrap{display:flex;flex-direction:column;align-items:center;gap:10px}
.pf-list{width:100%}
.pf-item{display:grid;grid-template-columns:20px 62px 1fr 116px 52px;gap:9px;align-items:center;
  padding:6px 2px;border-bottom:1px solid var(--grid);font-size:13px}
.pf-item:last-child{border-bottom:0}
.wbar{height:9px;border-radius:5px;background:var(--surface-3);overflow:hidden}
.wbar>i{display:block;height:100%;border-radius:5px}

/* table */
.tools{display:flex;gap:12px;align-items:center;flex-wrap:wrap;margin-bottom:12px}
.search{flex:1;min-width:180px;background:var(--surface-2);border:1px solid var(--border);
  color:var(--ink);border-radius:9px;padding:9px 12px;font-size:13px}
.count{color:var(--muted);font-size:12.5px}
.table-wrap{overflow-x:auto;border:1px solid var(--border);border-radius:12px}
table{border-collapse:collapse;width:100%;font-size:13px;min-width:820px}
thead th{position:sticky;top:0;background:var(--accent);color:#fff;font-weight:600;
  text-align:right;padding:9px 10px;white-space:nowrap;cursor:pointer;user-select:none}
thead th.l{text-align:left}
thead th:hover{background:#274a86}
thead th .ar{opacity:.5;font-size:10px;margin-left:3px}
tbody td{padding:7px 10px;text-align:right;white-space:nowrap;border-bottom:1px solid var(--grid)}
tbody td.l{text-align:left}
tbody tr:hover{background:var(--surface-2)}
tbody tr.pf{background:color-mix(in srgb,var(--accent-2) 8%,transparent)}
tbody tr.pf:hover{background:color-mix(in srgb,var(--accent-2) 14%,transparent)}
.sctag{display:inline-flex;align-items:center;gap:6px}
.star{color:var(--star)}
.cell-good{background:var(--good-bg);color:var(--good-ink);font-weight:650;border-radius:5px}
.cell-warn{background:var(--warn-bg);color:var(--warn-ink);font-weight:650;border-radius:5px}
.cell-bad{background:var(--bad-bg);color:var(--bad-ink);font-weight:700;border-radius:5px}
.pill{display:inline-block;padding:1px 7px;border-radius:20px;font-size:11.5px;font-weight:600}

/* bubble */
.svg-box{width:100%;overflow-x:auto}
svg text{fill:var(--ink-2)}
.bx-grid{stroke:var(--grid);stroke-width:1}
.bx-axis{stroke:var(--border);stroke-width:1}
.bx-lab{fill:var(--muted);font-size:11px}
.bx-tk{fill:var(--ink);font-size:9px;font-weight:600;pointer-events:none}

#tip{position:fixed;z-index:50;display:none;pointer-events:none;background:var(--surface);
  border:1px solid var(--border);border-radius:9px;box-shadow:var(--shadow);
  padding:8px 10px;font-size:12px;max-width:240px}
#tip b{color:var(--ink)}
#tip .r{color:var(--ink-2)}
footer{color:var(--muted);font-size:11.5px;margin-top:26px;text-align:center;line-height:1.6}
</style>
</head>
<body>
<div class="wrap">
  <header class="top">
    <div class="title">
      <h1>CEDEAR Heat Finder</h1>
      <div class="sub" id="subMeth"></div>
      <div class="meta" id="subDate"></div>
    </div>
    <button class="theme-btn" id="themeBtn">Tema</button>
  </header>

  <div class="stats" id="stats"></div>

  <div class="tabs" id="tabs">
    <button class="tab active" data-p="lb">Leaderboard</button>
    <button class="tab" data-p="pf">Cartera</button>
    <button class="tab" data-p="tb">Fundamentales</button>
    <button class="tab" data-p="mp">Mapa</button>
  </div>

  <section class="panel" id="p-lb">
    <div class="card">
      <h2>Leaderboard - Top 20 por Heat Score</h2>
      <p class="hint">Cada barra descompone el puntaje en sus 5 factores. Pasa el cursor por un segmento para ver el detalle.</p>
      <div class="legend" id="lbLegend"></div>
      <div id="lbRows"></div>
    </div>
  </section>

  <section class="panel" id="p-pf" hidden>
    <div class="card">
      <h2>Cartera recomendada</h2>
      <p class="hint">Top 20 ponderado por score (softmax, 2%-15% por posicion). Color por sector.</p>
      <div class="pf-grid">
        <div class="donut-wrap"><div class="svg-box" id="donut"></div><div class="legend" id="pfLegend"></div></div>
        <div class="pf-list" id="pfList"></div>
      </div>
    </div>
  </section>

  <section class="panel" id="p-tb" hidden>
    <div class="card">
      <h2>Datos fundamentales</h2>
      <p class="hint">Toda la universo analizada. Clic en una columna para ordenar. Buscar por ticker o empresa.</p>
      <div class="tools">
        <input class="search" id="search" placeholder="Buscar ticker o empresa...">
        <span class="count" id="tbCount"></span>
      </div>
      <div class="table-wrap"><table id="tbl"><thead></thead><tbody></tbody></table></div>
      <div class="legend" style="margin-top:14px">
        <span class="li"><span class="dot" style="background:var(--good-bg)"></span>Verde: 5-25% bajo el maximo (pullback sano)</span>
        <span class="li"><span class="dot" style="background:var(--warn-bg)"></span>Naranja: &le;3% del maximo (pegado al maximo)</span>
        <span class="li"><span class="dot" style="background:var(--bad-bg)"></span>Rojo: RSI &ge; 70 (sobrecomprado)</span>
        <span class="li"><span class="star">&#9733;</span> En cartera</span>
      </div>
    </div>
  </section>

  <section class="panel" id="p-mp" hidden>
    <div class="card">
      <h2>Mapa: Valuacion vs Confianza de Analistas</h2>
      <p class="hint">Cartera top 20. Eje X = score de valuacion/crecimiento, eje Y = score de analistas, tamano = heat score, color = sector. Arriba-derecha = mejor.</p>
      <div class="svg-box" id="bubble"></div>
      <div class="legend" id="mpLegend"></div>
    </div>
  </section>

  <footer>
    Generado localmente por cedear_portfolio_ai.py - datos de Yahoo Finance.<br>
    Solo para investigacion / uso educativo. No es asesoramiento de inversion.
  </footer>
</div>
<div id="tip"></div>

<script>
const DATA = __DATA_JSON__;
const SC = DATA.sectorColors || {};
const F = DATA.factors;
const scColor = s => SC[s] || "#9aa3b2";
const root = document.documentElement;
const tip = document.getElementById('tip');
function showTip(html,e){tip.innerHTML=html;tip.style.display='block';tip.style.left=(e.clientX+14)+'px';tip.style.top=(e.clientY+14)+'px';}
function hideTip(){tip.style.display='none';}
const esc = s => (''+s).replace(/[&<>]/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;'}[c]));

const fmt = {
  n1:v=>v==null?'—':(+v).toFixed(1),
  n2:v=>v==null?'—':(+v).toFixed(2),
  n0:v=>v==null?'—':(''+Math.round(v)),
  pdec:v=>v==null?'—':(v*100).toFixed(1)+'%',
  praw:v=>v==null?'—':(+v).toFixed(1)+'%',
};

/* ---------- header + stats ---------- */
document.getElementById('subMeth').textContent = DATA.methodology;
document.getElementById('subDate').textContent = 'Generado: ' + (DATA.generatedAt||'').replace('T',' ');
(function stats(){
  const pf = DATA.rows.filter(r=>r.inPortfolio);
  const top = DATA.rows[0]||{};
  const ob = pf.filter(r=>r.overbought).length;
  const pb = pf.filter(r=>r.pctBelow!=null && r.pctBelow>=5 && r.pctBelow<=25).length;
  const cards = [
    ['Analizados', DATA.analyzed],
    ['En cartera', DATA.nPortfolio],
    ['Top score', (top.total!=null?top.total.toFixed(1):'—')+' <small>'+esc(top.byma||'')+'</small>'],
    ['En pullback <small>(cartera)</small>', pb],
    ['Sobrecomprados <small>(cartera)</small>', ob],
  ];
  document.getElementById('stats').innerHTML = cards.map(c=>
    '<div class="stat"><div class="k">'+c[0]+'</div><div class="v num">'+c[1]+'</div></div>').join('');
})();

/* ---------- leaderboard ---------- */
(function leaderboard(){
  document.getElementById('lbLegend').innerHTML = F.map(f=>
    '<span class="li"><span class="dot" style="background:'+f.color+'"></span>'+esc(f.label)+' <span class="num" style="color:var(--muted)">('+f.max+')</span></span>').join('');
  const top = DATA.rows.slice(0,20);
  document.getElementById('lbRows').innerHTML = top.map(r=>{
    const segs = F.map(f=>{
      const v = r[f.key]||0;
      return '<div class="lb-seg" style="width:'+v+'%;background:'+f.color+'" '+
        'data-tip="<b>'+esc(f.label)+'</b><br><span class=r>'+esc(r.byma)+' &middot; '+v.toFixed(1)+' / '+f.max+' pts</span>"></div>';
    }).join('');
    return '<div class="lb-row">'+
      '<div class="lb-rank num">'+r.rank+'</div>'+
      '<div class="lb-tk">'+esc(r.byma)+'</div>'+
      '<div class="lb-nm"><span class="dot round" style="background:'+scColor(r.sector)+'"></span>'+esc(r.name)+'</div>'+
      '<div class="lb-bar">'+segs+'</div>'+
      '<div class="lb-total num">'+ (r.total!=null?r.total.toFixed(1):'—') +'</div>'+
    '</div>';
  }).join('');
  document.querySelectorAll('#lbRows .lb-seg').forEach(el=>{
    el.addEventListener('mousemove',e=>showTip(el.dataset.tip,e));
    el.addEventListener('mouseleave',hideTip);
  });
})();

/* ---------- portfolio: donut + list ---------- */
(function portfolio(){
  const pf = DATA.portfolio;
  const cx=120,cy=120,rO=104,rI=64;
  const polar=(r,a)=>{const t=(a-90)*Math.PI/180;return [cx+r*Math.cos(t),cy+r*Math.sin(t)];};
  const seg=(a0,a1)=>{const lg=(a1-a0)>180?1:0;const[x0,y0]=polar(rO,a0),[x1,y1]=polar(rO,a1),
    [x2,y2]=polar(rI,a1),[x3,y3]=polar(rI,a0);
    return `M${x0} ${y0} A${rO} ${rO} 0 ${lg} 1 ${x1} ${y1} L${x2} ${y2} A${rI} ${rI} 0 ${lg} 0 ${x3} ${y3} Z`;};
  let ang=0,paths='';
  pf.forEach(p=>{const a0=ang,a1=ang+p.weight/100*360;ang=a1;
    paths+='<path d="'+seg(a0,a1)+'" fill="'+scColor(p.sector)+'" stroke="var(--surface)" stroke-width="2" '+
      'data-tip="<b>'+esc(p.byma)+'</b> '+esc(p.name)+'<br><span class=r>'+p.weight.toFixed(1)+'% &middot; '+esc(p.sector)+' &middot; score '+(p.total!=null?p.total.toFixed(1):'—')+'</span>"></path>';});
  document.getElementById('donut').innerHTML =
    '<svg viewBox="0 0 240 240" width="240" height="240" role="img">'+paths+
    '<text x="120" y="114" text-anchor="middle" style="font-size:30px;font-weight:700;fill:var(--ink)">'+DATA.nPortfolio+'</text>'+
    '<text x="120" y="134" text-anchor="middle" style="font-size:12px;fill:var(--muted)">posiciones</text></svg>';

  const secs=[...new Set(pf.map(p=>p.sector))];
  document.getElementById('pfLegend').innerHTML = secs.map(s=>
    '<span class="li"><span class="dot round" style="background:'+scColor(s)+'"></span>'+esc(s)+'</span>').join('');

  const maxW=Math.max.apply(null,pf.map(p=>p.weight));
  document.getElementById('pfList').innerHTML = pf.map((p,i)=>
    '<div class="pf-item">'+
      '<span class="num" style="color:var(--muted)">'+(i+1)+'</span>'+
      '<span style="font-weight:650">'+esc(p.byma)+'</span>'+
      '<span style="color:var(--ink-2);overflow:hidden;text-overflow:ellipsis;white-space:nowrap">'+esc(p.name)+'</span>'+
      '<span class="wbar"><i style="width:'+(p.weight/maxW*100)+'%;background:'+scColor(p.sector)+'"></i></span>'+
      '<span class="num" style="font-weight:650;text-align:right">'+p.weight.toFixed(1)+'%</span>'+
    '</div>').join('');
  document.querySelectorAll('#donut path').forEach(el=>{
    el.addEventListener('mousemove',e=>showTip(el.dataset.tip,e));
    el.addEventListener('mouseleave',hideTip);
  });
})();

/* ---------- fundamentals table ---------- */
const COLS=[
  {k:'inPortfolio',t:'',ty:'star',s:false},
  {k:'byma',t:'BYMA',ty:'txt'},
  {k:'name',t:'Empresa',ty:'txt'},
  {k:'sector',t:'Sector',ty:'sec'},
  {k:'total',t:'Score',ty:'n1'},
  {k:'peg',t:'PEG',ty:'n2'},
  {k:'fpe',t:'F-P/E',ty:'n1'},
  {k:'rev',t:'Rev%',ty:'pdec'},
  {k:'eps',t:'EPS%',ty:'pdec'},
  {k:'roe',t:'ROE%',ty:'pdec'},
  {k:'consensus',t:'Consenso',ty:'txt'},
  {k:'upside',t:'Upside%',ty:'praw'},
  {k:'pctBelow',t:'% < Max',ty:'pbelow'},
  {k:'rsi',t:'RSI',ty:'rsi'},
];
let sortK='total', sortD=-1, query='';
function headRow(){
  return '<tr>'+COLS.map(c=>{
    const l=(c.ty==='txt'||c.ty==='sec')?' l':'';
    const ar=(c.k===sortK)?'<span class="ar">'+(sortD<0?'▼':'▲')+'</span>':'';
    return '<th class="'+l.trim()+'" data-k="'+c.k+'" data-ty="'+c.ty+'" '+(c.s===false?'style="cursor:default"':'')+'>'+c.t+ar+'</th>';
  }).join('')+'</tr>';
}
function cell(r,c){
  const v=r[c.k];
  if(c.ty==='star') return '<td>'+(v?'<span class="star" title="En cartera">★</span>':'')+'</td>';
  if(c.ty==='txt') return '<td class="l">'+(v==null?'—':esc(v))+'</td>';
  if(c.ty==='sec') return '<td class="l"><span class="sctag"><span class="dot round" style="background:'+scColor(v)+'"></span>'+esc(v)+'</span></td>';
  if(c.ty==='pbelow'){
    if(v==null) return '<td class="num">—</td>';
    let cl=''; if(v<=3)cl='cell-warn'; else if(v>=5&&v<=25)cl='cell-good';
    return '<td class="num"><span class="'+cl+'" style="padding:2px 6px">'+v.toFixed(1)+'%</span></td>';
  }
  if(c.ty==='rsi'){
    if(v==null) return '<td class="num">—</td>';
    const cl=r.overbought?'cell-bad':'';
    return '<td class="num"><span class="'+cl+'" style="padding:2px 6px">'+Math.round(v)+'</span></td>';
  }
  return '<td class="num">'+fmt[c.ty](v)+'</td>';
}
function cmp(a,b){
  let x=a[sortK],y=b[sortK];
  const na=(x==null||x===''),nb=(y==null||y==='');
  if(na&&nb)return 0; if(na)return 1; if(nb)return -1;
  if(typeof x==='string'){x=x.toLowerCase();y=(''+y).toLowerCase();return sortD*(x<y?-1:x>y?1:0);}
  return sortD*(x-y);
}
function renderTable(){
  document.querySelector('#tbl thead').innerHTML=headRow();
  let rows=DATA.rows;
  if(query){const q=query.toLowerCase();
    rows=rows.filter(r=>(r.byma&&r.byma.toLowerCase().includes(q))||(r.name&&r.name.toLowerCase().includes(q)));}
  rows=rows.slice().sort(cmp);
  document.querySelector('#tbl tbody').innerHTML=rows.map(r=>
    '<tr class="'+(r.inPortfolio?'pf':'')+'">'+COLS.map(c=>cell(r,c)).join('')+'</tr>').join('');
  document.getElementById('tbCount').textContent=rows.length+' de '+DATA.rows.length+' tickers';
  document.querySelectorAll('#tbl thead th').forEach(th=>{
    if(th.dataset.k==='inPortfolio')return;
    th.onclick=()=>{const k=th.dataset.k;
      if(k===sortK)sortD=-sortD;
      else{sortK=k;sortD=(th.dataset.ty==='txt'||th.dataset.ty==='sec')?1:-1;}
      renderTable();};
  });
}
document.getElementById('search').addEventListener('input',e=>{query=e.target.value;renderTable();});
renderTable();

/* ---------- bubble map ---------- */
(function bubble(){
  const pts=DATA.rows.filter(r=>r.inPortfolio && r.val!=null && r.ana!=null);
  const W=760,H=470,m={l:58,r:24,t:18,b:48};
  const xs=pts.map(p=>p.val),ys=pts.map(p=>p.ana),ts=pts.map(p=>p.total||0);
  const pad=(a,f)=>{const lo=Math.min.apply(null,a),hi=Math.max.apply(null,a),d=(hi-lo)||1;return [lo-d*f,hi+d*f];};
  const[x0,x1]=pad(xs,.18),[y0,y1]=pad(ys,.18),tMin=Math.min.apply(null,ts),tMax=Math.max.apply(null,ts);
  const sx=v=>m.l+(v-x0)/(x1-x0)*(W-m.l-m.r);
  const sy=v=>H-m.b-(v-y0)/(y1-y0)*(H-m.t-m.b);
  const sr=v=>8+(tMax>tMin?(v-tMin)/(tMax-tMin):.5)*20;
  const ticks=(lo,hi,n)=>{const st=(hi-lo)/n,out=[];for(let i=0;i<=n;i++)out.push(lo+st*i);return out;};
  let g='';
  ticks(x0,x1,5).forEach(t=>{const x=sx(t);g+='<line class="bx-grid" x1="'+x+'" y1="'+m.t+'" x2="'+x+'" y2="'+(H-m.b)+'"></line>'+
    '<text class="bx-lab" x="'+x+'" y="'+(H-m.b+16)+'" text-anchor="middle">'+t.toFixed(0)+'</text>';});
  ticks(y0,y1,5).forEach(t=>{const y=sy(t);g+='<line class="bx-grid" x1="'+m.l+'" y1="'+y+'" x2="'+(W-m.r)+'" y2="'+y+'"></line>'+
    '<text class="bx-lab" x="'+(m.l-8)+'" y="'+(y+3)+'" text-anchor="end">'+t.toFixed(0)+'</text>';});
  g+='<line class="bx-axis" x1="'+m.l+'" y1="'+(H-m.b)+'" x2="'+(W-m.r)+'" y2="'+(H-m.b)+'"></line>';
  g+='<line class="bx-axis" x1="'+m.l+'" y1="'+m.t+'" x2="'+m.l+'" y2="'+(H-m.b)+'"></line>';
  g+='<text class="bx-lab" x='+((m.l+W-m.r)/2)+' y='+(H-8)+' text-anchor="middle">Score Valuacion / Crecimiento (max 30)</text>';
  g+='<text class="bx-lab" transform="translate(15,'+((m.t+H-m.b)/2)+') rotate(-90)" text-anchor="middle">Score Analistas (max 25)</text>';
  let circ='';
  pts.forEach(p=>{const x=sx(p.val),y=sy(p.ana),r=sr(p.total||0);
    circ+='<circle cx='+x+' cy='+y+' r='+r+' fill="'+scColor(p.sector)+'" fill-opacity=".82" stroke="var(--surface)" stroke-width="2" '+
      'data-tip="<b>'+esc(p.byma)+'</b> '+esc(p.name)+'<br><span class=r>Val '+p.val.toFixed(1)+' &middot; Ana '+p.ana.toFixed(1)+' &middot; score '+(p.total!=null?p.total.toFixed(1):'—')+'</span>"></circle>'+
      '<text class="bx-tk" x='+x+' y='+(y+3)+' text-anchor="middle">'+esc(p.byma)+'</text>';});
  document.getElementById('bubble').innerHTML='<svg viewBox="0 0 '+W+' '+H+'" width="100%" style="max-width:'+W+'px">'+g+circ+'</svg>';
  const secs=[...new Set(pts.map(p=>p.sector))];
  document.getElementById('mpLegend').innerHTML=secs.map(s=>
    '<span class="li"><span class="dot round" style="background:'+scColor(s)+'"></span>'+esc(s)+'</span>').join('');
  document.querySelectorAll('#bubble circle').forEach(el=>{
    el.addEventListener('mousemove',e=>showTip(el.dataset.tip,e));
    el.addEventListener('mouseleave',hideTip);
  });
})();

/* ---------- tabs + theme ---------- */
document.getElementById('tabs').addEventListener('click',e=>{
  const b=e.target.closest('.tab'); if(!b)return;
  document.querySelectorAll('.tab').forEach(t=>t.classList.toggle('active',t===b));
  document.querySelectorAll('.panel').forEach(p=>p.hidden=(p.id!=='p-'+b.dataset.p));
});
document.getElementById('themeBtn').addEventListener('click',()=>{
  const dark=matchMedia('(prefers-color-scheme: dark)').matches;
  const cur=root.classList.contains('dark')||(dark&&!root.classList.contains('light'));
  root.classList.remove('dark','light');
  root.classList.add(cur?'light':'dark');
});
</script>
</body>
</html>"""


def generate_html_dashboard(scores_df: pd.DataFrame, weights: dict,
                            universe_df: pd.DataFrame, html_path: str) -> None:
    """Write a self-contained interactive HTML dashboard (data baked in)."""
    print("  Generando dashboard HTML...")
    idx = universe_df.set_index("yf_ticker")

    def byma(t):   return str(idx.loc[t, "byma_ticker"]) if t in idx.index else str(t)
    def uname(t):  return str(idx.loc[t, "name"])        if t in idx.index else str(t)
    def sector(t): return str(idx.loc[t, "sector"])      if t in idx.index else "Otro"

    def num(v):
        if v is None:
            return None
        try:
            f = float(v)
            return None if (math.isnan(f) or math.isinf(f)) else f
        except (TypeError, ValueError):
            return None

    def rnd(v, d=1):
        v = num(v)
        return None if v is None else round(v, d)

    def rec_label(v):
        v = num(v)
        if v is None:      return None
        if v <= 1.5:       return "Compra fuerte"
        if v <= 2.5:       return "Comprar"
        if v <= 3.5:       return "Mantener"
        return "Vender"

    rows = []
    for rank, (t, row) in enumerate(scores_df.iterrows(), start=1):
        price  = num(row.get("price"))
        target = num(row.get("target"))
        upside = ((target - price) / price * 100) if (price and target and price > 0) else None
        pbh    = num(row.get("pct_below_high"))
        nan    = num(row.get("n_analysts"))
        rows.append({
            "rank": rank, "byma": byma(t), "name": uname(t), "sector": sector(t),
            "total": rnd(row.get("total")),
            "val": rnd(row.get("Valuacion")), "ana": rnd(row.get("Analistas")),
            "ear": rnd(row.get("Earnings")), "mom": rnd(row.get("Momentum")),
            "tec": rnd(row.get("Tecnico")),
            "peg": rnd(row.get("peg"), 2), "fpe": rnd(row.get("fwd_pe"), 1),
            "rev": rnd(row.get("rev_growth"), 4), "eps": rnd(row.get("eps_growth"), 4),
            "roe": rnd(row.get("roe"), 4),
            "consensus": rec_label(row.get("rec_mean")),
            "nanalysts": (int(nan) if nan is not None else None),
            "upside": rnd(upside, 1),
            "pctBelow": (round(abs(pbh), 1) if pbh is not None else None),
            "rsi": rnd(row.get("rsi"), 0),
            "overbought": bool(row.get("overbought", False)),
            "inPortfolio": (t in weights),
            "weight": (round(num(weights.get(t)) * 100, 2) if t in weights else None),
        })

    portfolio = [{
        "byma": byma(t), "name": uname(t), "sector": sector(t),
        "weight": round(w * 100, 2),
        "total": rnd(scores_df.loc[t, "total"]) if t in scores_df.index else None,
    } for t, w in sorted(weights.items(), key=lambda x: -x[1])]

    factors = [
        {"key": "val", "label": "Valuacion", "max": MAX_VALUACION, "color": SCORE_CLR["Valuacion"]},
        {"key": "ana", "label": "Analistas", "max": MAX_ANALISTAS, "color": SCORE_CLR["Analistas"]},
        {"key": "ear", "label": "Earnings",  "max": MAX_EARNINGS,  "color": SCORE_CLR["Earnings"]},
        {"key": "mom", "label": "Momentum",  "max": MAX_MOMENTUM,  "color": SCORE_CLR["Momentum"]},
        {"key": "tec", "label": "Tecnico",   "max": MAX_TECNICO,   "color": SCORE_CLR["Tecnico"]},
    ]

    data = {
        "generatedAt": datetime.now().isoformat(timespec="minutes"),
        "methodology": "Heat score 5 factores: Valuacion(30) + Analistas(25) + Earnings(20) + Momentum(15) + Tecnico(10)",
        "analyzed": int(len(scores_df)),
        "nPortfolio": int(len(weights)),
        "factors": factors,
        "sectorColors": SECTOR_COLORS,
        "rows": rows,
        "portfolio": portfolio,
    }
    blob = json.dumps(data, ensure_ascii=False).replace("<", "\\u003c")
    html = _DASHBOARD_TEMPLATE.replace("__DATA_JSON__", blob)
    with open(html_path, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"  Dashboard guardado: {html_path}")


# ── MAIN ──────────────────────────────────────────────────────────────────────
def main():
    sep = "=" * 72
    print(sep)
    print("  CEDEAR Heat Finder  -  Analisis Fundamental Mid-Short Term")
    print(sep)

    universe_df = pd.DataFrame(
        UNIVERSE, columns=["byma_ticker", "yf_ticker", "name", "exchange", "sector"]
    ).drop_duplicates(subset="yf_ticker")

    yf_tickers = universe_df["yf_ticker"].unique().tolist()

    # 1. Price data
    print("\n[1/5] Descargando precios...")
    prices       = fetch_prices(yf_tickers, lookback_years=1)
    price_metrics = compute_price_metrics(prices)
    print(f"  Tickers con metricas de precio: {len(price_metrics)}")

    # 2. Fundamentals
    print("\n[2/5] Descargando fundamentals...")
    fundamentals = fetch_fundamentals(yf_tickers)
    print(f"  Fundamentals descargados: {len(fundamentals)}")

    # 3. Score
    print("\n[3/5] Calculando Heat Scores...")
    scores_df = compute_all_scores(fundamentals, price_metrics)
    print(f"  Tickers con score: {len(scores_df)}")
    idx_u = universe_df.set_index("yf_ticker")
    print("\nTop 10 por Heat Score:")
    print(f"  {'BYMA':<8} {'Score':>6}  {'Val':>5}  {'Ana':>5}  {'Ear':>5}  {'Mom':>5}  {'Tec':>5}  Empresa")
    print("  " + "-" * 65)
    for t, row in scores_df.head(10).iterrows():
        b = idx_u.loc[t, "byma_ticker"] if t in idx_u.index else t
        n = idx_u.loc[t, "name"][:24]   if t in idx_u.index else t
        print(f"  {b:<8} {row['total']:>6.1f}  {row['Valuacion']:>5.1f}  {row['Analistas']:>5.1f}  "
              f"{row['Earnings']:>5.1f}  {row['Momentum']:>5.1f}  {row['Tecnico']:>5.1f}  {n}")

    # 4. Build portfolio
    print(f"\n[4/5] Construyendo portfolio (top {N_PICKS})...")
    weights = build_portfolio(scores_df, n=N_PICKS)
    print(f"\n  {'BYMA':<10} {'Peso':>6}  {'Score':>6}  Empresa")
    print("  " + "-" * 50)
    for t, w in weights.items():
        b = idx_u.loc[t, "byma_ticker"] if t in idx_u.index else t
        n = idx_u.loc[t, "name"][:28]   if t in idx_u.index else t
        sc = scores_df.loc[t, "total"] if t in scores_df.index else 0
        print(f"  {b:<10} {w:>6.1%}  {sc:>6.1f}  {n}")

    # 5. Generate PDF
    print(f"\n[5/5] Generando reporte PDF...")
    here      = os.path.dirname(os.path.abspath(__file__))
    pdf_path  = os.path.join(here, "cedear_heat_report.pdf")
    json_path = os.path.join(here, "cedear_heat_result.json")

    generate_pdf_report(scores_df, weights, universe_df, pdf_path)

    # Save JSON
    result = {
        "generated_at": datetime.now().isoformat(),
        "methodology":  "5-factor heat score: Valuacion(30) + Analistas(25) + Earnings(20) + Momentum(15) + Tecnico(10)",
        "portfolio":    weights,
        "scores": {
            t: {
                "total":     float(row["total"]),
                "Valuacion": float(row["Valuacion"]),
                "Analistas": float(row["Analistas"]),
                "Earnings":  float(row["Earnings"]),
                "Momentum":  float(row["Momentum"]),
                "Tecnico":   float(row["Tecnico"]),
                "rsi":              (None if row.get("rsi") is None or (isinstance(row.get("rsi"), float) and np.isnan(row.get("rsi"))) else round(float(row["rsi"]), 1)),
                "pct_below_52w_high": (None if row.get("pct_below_high") is None or (isinstance(row.get("pct_below_high"), float) and np.isnan(row.get("pct_below_high"))) else round(abs(float(row["pct_below_high"])), 1)),
                "overbought":       bool(row.get("overbought", False)),
            }
            for t, row in scores_df.head(N_PICKS).iterrows()
        },
    }
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    html_path = os.path.join(here, "dashboard.html")
    generate_html_dashboard(scores_df, weights, universe_df, html_path)

    print(f"\n[OK] PDF       -> {pdf_path}")
    print(f"[OK] JSON      -> {json_path}")
    print(f"[OK] Dashboard -> {html_path}")


if __name__ == "__main__":
    main()
