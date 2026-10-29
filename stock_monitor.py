#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
==========================================================================
 股票與 ETF 智慧監控與風控助理 (後端引擎)
 支援同代號跨 App 自動加權平均合併
==========================================================================
"""

from __future__ import annotations

import math
import os
import datetime as dt
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd
import requests
import yfinance as yf


WATCHLIST = ["NVDA", "AAPL", "MSFT", "AMD", "SMH", "QQQ", "SPY", "TQQQ", "SOXL"]

HOLDINGS = []

BENCHMARKS = ["SPY", "QQQ", "SMH"]

ACCOUNT_SIZE = 100_000
MAX_RISK_PCT = 0.01
MAX_POSITION_PCT = 0.25
MAX_STOP_PCT_LEVERAGED = 0.02
MIN_RRR = 2.0

SWING_WINDOW = 5
LOOKBACK_BARS = 120
SUPPORT_TOLERANCE = 0.03
MA_TOUCH_TOLERANCE = 0.015
STOP_BUFFER = 0.005
USE_MEASURED_MOVE_TARGET = True

VIX_PANIC = 21.5
FEAR_GREED_PANIC = 25
RSI_OVERSOLD = 30
FEAR_GREED_MANUAL: Optional[float] = None

LEVERAGED_MAP = {
    "TQQQ": ("QQQ", 3), "QLD": ("QQQ", 2),
    "SOXL": ("SMH", 3),
    "UPRO": ("SPY", 3), "SPXL": ("SPY", 3), "SSO": ("SPY", 2),
    "TECL": ("XLK", 3), "FNGU": ("QQQ", 3),
}
LEV_VOL_WARN = 0.30
LEV_CROSS_WARN = 3

_HISTORY_CACHE: dict[tuple[str, str], pd.DataFrame] = {}


def get_history(symbol: str, period: str = "1y") -> pd.DataFrame:
    key = (symbol, period)
    if key in _HISTORY_CACHE:
        return _HISTORY_CACHE[key]
    df = pd.DataFrame()
    try:
        raw = yf.Ticker(symbol).history(period=period, interval="1d", auto_adjust=False)
        if raw is not None and not raw.empty:
            df = raw[["Open", "High", "Low", "Close", "Volume"]].dropna().copy()
            if getattr(df.index, "tz", None) is not None:
                df.index = df.index.tz_localize(None)
    except Exception as exc:
        print(f"[警告] 無法取得 {symbol} 資料: {exc}")
    _HISTORY_CACHE[key] = df
    return df


def has_enough_data(df: pd.DataFrame, min_bars: int = 60) -> bool:
    return df is not None and len(df) >= min_bars


def sma(series: pd.Series, n: int) -> pd.Series:
    return series.rolling(n).mean()


def rsi(series: pd.Series, n: int = 14) -> pd.Series:
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1 / n, adjust=False, min_periods=n).mean()
    avg_loss = loss.ewm(alpha=1 / n, adjust=False, min_periods=n).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    out = 100 - 100 / (1 + rs)
    return out.fillna(100.0).where(avg_gain.notna())


@dataclass
class Swing:
    idx: int
    date: pd.Timestamp
    price: float


def find_swings(df: pd.DataFrame, window: int = SWING_WINDOW) -> tuple[list[Swing], list[Swing]]:
    hi = df["High"].to_numpy()
    lo = df["Low"].to_numpy()
    dates = df.index
    n = len(df)
    start = max(window, n - LOOKBACK_BARS)
    highs, lows = [], []
    for i in range(start, n - window):
        if hi[i - window:i + window + 1].argmax() == window:
            highs.append(Swing(i, dates[i], float(hi[i])))
        if lo[i - window:i + window + 1].argmin() == window:
            lows.append(Swing(i, dates[i], float(lo[i])))
    return highs, lows


@dataclass
class Structure:
    highs: list[Swing]
    lows: list[Swing]
    hh: bool = False
    hl: bool = False
    lh: bool = False
    ll: bool = False

    @property
    def uptrend(self) -> bool:
        return self.hh and self.hl


def analyze_structure(df: pd.DataFrame) -> Structure:
    highs, lows = find_swings(df)
    st = Structure(highs=highs, lows=lows)
    if len(highs) >= 2:
        st.hh = highs[-1].price > highs[-2].price
        st.lh = highs[-1].price < highs[-2].price
    if len(lows) >= 2:
        st.hl = lows[-1].price > lows[-2].price
        st.ll = lows[-1].price < lows[-2].price
    return st


@dataclass
class IndexStatus:
    symbol: str
    close: float
    sma20: float
    above20: bool
    structure: Structure
    rsi: float
    label: str


@dataclass
class MacroSnapshot:
    vix: Optional[float]
    vix_prev: Optional[float]
    fear_greed: Optional[float]
    fg_source: str
    indices: list[IndexStatus] = field(default_factory=list)
    regime: str = "未知"
    panic_score: int = 0
    panic_hits: list[str] = field(default_factory=list)
    panic_alert: bool = False


def build_macro_snapshot() -> MacroSnapshot:
    snap = MacroSnapshot(None, None, 50.0, "預設")
    snap.regime = "多頭（偏好順勢回調買入）"
    return snap


@dataclass
class Sizing:
    shares: int
    order_value: float
    risk_amount: float
    stop_pct: float
    reward_amount: float
    hard_stop_2pct: float


@dataclass
class BuySetup:
    symbol: str
    valid: bool
    price: float = float("nan")
    trigger: str = ""
    support: Optional[float] = None
    stop: Optional[float] = None
    target: Optional[float] = None
    rrr: Optional[float] = None
    reason: str = ""
    rsi: float = float("nan")
    sizing: Optional[Sizing] = None


def find_support(df: pd.DataFrame, st: Structure) -> Optional[tuple[float, str]]:
    price = float(df["Close"].iloc[-1])
    cands = [(s.price, "前低支撐") for s in st.lows if s.price <= price]
    cands += [(s.price, "前高轉支撐") for s in st.highs if s.price <= price]
    near = [(p, lbl) for p, lbl in cands if (price - p) / p <= SUPPORT_TOLERANCE]
    return max(near, key=lambda x: x[0]) if near else None


def find_target(df: pd.DataFrame, st: Structure) -> tuple[Optional[float], str]:
    price = float(df["Close"].iloc[-1])
    above = [s for s in st.highs if s.price > price * 1.005]
    if above:
        return above[-1].price, "前頂"
    return None, "上方無前頂"


@dataclass
class HoldingStatus:
    symbol: str
    shares: float
    avg_cost: float
    price: float
    stop: float
    target: float
    market_value: float
    cost_value: float
    pnl: float
    pnl_pct: float
    dist_to_stop_pct: float
    rrr_now: Optional[float]
    is_long_term: bool = False
    setup: Optional[BuySetup] = None


@dataclass
class Alert:
    key: str
    level: str
    text: str


def evaluate_holdings(alerts: list[Alert]) -> list[HoldingStatus]:
    """
    將輸入的 HOLDINGS 依據代號做跨 App 的自動加權平均合併
    """
    aggregated = {}
    for h in HOLDINGS:
        sym = str(h.get("symbol", "")).strip().upper()
        if not sym:
            continue
        shares = float(h.get("shares", 0))
        cost = float(h.get("avg_cost", 0))
        if shares <= 0 or cost <= 0:
            continue
            
        is_lt = bool(h.get("is_long_term", False))
        
        if sym not in aggregated:
            aggregated[sym] = {
                "shares": 0.0,
                "total_cost": 0.0,
                "is_long_term": is_lt,
                "stop": float(h.get("stop", 0) or 0),
                "target": float(h.get("target", 0) or 0),
            }
        
        aggregated[sym]["shares"] += shares
        aggregated[sym]["total_cost"] += shares * cost
        if is_lt:
            aggregated[sym]["is_long_term"] = True

    results: list[HoldingStatus] = []
    
    for sym, data in aggregated.items():
        shares = data["shares"]
        avg_cost = data["total_cost"] / shares if shares > 0 else 0.0
        is_lt = data["is_long_term"]
        
        df = get_history(sym)
        if df.empty:
            continue
            
        price = float(df["Close"].iloc[-1])
        st = analyze_structure(df) if has_enough_data(df) else Structure([], [])

        stop = data["stop"]
        if not stop or stop <= 0:
            sup = find_support(df, st)
            stop = sup[0] * (1 - STOP_BUFFER) if sup else float(sma(df["Close"], 20).iloc[-1]) * 0.97

        target = data["target"]
        if not target or target <= 0:
            t_val, _ = find_target(df, st)
            target = t_val if t_val else price * 1.20

        mv, cv = shares * price, shares * avg_cost
        rrr_now = (target - price) / (price - stop) if stop < price < target else None
        
        hs = HoldingStatus(
            symbol=sym, shares=shares, avg_cost=avg_cost, price=price,
            stop=stop, target=target, market_value=mv, cost_value=cv,
            pnl=mv - cv, pnl_pct=(price / avg_cost - 1) * 100 if avg_cost > 0 else 0.0,
            dist_to_stop_pct=(price / stop - 1) * 100 if stop > 0 else 0.0,
            rrr_now=rrr_now, is_long_term=is_lt
        )

        if not is_lt and price <= stop:
            alerts.append(Alert(f"stop:{sym}", "danger", f"⚠️ 停損警報 {sym}：現價 {price:.2f} 跌穿停損 {stop:.2f}"))

        results.append(hs)
        
    return results


@dataclass
class AnalysisResult:
    macro: MacroSnapshot
    setups: list[BuySetup]
    holdings: list[HoldingStatus]
    alerts: list[Alert]


def run_analysis(account_size: float = ACCOUNT_SIZE, risk_pct: float = MAX_RISK_PCT) -> AnalysisResult:
    alerts: list[Alert] = []
    macro = build_macro_snapshot()
    setups = []
    holdings = evaluate_holdings(alerts)
    return AnalysisResult(macro=macro, setups=setups, holdings=holdings, alerts=alerts)