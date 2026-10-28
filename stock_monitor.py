#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
==========================================================================
 股票與 ETF 智慧監控與風控助理 (網頁與命令列雙用引擎)
==========================================================================
"""

from __future__ import annotations

import argparse
import json
import math
import os
import time
import datetime as dt
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd
import requests
import yfinance as yf


# ==========================================================================
# 1. CONFIG 設定區
# ==========================================================================

WATCHLIST = ["NVDA", "AAPL", "MSFT", "AMD", "SMH", "QQQ", "SPY", "TQQQ", "SOXL"]

HOLDINGS = [
    {"symbol": "SMH",  "shares": 20,  "avg_cost": 290.0, "stop": 0.0, "target": 0.0, "is_long_term": False},
    {"symbol": "NVDA", "shares": 30,  "avg_cost": 120.0, "stop": 0.0, "target": 0.0, "is_long_term": False},
    {"symbol": "VOO",  "shares": 50,  "avg_cost": 450.0, "stop": 0.0, "target": 0.0, "is_long_term": True},
]

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

TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "")
DISCORD_WEBHOOK_URL = os.getenv("DISCORD_WEBHOOK_URL", "")
ALERT_STATE_FILE = ".alert_state.json"
CNN_FEAR_GREED_URL = "https://production.dataviz.cnn.io/index/fearandgreed/graphdata"


# ==========================================================================
# 2. DATA LAYER & INDICATORS
# ==========================================================================

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


def hist_vol(close: pd.Series, n: int = 20) -> float:
    r = np.log(close / close.shift(1)).dropna()
    if len(r) < n:
        return float("nan")
    return float(r.iloc[-n:].std() * math.sqrt(252))


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

    @property
    def downtrend(self) -> bool:
        return self.lh and self.ll


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


def uptrend_line_value(st: Structure, n_bars: int) -> Optional[float]:
    if len(st.lows) < 2:
        return None
    a, b = st.lows[-2], st.lows[-1]
    if b.idx == a.idx:
        return None
    slope = (b.price - a.price) / (b.idx - a.idx)
    if slope <= 0:
        return None
    return b.price + slope * (n_bars - 1 - b.idx)


# ==========================================================================
# 3. MARKET REGIME
# ==========================================================================

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


def analyze_index(symbol: str) -> Optional[IndexStatus]:
    df = get_history(symbol)
    if not has_enough_data(df):
        return None
    close = df["Close"]
    s20 = sma(close, 20)
    st = analyze_structure(df)
    c, m = float(close.iloc[-1]), float(s20.iloc[-1])
    above = c > m
    if st.uptrend and above:
        label = "上升趨勢 ✅"
    elif st.downtrend:
        label = "下降趨勢 ❌"
    else:
        label = "震盪／趨勢未明 ⚠️"
    return IndexStatus(symbol, c, m, above, st, float(rsi(close).iloc[-1]), label)


def fetch_vix() -> tuple[Optional[float], Optional[float]]:
    df = get_history("^VIX", period="6mo")
    if df.empty:
        return None, None
    last = float(df["Close"].iloc[-1])
    prev = float(df["Close"].iloc[-2]) if len(df) > 1 else None
    return last, prev


def estimate_fear_greed(vix: Optional[float]) -> Optional[float]:
    spy = get_history("SPY", period="1y")
    if vix is None or not has_enough_data(spy, 130):
        return None
    vix_part = float(np.clip((30 - vix) / (30 - 12) * 100, 0, 100))
    ratio = float(spy["Close"].iloc[-1] / spy["Close"].rolling(125).mean().iloc[-1] - 1)
    mom_part = float(np.clip((ratio + 0.10) / 0.20 * 100, 0, 100))
    return round(0.5 * vix_part + 0.5 * mom_part, 1)


def fetch_fear_greed(vix: Optional[float]) -> tuple[Optional[float], str]:
    if FEAR_GREED_MANUAL is not None:
        return float(FEAR_GREED_MANUAL), "手動輸入"
    try:
        r = requests.get(CNN_FEAR_GREED_URL, headers={"User-Agent": "Mozilla/5.0"}, timeout=10)
        r.raise_for_status()
        return float(r.json()["fear_and_greed"]["score"]), "CNN"
    except Exception:
        est = estimate_fear_greed(vix)
        return est, "估算(非CNN官方)" if est is not None else "無資料"


def build_macro_snapshot() -> MacroSnapshot:
    vix, vix_prev = fetch_vix()
    fg, fg_src = fetch_fear_greed(vix)
    snap = MacroSnapshot(vix, vix_prev, fg, fg_src)
    snap.indices = [s for s in (analyze_index(b) for b in BENCHMARKS) if s is not None]

    ups = sum(1 for s in snap.indices if s.structure.uptrend and s.above20)
    downs = sum(1 for s in snap.indices if s.structure.downtrend)
    total = len(snap.indices)
    if total == 0:
        snap.regime = "無資料"
    elif ups >= math.ceil(total / 2) + (0 if total % 2 else 0) and ups >= 2:
        snap.regime = "多頭（偏好順勢回調買入）"
    elif downs >= 2:
        snap.regime = "空頭（以防守為主，留意恐慌低接機會）"
    else:
        snap.regime = "震盪／方向未明（降低槓桿、縮小部位）"

    if vix is not None and vix >= VIX_PANIC:
        snap.panic_hits.append(f"VIX {vix:.1f} ≥ {VIX_PANIC}")
    if fg is not None and fg <= FEAR_GREED_PANIC:
        snap.panic_hits.append(f"恐懼貪婪 {fg:.0f} ≤ {FEAR_GREED_PANIC}")
    oversold = [s for s in snap.indices if s.rsi < RSI_OVERSOLD]
    if oversold:
        snap.panic_hits.append("RSI<30: " + ", ".join(f"{s.symbol}({s.rsi:.0f})" for s in oversold))
    snap.panic_score = len(snap.panic_hits)
    vix_ok = vix is not None and vix >= VIX_PANIC
    snap.panic_alert = vix_ok and ((fg is not None and fg <= FEAR_GREED_PANIC) or bool(oversold))
    return snap


# ==========================================================================
# 4. BUY SCANNER & SIZING
# ==========================================================================

@dataclass
class Sizing:
    shares: int
    order_value: float
    risk_amount: float
    stop_pct: float
    reward_amount: float
    hard_stop_2pct: float
    notes: list[str] = field(default_factory=list)


@dataclass
class BuySetup:
    symbol: str
    valid: bool
    price: float = float("nan")
    trigger: str = ""
    support: Optional[float] = None
    stop: Optional[float] = None
    target: Optional[float] = None
    target_note: str = ""
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
    if USE_MEASURED_MOVE_TARGET and st.highs:
        top = st.highs[-1]
        base = [l for l in st.lows if l.idx < top.idx]
        if base:
            leg = top.price - base[-1].price
            if leg > 0:
                return top.price + leg, "量度漲幅(估算)"
    return None, "上方無前頂"


def evaluate_buy_setup(symbol: str, df: pd.DataFrame) -> BuySetup:
    if not has_enough_data(df):
        return BuySetup(symbol, False, reason="資料不足")
    close = df["Close"]
    price = float(close.iloc[-1])
    low = float(df["Low"].iloc[-1])
    s20 = float(sma(close, 20).iloc[-1])
    cur_rsi = float(rsi(close).iloc[-1])
    st = analyze_structure(df)
    setup = BuySetup(symbol, False, price=price, rsi=cur_rsi)

    if not st.uptrend:
        setup.reason = "非上升結構"
        return setup
    if st.lows and price < st.lows[-1].price:
        setup.reason = "跌穿前低"
        return setup

    sup = find_support(df, st)
    ma_touch = (low <= s20 * (1 + 0.005)) and (price >= s20 * (1 - MA_TOUCH_TOLERANCE))
    if sup is None and not ma_touch:
        setup.reason = "未回調到支撐或20MA"
        return setup

    if sup is not None:
        support_price, label = sup
        setup.trigger = f"回調至{label} {support_price:.2f}"
        setup.support = support_price
        stop = support_price * (1 - STOP_BUFFER)
    else:
        last_low = st.lows[-1].price if st.lows and st.lows[-1].price < price else s20 * 0.97
        setup.trigger = f"觸及20MA {s20:.2f}"
        setup.support = last_low
        stop = last_low * (1 - STOP_BUFFER)

    if stop >= price:
        setup.reason = "停損高於現價"
        return setup

    target, note = find_target(df, st)
    setup.stop, setup.target, setup.target_note = stop, target, note
    if target is None:
        setup.reason = note
        return setup

    risk, reward = price - stop, target - price
    setup.rrr = reward / risk if risk > 0 else None
    if setup.rrr is None or setup.rrr < MIN_RRR:
        setup.reason = f"RRR不足 ({setup.rrr})" if setup.rrr else "RRR無"
        return setup

    setup.valid = True
    setup.reason = f"RRR {setup.rrr:.2f} ≥ {MIN_RRR}"
    return setup


def gate_leveraged_setup(setup: BuySetup) -> BuySetup:
    if not setup.valid or setup.symbol not in LEVERAGED_MAP:
        return setup
    under, _ = LEVERAGED_MAP[setup.symbol]
    du = get_history(under)
    ok = False
    if has_enough_data(du):
        st = analyze_structure(du)
        ok = st.uptrend and float(du["Close"].iloc[-1]) > float(sma(du["Close"], 20).iloc[-1])
    if not ok:
        setup.reason = f"槓桿對應標的 {under} 未呈上升趨勢"
        setup.valid = False
    return setup


def calc_position(entry: float, stop: float, target: Optional[float],
                  account: float, risk_pct: float, leveraged: bool) -> Sizing:
    per_share_risk = entry - stop
    max_risk = account * risk_pct
    shares = int(max_risk // per_share_risk) if per_share_risk > 0 else 0
    notes = []

    cap_shares = int((account * MAX_POSITION_PCT) // entry)
    if shares > cap_shares:
        shares = cap_shares
        notes.append("受單一標的上限限制")
    shares = min(shares, int(account // entry))
    stop_pct = per_share_risk / entry if entry > 0 else 0
    if leveraged and stop_pct > MAX_STOP_PCT_LEVERAGED:
        notes.append("槓桿停損距離過大")
    reward = shares * (target - entry) if target else 0.0
    return Sizing(shares, shares * entry, shares * per_share_risk, stop_pct, reward,
                  entry * (1 - 0.02), notes)


# ==========================================================================
# 5. PORTFOLIO & ALERTS
# ==========================================================================

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
    results = []
    for h in HOLDINGS:
        sym = h["symbol"]
        df = get_history(sym)
        if df.empty:
            continue
        price = float(df["Close"].iloc[-1])
        shares, cost = float(h["shares"]), float(h["avg_cost"])
        is_lt = bool(h.get("is_long_term", False))
        st = analyze_structure(df) if has_enough_data(df) else Structure([], [])

        raw_stop = h.get("stop", 0)
        if not raw_stop or raw_stop <= 0:
            sup = find_support(df, st)
            stop = sup[0] * (1 - STOP_BUFFER) if sup else float(sma(df["Close"], 20).iloc[-1]) * 0.97
        else:
            stop = float(raw_stop)

        raw_target = h.get("target", 0)
        if not raw_target or raw_target <= 0:
            t_val, _ = find_target(df, st)
            target = t_val if t_val else price * 1.20
        else:
            target = float(raw_target)

        mv, cv = shares * price, shares * cost
        rrr_now = (target - price) / (price - stop) if stop < price < target else None
        hs = HoldingStatus(sym, shares, cost, price, stop, target, mv, cv, mv - cv,
                           (price / cost - 1) * 100, (price / stop - 1) * 100, rrr_now, is_lt)

        # ⚠️ 若勾選長線投資（is_long_term），則略過短線停損與結構破壞警報！
        if not is_lt:
            if price <= stop:
                alerts.append(Alert(f"stop:{sym}", "danger", f"⚠️ 停損警報 {sym}：現價 {price:.2f} 已跌穿停損 {stop:.2f}"))
            elif float(df["Low"].iloc[-1]) <= stop:
                alerts.append(Alert(f"stoptouch:{sym}", "warn", f"⚠️ {sym} 盤中曾觸及停損 {stop:.2f}"))

            if has_enough_data(df) and st.lows and price < st.lows[-1].price and price > stop:
                alerts.append(Alert(f"lowbreak:{sym}", "danger", f"⚠️ 結構破壞 {sym}：跌穿前低"))

        if price >= target:
            alerts.append(Alert(f"target:{sym}", "info", f"🎯 {sym} 已達目標價 {target:.2f}"))

        setup = gate_leveraged_setup(evaluate_buy_setup(sym, df))
        hs.setup = setup
        results.append(hs)
    return results


# ==========================================================================
# 6. RUN ANALYSIS (網頁版專用聚合介面)
# ==========================================================================

@dataclass
class AnalysisResult:
    macro: MacroSnapshot
    setups: list[BuySetup]
    holdings: list[HoldingStatus]
    alerts: list[Alert]


def run_analysis(account_size: float = ACCOUNT_SIZE, risk_pct: float = MAX_RISK_PCT) -> AnalysisResult:
    alerts: list[Alert] = []
    macro = build_macro_snapshot()
    if macro.panic_alert:
        alerts.append(Alert("panic", "opportunity", f"🚨 熊市恐慌低接機會：{' ; '.join(macro.panic_hits)}"))

    setups = []
    for sym in WATCHLIST:
        df = get_history(sym)
        setups.append(gate_leveraged_setup(evaluate_buy_setup(sym, df)) if not df.empty else BuySetup(sym, False, reason="無資料"))

    holdings = evaluate_holdings(alerts)

    for s in setups:
        if s.valid:
            s.sizing = calc_position(s.price, s.stop, s.target, account_size, risk_pct, s.symbol in LEVERAGED_MAP)
            alerts.append(Alert(f"buy:{s.symbol}", "opportunity", f"✅ 買入訊號 {s.symbol}：{s.trigger} | RRR {s.rrr:.2f}"))

    return AnalysisResult(macro=macro, setups=setups, holdings=holdings, alerts=alerts)