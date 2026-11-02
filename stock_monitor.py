#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
==========================================================================
 股票與 ETF 智慧監控與風控助理 (後端引擎)
 支援互惠基金、USD/CAD 匯率換算與持倉風控（買點掃描全面回傳原幣）
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

NEAR_STOP_WARN_PCT = 3.0  # 距離停損 3% 以內發出預警

LEVERAGED_MAP = {
    "TQQQ": ("QQQ", 3), "QLD": ("QQQ", 2),
    "SOXL": ("SMH", 3),
    "UPRO": ("SPY", 3), "SPXL": ("SPY", 3), "SSO": ("SPY", 2),
    "TECL": ("XLK", 3), "FNGU": ("QQQ", 3),
}
LEV_VOL_WARN = 0.30
LEV_CROSS_WARN = 3

_HISTORY_CACHE: dict[tuple[str, str], pd.DataFrame] = {}


def clear_cache() -> None:
    """清除歷史資料快取，強制重新抓取最新報價"""
    global _HISTORY_CACHE
    _HISTORY_CACHE.clear()


def get_usdcad_rate() -> float:
    try:
        df_fx = yf.Ticker("USDCAD=X").history(period="1d")
        if df_fx is not None and not df_fx.empty:
            return float(df_fx["Close"].iloc[-1])
    except Exception:
        pass
    return 1.35


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
        print(f"[提示] {symbol} 非標準股市代號，將採用手動成本價。")
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


@dataclass
class TrendSignals:
    last_low: Optional[float] = None
    broke_prior_low: bool = False
    uptrend_line_broken: bool = False
    downtrend_line_breakout: bool = False


def _line_value(a: Swing, b: Swing, idx: int) -> float:
    if b.idx == a.idx:
        return b.price
    slope = (b.price - a.price) / (b.idx - a.idx)
    return b.price + slope * (idx - b.idx)


def trend_signals(df: pd.DataFrame, st: Structure) -> TrendSignals:
    sig = TrendSignals()
    if df is None or df.empty:
        return sig
    n = len(df)
    close = float(df["Close"].iloc[-1])
    if st.lows:
        sig.last_low = st.lows[-1].price
        sig.broke_prior_low = close < st.lows[-1].price
    if len(st.lows) >= 2 and st.lows[-1].price > st.lows[-2].price:
        sig.uptrend_line_broken = close < _line_value(st.lows[-2], st.lows[-1], n - 1)
    if len(st.highs) >= 2 and st.highs[-1].price < st.highs[-2].price:
        sig.downtrend_line_breakout = close > _line_value(st.highs[-2], st.highs[-1], n - 1)
    return sig


@dataclass
class Alert:
    key: str
    level: str
    text: str
    symbol: str = ""
    action: str = ""

    def __post_init__(self) -> None:
        if not self.symbol and "_" in self.key:
            prefix, rest = self.key.split("_", 1)
            if prefix in {"buy", "stop", "nearstop", "target", "lev", "advice"}:
                self.symbol = rest

    @property
    def is_advice(self) -> bool:
        return self.key.startswith("advice_")


def money(cur: str, value: float) -> str:
    return f"US${value:,.2f}" if cur == "USD" else f"C${value:,.2f}"


ALERT_SEVERITY = {"danger": 0, "warn": 1, "info": 2, "success": 3}


def group_alerts(alerts: list["Alert"]) -> tuple[list["Alert"], dict[str, list["Alert"]]]:
    global_alerts: list[Alert] = []
    by_sym: dict[str, list[Alert]] = {}
    for a in alerts:
        if a.symbol:
            by_sym.setdefault(a.symbol, []).append(a)
        else:
            global_alerts.append(a)
    for lst in by_sym.values():
        lst.sort(key=lambda x: ALERT_SEVERITY.get(x.level, 9))
    ordered = dict(sorted(by_sym.items(),
                          key=lambda kv: (ALERT_SEVERITY.get(kv[1][0].level, 9), kv[0])))
    return global_alerts, ordered


@dataclass
class IndexStatus:
    symbol: str
    close: float
    sma20: float
    above20: bool
    structure: Structure
    rsi: float
    label: str
    signals: TrendSignals = field(default_factory=TrendSignals)


@dataclass
class MacroSnapshot:
    vix: Optional[float]
    vix_prev: Optional[float]
    fear_greed: Optional[float]
    fg_source: str
    indices: list[IndexStatus] = field(default_factory=list)
    regime: str = "多頭（偏好順勢回調買入）"
    panic_score: int = 0
    panic_hits: list[str] = field(default_factory=list)
    panic_alert: bool = False
    is_bear: bool = False
    early_warning: list[str] = field(default_factory=list)


_CNN_FG_URL = "https://production.dataviz.cnn.io/index/fearandgreed/graphdata"
_CNN_HEADERS = {
    "User-Agent": ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                   "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"),
    "Accept": "application/json, text/plain, */*",
    "Accept-Language": "en-US,en;q=0.9",
    "Origin": "https://www.cnn.com",
    "Referer": "https://www.cnn.com/markets/fear-and-greed",
}


def _estimate_fear_greed(vix: Optional[float]) -> Optional[float]:
    parts: list[float] = []
    if vix is not None:
        parts.append(min(100.0, max(0.0, 100 - (vix - 12) / (35 - 12) * 100)))
    spy = get_history("SPY", "1y")
    if has_enough_data(spy, 130):
        c = float(spy["Close"].iloc[-1])
        ma = float(sma(spy["Close"], 125).iloc[-1])
        parts.append(min(100.0, max(0.0, 50 + (c / ma - 1) * 500)))
    return sum(parts) / len(parts) if parts else None


def _fetch_fear_greed(vix: Optional[float] = None) -> tuple[Optional[float], str]:
    if FEAR_GREED_MANUAL is not None:
        return float(FEAR_GREED_MANUAL), "手動設定"

    urls = [_CNN_FG_URL, f"{_CNN_FG_URL}/{dt.date.today().isoformat()}"]
    for url in urls:
        try:
            resp = requests.get(url, headers=_CNN_HEADERS, timeout=8)
            if resp.status_code == 200:
                score = resp.json()["fear_and_greed"]["score"]
                return float(score), "CNN"
        except Exception:
            pass

    est = _estimate_fear_greed(vix)
    if est is not None:
        return est, "估算（VIX+SPY動能）"
    return None, "無法取得"


def _index_status(sym: str) -> Optional[IndexStatus]:
    df = get_history(sym)
    if not has_enough_data(df, 30):
        return None
    close = df["Close"]
    c = float(close.iloc[-1])
    s20 = float(sma(close, 20).iloc[-1])
    st = analyze_structure(df)
    r = float(rsi(close).iloc[-1])
    above = c >= s20
    if st.uptrend and above:
        label = "🟢 強勢多頭"
    elif above:
        label = "🟡 偏多"
    elif st.downtrend:
        label = "🔴 空頭結構"
    else:
        label = "🟠 偏空/整理"
    return IndexStatus(sym, c, s20, above, st, r, label, trend_signals(df, st))


def build_macro_snapshot() -> MacroSnapshot:
    vix = vix_prev = None
    vdf = get_history("^VIX", "3mo")
    if has_enough_data(vdf, 2):
        vix = float(vdf["Close"].iloc[-1])
        vix_prev = float(vdf["Close"].iloc[-2])

    fg, fg_src = _fetch_fear_greed(vix)
    indices = [s for s in (_index_status(b) for b in BENCHMARKS) if s is not None]

    hits: list[str] = []
    if vix is not None and vix >= VIX_PANIC:
        hits.append(f"VIX {vix:.1f} ≥ {VIX_PANIC}")
    if fg is not None and fg <= FEAR_GREED_PANIC:
        hits.append(f"恐懼貪婪指數 {fg:.0f} ≤ {FEAR_GREED_PANIC}")
    oversold = [i.symbol for i in indices if i.rsi <= RSI_OVERSOLD]
    if oversold:
        hits.append(f"RSI 超賣：{', '.join(oversold)}")

    is_bear = False
    if indices:
        n_above = sum(1 for i in indices if i.above20)
        n_down = sum(1 for i in indices if i.structure.downtrend)
        if n_above == len(indices):
            regime = "多頭（偏好順勢回調買入）"
        elif n_above == 0 or n_down >= 2:
            regime = "空頭／高風險（降低倉位、嚴守停損）"
            is_bear = True
        else:
            regime = "震盪（縮小倉位、等待確認）"
    else:
        regime = "資料不足，無法判斷"

    early: list[str] = []
    for i in indices:
        why = []
        if i.structure.downtrend:
            why.append("結構轉為下降")
        if i.signals.broke_prior_low:
            why.append("收盤跌穿前低")
        if why:
            early.append(f"{i.symbol}：{'、'.join(why)}")

    return MacroSnapshot(
        vix=vix, vix_prev=vix_prev, fear_greed=fg, fg_source=fg_src,
        indices=indices, regime=regime,
        panic_score=len(hits), panic_hits=hits, panic_alert=len(hits) >= 2,
        is_bear=is_bear, early_warning=early,
    )


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
    return price * 1.15, "上方預設目標價 (+15%)"


def scan_setups(account_size: float, risk_pct: float, alerts: list[Alert]) -> list[BuySetup]:
    fx = get_usdcad_rate()
    setups = []
    for sym in WATCHLIST:
        df = get_history(sym)
        if not has_enough_data(df):
            setups.append(BuySetup(symbol=sym, valid=False, reason="歷史資料不足 60 根 K 線"))
            continue

        is_cad = sym.endswith(".TO") or sym.endswith(".V")
        fx_mult = fx if not is_cad else 1.0

        price_native = float(df["Close"].iloc[-1])
        price_cad = price_native * fx_mult
        st = analyze_structure(df)
        r = float(rsi(df["Close"]).iloc[-1])

        sup = find_support(df, st)
        tgt, _ = find_target(df, st)

        if sup is None:
            sma20 = float(sma(df["Close"], 20).iloc[-1])
            sup_price_native = sma20 * 0.98
            sup_lbl = "20日均線附近"
        else:
            sup_price_native, sup_lbl = sup

        stop_native = sup_price_native * (1 - STOP_BUFFER)
        tgt_native = tgt if tgt else price_native * 1.15

        # 內部風控與 RRR 計算維持使用 CAD
        sup_price_cad = sup_price_native * fx_mult
        stop_cad = stop_native * fx_mult
        tgt_cad = tgt_native * fx_mult

        risk_dist = price_cad - stop_cad
        reward_dist = tgt_cad - price_cad if tgt_cad > 0 else 0
        rrr = reward_dist / risk_dist if risk_dist > 0 and reward_dist > 0 else 0.0

        is_uptrend = st.uptrend or (price_cad >= float(sma(df["Close"], 20).iloc[-1]) * fx_mult)
        is_near_support = (price_cad - sup_price_cad) / sup_price_cad <= SUPPORT_TOLERANCE * 1.5
        valid = is_uptrend and is_near_support and (rrr >= MIN_RRR)

        risk_amt = account_size * risk_pct
        shares = int(risk_amt / risk_dist) if risk_dist > 0 else 0
        order_val = shares * price_cad

        max_pos_val = account_size * MAX_POSITION_PCT
        if order_val > max_pos_val:
            shares = int(max_pos_val / price_cad) if price_cad > 0 else 0
            order_val = shares * price_cad

        sizing = Sizing(
            shares=shares, order_value=order_val, risk_amount=shares * risk_dist if risk_dist > 0 else 0,
            stop_pct=(risk_dist / price_cad) * 100 if price_cad > 0 else 0, reward_amount=shares * reward_dist if reward_dist > 0 else 0,
            hard_stop_2pct=price_cad * 0.98
        )
        
        cur_s = "CAD" if is_cad else "USD"
        reason_text = f"回調至{sup_lbl} {money(cur_s, sup_price_native)}，RRR 達 {rrr:.2f}" if valid else f"目前價格未緊貼支撐或 RRR ({rrr:.2f}) 未達門檻"

        # 這裡直接回傳「原幣」數值供前端顯示
        setups.append(BuySetup(
            symbol=sym, valid=valid, price=price_native, trigger=f"回調至{sup_lbl} {money(cur_s, sup_price_native)}",
            support=sup_price_native, stop=stop_native, target=tgt_native, rrr=rrr, reason=reason_text, rsi=r, sizing=sizing
        ))

        if valid:
            alerts.append(Alert(
                f"buy_{sym}", "success",
                f"🟢 {sym} 符合買點：回調至{sup_lbl} {money(cur_s, sup_price_native)}，"
                f"停損 {money(cur_s, stop_native)}，目標 {money(cur_s, tgt_native)}，RRR 達 {rrr:.2f}"))
    return setups


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
    risk_level: str = "高風險 (High)"
    account_type: str = "TFSA"
    is_long_term: bool = False
    currency: str = "CAD"
    fx_mult: float = 1.0
    avg_cost_native: float = 0.0
    price_native: float = 0.0
    stop_native: float = 0.0
    target_native: float = 0.0
    has_live: bool = True
    rsi: Optional[float] = None
    trend: str = ""
    above20: Optional[bool] = None
    weight_pct: float = 0.0


def evaluate_holdings(alerts: list[Alert], account_size: Optional[float] = None) -> list[HoldingStatus]:
    account_size = float(account_size) if account_size else float(ACCOUNT_SIZE)
    fx = get_usdcad_rate()
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
        risk_lvl = str(h.get("risk_level", "高風險 (High)"))
        acc_type = str(h.get("account_type", "TFSA"))

        if sym not in aggregated:
            aggregated[sym] = {
                "shares": 0.0, "total_cost": 0.0, "is_long_term": is_lt,
                "risk_level": risk_lvl, "account_type": acc_type,
                "stop": float(h.get("stop", 0) or 0), "target": float(h.get("target", 0) or 0),
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
        risk_lvl = data["risk_level"]
        acc_type = data["account_type"]

        df = get_history(sym)
        has_live = not df.empty
        is_cad = sym.endswith(".TO") or sym.endswith(".V")
        if df.empty:
            price_native = avg_cost
            fx_mult = 1.0
        else:
            fx_mult = fx if not is_cad else 1.0
            price_native = float(df["Close"].iloc[-1])

        price_cad = price_native * fx_mult
        st = analyze_structure(df) if has_enough_data(df) else Structure([], [])
        
        stop_native = data["stop"] if data["stop"] > 0 else 0
        if not stop_native or stop_native <= 0:
            sup = find_support(df, st) if not df.empty else None
            stop_native = sup[0] * (1 - STOP_BUFFER) if sup else avg_cost * 0.90
        stop_cad = stop_native * fx_mult

        target_native = data["target"] if data["target"] > 0 else 0
        if not target_native or target_native <= 0:
            t_val, _ = find_target(df, st) if not df.empty else (avg_cost * 1.20, "")
            target_native = t_val
        target_cad = target_native * fx_mult

        currency = "CAD" if (is_cad or not has_live) else "USD"
        avg_cost_cad = avg_cost * fx_mult
        mv, cv = shares * price_cad, shares * avg_cost_cad
        rrr_now = (target_cad - price_cad) / (price_cad - stop_cad) if stop_cad < price_cad < target_cad else None
        dist_stop = (price_cad / stop_cad - 1) * 100 if stop_cad > 0 else 0.0

        if has_enough_data(df):
            rsi_val: Optional[float] = float(rsi(df["Close"]).iloc[-1])
            above20: Optional[bool] = bool(price_cad >= float(sma(df["Close"], 20).iloc[-1]) * fx_mult)
            trend = "上升結構" if st.uptrend else ("下降結構" if st.downtrend else "盤整")
        else:
            rsi_val, above20, trend = None, None, ""

        results.append(HoldingStatus(
            symbol=sym, shares=shares, avg_cost=avg_cost_cad, price=price_cad,
            stop=stop_cad, target=target_cad, market_value=mv, cost_value=cv,
            pnl=mv - cv, pnl_pct=(price_cad / avg_cost_cad - 1) * 100 if avg_cost_cad > 0 else 0.0,
            dist_to_stop_pct=dist_stop,
            rrr_now=rrr_now, risk_level=risk_lvl, account_type=acc_type, is_long_term=is_lt,
            currency=currency, fx_mult=fx_mult, avg_cost_native=avg_cost,
            price_native=price_native, stop_native=stop_native, target_native=target_native,
            has_live=has_live, rsi=rsi_val, trend=trend, above20=above20,
            weight_pct=(mv / account_size * 100) if account_size > 0 else 0.0,
        ))

        if has_live and not is_lt:
            if price_cad <= stop_cad:
                alerts.append(Alert(f"stop_{sym}", "danger",
                                    f"🔴 {sym} 現價 {money(currency, price_native)} 已跌破停損價 {money(currency, stop_native)}，請檢視是否出場"))
            elif dist_stop <= NEAR_STOP_WARN_PCT:
                alerts.append(Alert(f"nearstop_{sym}", "warn",
                                    f"🟠 {sym} 距離停損僅 {dist_stop:.1f}%（現價 {money(currency, price_native)}／停損 {money(currency, stop_native)}）"))
            if price_cad >= target_cad:
                alerts.append(Alert(f"target_{sym}", "info",
                                    f"🎯 {sym} 現價 {money(currency, price_native)} 已達目標價 {money(currency, target_native)}，可考慮分批停利"))
    return results


def advise_holding(h: HoldingStatus, setup: Optional[BuySetup], account_size: float) -> Optional[tuple[str, str, str]]:
    if not h.has_live:
        return None
    cur = h.currency
    p, s, t = h.price_native, h.stop_native, h.target_native
    max_pct = MAX_POSITION_PCT * 100
    cap_val = account_size * MAX_POSITION_PCT
    w = h.weight_pct

    if h.price <= h.stop:
        if h.is_long_term:
            return ("檢視持有理由", "warn",
                    f"長線資產已跌破參考停損價 {money(cur, s)}（現價 {money(cur, p)}）；請重新評估持有理由")
        return ("減碼／出場", "danger",
                f"現價 {money(cur, p)} 已跌破停損價 {money(cur, s)}，依停損紀律應減碼或出場")

    if h.price >= h.target:
        return ("分批停利", "warn",
                    f"現價 {money(cur, p)} 已達目標價 {money(cur, t)}，可考慮分批賣出並上移停損")

    if w > max_pct and h.price > 0:
        trim = math.ceil((h.market_value - cap_val) / h.price / h.fx_mult)
        return ("減碼", "warn",
                f"倉位占總資金 {w:.1f}%，超過單一標的上限 {max_pct:.0f}%；需減碼約 {trim} 股")

    if (not h.is_long_term) and h.dist_to_stop_pct <= NEAR_STOP_WARN_PCT and h.trend == "下降結構":
        return ("減碼／收緊停損", "warn",
                f"距停損僅 {h.dist_to_stop_pct:.1f}% 且結構轉為下降，可先減碼或嚴守停損 {money(cur, s)}")

    if setup is not None and setup.valid:
        room_val = cap_val - h.market_value
        room_sh = int(room_val / h.price / h.fx_mult) if (room_val > 0 and h.price > 0) else 0
        rule_sh = setup.sizing.shares if setup.sizing else 0
        add = min(rule_sh, room_sh)
        if add >= 1:
            return ("可加碼", "success",
                    f"出現買點（回調至支撐 {money(cur, setup.support)}，RRR {setup.rrr:.2f}），最多可加約 {add} 股")
        return ("持有", "info", f"雖有買點，但倉位 {w:.1f}% 已接近上限")

    if h.rsi is not None and h.rsi >= 70 and h.pnl_pct >= 20:
        return ("持有（可部分停利）", "info", f"已獲利 {h.pnl_pct:.1f}% 且 RSI {h.rsi:.0f} 偏高（過熱）")

    extra = f"；距停損 {h.dist_to_stop_pct:.1f}%（停損 {money(cur, s)}）"
    rsi_txt = f"，RSI {h.rsi:.0f}" if h.rsi is not None else ""
    return ("持有", "info", f"趨勢：{h.trend or '資料不足'}{rsi_txt}{extra}")


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
        alerts.append(Alert("panic", "warn",
                            "😱 市場恐慌訊號：" + "；".join(macro.panic_hits) + "（分批布局，勿一次重倉）"))
    holdings = evaluate_holdings(alerts, account_size)
    setups = scan_setups(account_size, risk_pct, alerts)

    setup_map = {s.symbol: s for s in setups}
    for h in holdings:
        adv = advise_holding(h, setup_map.get(h.symbol), account_size)
        if adv is None:
            continue
        action, level, reason = adv
        alerts.append(Alert(f"advice_{h.symbol}", level, f"🧭 {h.symbol} 規則建議【{action}】{reason}", action=action))
    return AnalysisResult(macro=macro, setups=setups, holdings=holdings, alerts=alerts)
