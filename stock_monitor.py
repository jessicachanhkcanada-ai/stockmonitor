#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
==========================================================================
 股票與 ETF 智慧監控與風控助理  (Smart Stock & ETF Risk Monitor)
==========================================================================

【安裝】
    pip install yfinance pandas numpy requests

【執行】
    python stock_monitor.py                 # 執行一次，輸出報表
    python stock_monitor.py --interactive   # 有買入訊號時，互動輸入總資金與單筆風險%
    python stock_monitor.py --notify        # 同時推送 Telegram / Discord（見 CONFIG）
    python stock_monitor.py --loop 30       # 每 30 分鐘自動執行一次（Ctrl+C 結束）
    python stock_monitor.py --notify --loop 60 --interactive   # 可組合使用

【模組總覽】
    1. Config            設定區：自選股、持倉、風控參數、通知 Token
    2. Data Layer        yfinance 資料抓取（含快取）
    3. Indicators        SMA / RSI / 歷史波動率 / 波段高低點（Swing）
    4. Market Regime     大盤體制過濾 + VIX + 恐懼貪婪 + 熊市恐慌低接偵測
    5. Buy Scanner       牛市回調買點 + 風險報酬比 (RRR)
    6. Portfolio         持倉損益、停損/結構破壞警報、加碼提醒
    7. Leveraged Decay   槓桿 ETF 複利損耗與波動監控
    8. Position Sizing   資金部位與風控計算機
    9. Notifier          Telegram / Discord 通知（同日同警報去重）
   10. Report / Main     終端機報表與主流程

【重要聲明】
    本程式只依技術指標產生「提示」，不構成投資建議。
    波段高低點需要右側 N 根 K 線確認（預設 N=5），因此結構判斷有數日延遲，
    屬於技術分析的固有限制。槓桿 ETF 風險極高，請務必自行驗證並嚴守停損。
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
# 1. CONFIG 設定區  —— 日常只需要修改這一區
# ==========================================================================

# ---- 1.1 觀察清單（Watchlist）：想掃描買點的標的 -------------------------
WATCHLIST = ["NVDA", "AAPL", "MSFT", "AMD", "SMH", "QQQ", "SPY", "TQQQ", "SOXL"]

# ---- 1.2 目前持倉（Holdings）：請改成你自己的資料 -------------------------
#   symbol    : 代號
#   shares    : 持股數
#   avg_cost  : 平均成本
#   stop      : 停損價 / 關鍵支撐位（以「收市價」跌穿為準）
#   target    : 目標價（例如前頂）
# 以下僅為示範數字，請替換！
HOLDINGS = [
    {"symbol": "SMH",  "shares": 20,  "avg_cost": 290.0, "stop": 281.0, "target": 360.0},
    {"symbol": "NVDA", "shares": 30,  "avg_cost": 120.0, "stop": 110.0, "target": 160.0},
    {"symbol": "SOXL", "shares": 100, "avg_cost": 35.0,  "stop": 30.0,  "target": 50.0},
]

# ---- 1.3 大盤基準 ------------------------------------------------------
BENCHMARKS = ["SPY", "QQQ", "SMH"]

# ---- 1.4 資金與風控參數 -------------------------------------------------
ACCOUNT_SIZE = 100_000        # 總資金規模（可在 --interactive 模式覆蓋）
MAX_RISK_PCT = 0.01           # 單筆最大容忍虧損占總資金比例（1% = 0.01）
MAX_POSITION_PCT = 0.25       # 單一標的最大倉位占總資金比例（防止停損過近時股數爆大）
MAX_STOP_PCT_LEVERAGED = 0.02 # 槓桿 ETF：停損距離超過此比例（2%）時發出提示
MIN_RRR = 2.0                 # 最低風險報酬比，低於此值不發買入通知

# ---- 1.5 技術分析參數 ---------------------------------------------------
SWING_WINDOW = 5              # 波段高低點確認窗口：左右各 N 根 K 線
LOOKBACK_BARS = 120           # 只使用最近 N 根 K 線內的波段點
SUPPORT_TOLERANCE = 0.03      # 價格距支撐 3% 以內視為「回調到支撐附近」
MA_TOUCH_TOLERANCE = 0.015    # 觸及 20 日線的容忍度（1.5%）
STOP_BUFFER = 0.005           # 停損設在支撐下方 0.5% 緩衝
USE_MEASURED_MOVE_TARGET = True  # 已創新高、上方無前頂時，用「量度漲幅」估算目標價

# ---- 1.6 熊市／恐慌低接條件 --------------------------------------------
VIX_PANIC = 21.5              # VIX >= 此值
FEAR_GREED_PANIC = 25         # 恐懼貪婪指數 <= 此值
RSI_OVERSOLD = 30             # RSI < 此值
FEAR_GREED_MANUAL: Optional[float] = None  # 若想手動指定恐懼貪婪值，填數字（0~100）

# ---- 1.7 槓桿 ETF 對照表：代號 -> (對應一倍標的, 槓桿倍數) -----------------
LEVERAGED_MAP = {
    "TQQQ": ("QQQ", 3), "QLD": ("QQQ", 2),
    "SOXL": ("SMH", 3),
    "UPRO": ("SPY", 3), "SPXL": ("SPY", 3), "SSO": ("SPY", 2),
    "TECL": ("XLK", 3), "FNGU": ("QQQ", 3),
}
LEV_VOL_WARN = 0.30           # 標的 20 日年化波動率 >= 30% 視為高波動
LEV_CROSS_WARN = 3            # 近 20 日收市價穿越 20 日線次數 >= 3 視為震盪市

# ---- 1.8 通知設定（留空則不推送；建議用環境變數，避免把 Token 寫進程式碼）----
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "")   # 例: "123456:ABC-DEF..."
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "")       # 例: "123456789"
DISCORD_WEBHOOK_URL = os.getenv("DISCORD_WEBHOOK_URL", "") # 例: "https://discord.com/api/webhooks/..."
ALERT_STATE_FILE = ".alert_state.json"                     # 通知去重用的狀態檔

CNN_FEAR_GREED_URL = "https://production.dataviz.cnn.io/index/fearandgreed/graphdata"


# ==========================================================================
# 2. DATA LAYER 資料層
# ==========================================================================

_HISTORY_CACHE: dict[tuple[str, str], pd.DataFrame] = {}


def get_history(symbol: str, period: str = "1y") -> pd.DataFrame:
    """抓取日線 OHLCV（含快取）。失敗時回傳空 DataFrame，不讓整個程式中斷。"""
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
    except Exception as exc:  # 網絡或代號錯誤
        print(f"[警告] 無法取得 {symbol} 資料: {exc}")
    _HISTORY_CACHE[key] = df
    return df


def has_enough_data(df: pd.DataFrame, min_bars: int = 60) -> bool:
    return df is not None and len(df) >= min_bars


# ==========================================================================
# 3. INDICATORS 技術指標
# ==========================================================================

def sma(series: pd.Series, n: int) -> pd.Series:
    """簡單移動平均線。"""
    return series.rolling(n).mean()


def rsi(series: pd.Series, n: int = 14) -> pd.Series:
    """Wilder RSI。"""
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1 / n, adjust=False, min_periods=n).mean()
    avg_loss = loss.ewm(alpha=1 / n, adjust=False, min_periods=n).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    out = 100 - 100 / (1 + rs)
    return out.fillna(100.0).where(avg_gain.notna())


def hist_vol(close: pd.Series, n: int = 20) -> float:
    """最近 n 日的年化歷史波動率（對數報酬標準差 × sqrt(252)）。"""
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
    """
    找出波段高點與低點：某根 K 線的高（低）點為左右各 window 根內的最高（最低）。
    只回傳最近 LOOKBACK_BARS 根內、且已被右側 window 根確認的波段點。
    """
    hi = df["High"].to_numpy()
    lo = df["Low"].to_numpy()
    dates = df.index
    n = len(df)
    start = max(window, n - LOOKBACK_BARS)
    highs: list[Swing] = []
    lows: list[Swing] = []
    for i in range(start, n - window):
        if hi[i - window:i + window + 1].argmax() == window:
            highs.append(Swing(i, dates[i], float(hi[i])))
        if lo[i - window:i + window + 1].argmin() == window:
            lows.append(Swing(i, dates[i], float(lo[i])))
    return highs, lows


@dataclass
class Structure:
    """價格結構：一底高於一底、一頂高於一頂 = 上升趨勢。"""
    highs: list[Swing]
    lows: list[Swing]
    hh: bool = False   # Higher High 一頂高於一頂
    hl: bool = False   # Higher Low  一底高於一底
    lh: bool = False   # Lower High  一頂低於一頂
    ll: bool = False   # Lower Low   一底低於一底

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
    """以最近兩個波段低點連成上升趨勢線，回傳「最新一根 K 線」位置的趨勢線價格。"""
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
# 4. MARKET REGIME 大盤體制過濾
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
    """
    CNN 指數取得失敗時的「估算替代」（非官方）：
    50% 來自 VIX（12→貪婪 100，30→恐懼 0），50% 來自 SPY 相對 125 日均線的動能（-10%→0，+10%→100）。
    """
    spy = get_history("SPY", period="1y")
    if vix is None or not has_enough_data(spy, 130):
        return None
    vix_part = float(np.clip((30 - vix) / (30 - 12) * 100, 0, 100))
    ratio = float(spy["Close"].iloc[-1] / spy["Close"].rolling(125).mean().iloc[-1] - 1)
    mom_part = float(np.clip((ratio + 0.10) / 0.20 * 100, 0, 100))
    return round(0.5 * vix_part + 0.5 * mom_part, 1)


def fetch_fear_greed(vix: Optional[float]) -> tuple[Optional[float], str]:
    """優先順序：手動指定 > CNN 官方（非公開 API，可能失效）> 以 VIX/動能估算。"""
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

    # ---- 大盤體制：多數基準為上升趨勢 = 多頭；多數為下降 = 空頭；其餘 = 震盪 ----
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

    # ---- 恐慌低接評分：VIX / 恐懼貪婪 / 基準 RSI，越多條件同時出現越接近低點 ----
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
# 5. BUY SCANNER 買入訊號掃描
# ==========================================================================

@dataclass
class BuySetup:
    symbol: str
    valid: bool                      # True = 滿足全部條件且 RRR >= MIN_RRR
    price: float = float("nan")
    trigger: str = ""
    support: Optional[float] = None
    stop: Optional[float] = None
    target: Optional[float] = None
    target_note: str = ""
    rrr: Optional[float] = None
    reason: str = ""                 # 未成立時的原因 / 成立時的補充
    rsi: float = float("nan")
    sizing: Optional["Sizing"] = None


def find_support(df: pd.DataFrame, st: Structure) -> Optional[tuple[float, str]]:
    """
    找出距離現價最近、且在容忍範圍內的支撐：
      - 前低（波段低點）
      - 前高轉支撐（已被收市價突破的波段高點）
    """
    price = float(df["Close"].iloc[-1])
    cands = [(s.price, "前低支撐") for s in st.lows if s.price <= price]
    cands += [(s.price, "前高轉支撐") for s in st.highs if s.price <= price]
    near = [(p, lbl) for p, lbl in cands if (price - p) / p <= SUPPORT_TOLERANCE]
    return max(near, key=lambda x: x[0]) if near else None


def find_target(df: pd.DataFrame, st: Structure) -> tuple[Optional[float], str]:
    """目標價：現價上方最近一個波段高點（前頂）；若已創新高，可用量度漲幅估算。"""
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
                return top.price + leg, "量度漲幅(估算，上方無前頂)"
    return None, "上方無前頂可參考"


def evaluate_buy_setup(symbol: str, df: pd.DataFrame) -> BuySetup:
    """
    牛市回調買點判斷：
      前提：一底高於一底、一頂高於一頂（上升結構）。
      觸發：(A) 價格回調到前低 / 前高轉支撐附近；(B) 觸及 20 日均線。
      風報：停損設在支撐下方緩衝處；目標為前頂；RRR >= MIN_RRR 才成立。
    """
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
        setup.reason = "非上升結構（未見 一底高於一底＋一頂高於一頂）"
        return setup
    if st.lows and price < st.lows[-1].price:
        setup.reason = "收市價已跌穿前低，結構受損"
        return setup

    # ---- 觸發條件 ----
    sup = find_support(df, st)
    ma_touch = (low <= s20 * (1 + 0.005)) and (price >= s20 * (1 - MA_TOUCH_TOLERANCE))
    if sup is None and not ma_touch:
        setup.reason = "價格未回調到支撐或 20 日線附近（等待回調）"
        return setup

    if sup is not None:
        support_price, label = sup
        setup.trigger = f"回調至{label} {support_price:.2f}" + ("，並觸及 20 日線" if ma_touch else "")
        setup.support = support_price
        stop = support_price * (1 - STOP_BUFFER)
    else:
        last_low = st.lows[-1].price if st.lows and st.lows[-1].price < price else s20 * 0.97
        setup.trigger = f"觸及 20 日線 {s20:.2f}（次選買點，RRR 通常較低）"
        setup.support = last_low
        stop = last_low * (1 - STOP_BUFFER)

    if stop >= price:
        setup.reason = "停損價高於或等於現價，無法計算風險"
        return setup

    target, note = find_target(df, st)
    setup.stop, setup.target, setup.target_note = stop, target, note
    if target is None:
        setup.reason = note + "，請手動設定目標價"
        return setup

    risk, reward = price - stop, target - price
    setup.rrr = reward / risk if risk > 0 else None
    if setup.rrr is None or setup.rrr < MIN_RRR:
        setup.reason = f"RRR {setup.rrr:.2f} < {MIN_RRR}，風險報酬不足" if setup.rrr is not None else "RRR 無法計算"
        return setup

    setup.valid = True
    setup.reason = f"RRR {setup.rrr:.2f} ≥ {MIN_RRR}"
    return setup


def gate_leveraged_setup(setup: BuySetup) -> BuySetup:
    """
    槓桿 ETF 買入閘門：只有在「對應一倍標的」呈清晰上升趨勢（一底高於一底、
    一頂高於一頂，且收市價在 20 日線之上）時，才保留買入／加碼訊號。
    這是為了避免在震盪或下跌市中買入槓桿 ETF 而承受複利損耗。
    """
    if not setup.valid or setup.symbol not in LEVERAGED_MAP:
        return setup
    under, _ = LEVERAGED_MAP[setup.symbol]
    du = get_history(under)
    ok = False
    if has_enough_data(du):
        st = analyze_structure(du)
        ok = st.uptrend and float(du["Close"].iloc[-1]) > float(sma(du["Close"], 20).iloc[-1])
    if not ok:
        setup.reason = (f"槓桿 ETF：對應一倍標的 {under} 未呈清晰上升趨勢，暫不買入"
                        f"（原訊號 RRR {setup.rrr:.2f}）")
        setup.valid = False
    return setup


# ==========================================================================
# 8. POSITION SIZING 資金部位與風控計算機
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


def calc_position(entry: float, stop: float, target: Optional[float],
                  account: float, risk_pct: float, leveraged: bool) -> Sizing:
    """
    股數 = (總資金 × 單筆風險%) / (買入價 - 停損價)
    再受「單一標的最大倉位」與「資金上限」限制。實際風險金額永遠 ≤ 容忍虧損。
    """
    per_share_risk = entry - stop
    max_risk = account * risk_pct
    shares = int(max_risk // per_share_risk) if per_share_risk > 0 else 0
    notes: list[str] = []

    cap_shares = int((account * MAX_POSITION_PCT) // entry)
    if shares > cap_shares:
        shares = cap_shares
        notes.append(f"已受單一標的上限 {MAX_POSITION_PCT:.0%} 限制")
    cash_shares = int(account // entry)
    shares = min(shares, cash_shares)

    stop_pct = per_share_risk / entry
    if leveraged and stop_pct > MAX_STOP_PCT_LEVERAGED:
        notes.append(f"槓桿 ETF 停損距離 {stop_pct:.1%} 超過 {MAX_STOP_PCT_LEVERAGED:.0%}，"
                     f"建議等更貼近支撐的買點或再縮小部位")
    reward = shares * (target - entry) if target else 0.0
    return Sizing(shares, shares * entry, shares * per_share_risk, stop_pct, reward,
                  entry * (1 - 0.02), notes)


# ==========================================================================
# 6. PORTFOLIO 持倉管理與風控
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
    setup: Optional[BuySetup] = None


def evaluate_holdings(alerts: list["Alert"]) -> list[HoldingStatus]:
    results: list[HoldingStatus] = []
    for h in HOLDINGS:
        sym = h["symbol"]
        df = get_history(sym)
        if df.empty:
            print(f"[警告] 持倉 {sym} 無價格資料，略過")
            continue
        price = float(df["Close"].iloc[-1])
        shares, cost = float(h["shares"]), float(h["avg_cost"])
        stop, target = float(h["stop"]), float(h["target"])
        mv, cv = shares * price, shares * cost
        # 現價已高於目標價或已跌穿停損時，RRR 沒有意義，顯示 "-"
        rrr_now = (target - price) / (price - stop) if stop < price < target else None
        hs = HoldingStatus(sym, shares, cost, price, stop, target, mv, cv, mv - cv,
                           (price / cost - 1) * 100, (price / stop - 1) * 100, rrr_now)

        # ---- 賣出／風控警報 ----
        if price <= stop:
            alerts.append(Alert(f"stop:{sym}", "danger",
                                f"⚠️ 停損/賣出警報 {sym}：收市價 {price:.2f} 已跌穿停損/支撐 {stop:.2f}"))
        elif float(df["Low"].iloc[-1]) <= stop:
            alerts.append(Alert(f"stoptouch:{sym}", "warn",
                                f"⚠️ {sym} 盤中曾觸及停損 {stop:.2f}，但收市 {price:.2f} 仍守住，請密切留意"))

        if has_enough_data(df):
            st = analyze_structure(df)
            if st.lows and price < st.lows[-1].price and price > stop:
                alerts.append(Alert(f"lowbreak:{sym}", "danger",
                                    f"⚠️ 結構破壞 {sym}：收市價 {price:.2f} 跌穿前低 {st.lows[-1].price:.2f}"))
            elif st.downtrend:
                alerts.append(Alert(f"ll_lh:{sym}", "danger",
                                    f"⚠️ 結構破壞 {sym}：出現 一底低於一底＋一頂低於一頂"))
            tl = uptrend_line_value(st, len(df))
            if tl is not None and price < tl:
                alerts.append(Alert(f"trendline:{sym}", "warn",
                                    f"⚠️ {sym} 收市價 {price:.2f} 跌穿上升趨勢線（約 {tl:.2f}）"))

        if price >= target:
            alerts.append(Alert(f"target:{sym}", "info",
                                f"🎯 {sym} 已達目標價 {target:.2f}（現價 {price:.2f}），可考慮分批止盈並上移停損"))

        # ---- 加碼提醒：再次出現合格回調買點 ----
        setup = gate_leveraged_setup(evaluate_buy_setup(sym, df))
        hs.setup = setup
        if setup.valid and price > stop:
            alerts.append(Alert(f"add:{sym}", "info",
                                f"💡 {sym} 可考慮分批加碼：{setup.trigger}；"
                                f"停損 {setup.stop:.2f}、目標 {setup.target:.2f}、RRR {setup.rrr:.2f}"))
        results.append(hs)
    return results


# ==========================================================================
# 7. LEVERAGED DECAY 槓桿 ETF 複利損耗與波動監控
# ==========================================================================

@dataclass
class LevStatus:
    symbol: str
    underlying: str
    leverage: int
    vol_underlying: float
    vol_lev: float
    crosses: int
    clear_uptrend: bool
    downtrend: bool
    drag_annual: float
    shortfall_60d: Optional[float]
    warn: bool


def check_leveraged(symbol: str) -> Optional[LevStatus]:
    """
    以一倍標的判斷市況：
      - 20 日年化波動率
      - 近 20 日穿越 20 日線次數（震盪程度）
      - 是否為清晰上升趨勢
      - 理論年化損耗 ≈ 0.5 × L × (L-1) × σ²
      - 近 60 日「實際槓桿 ETF 報酬 − 倍數×標的報酬」差距（粗略觀察損耗）
    """
    under, lev = LEVERAGED_MAP[symbol]
    du, dl = get_history(under), get_history(symbol)
    if not has_enough_data(du) or dl.empty:
        return None
    cu = du["Close"]
    vol_u = hist_vol(cu)
    vol_l = hist_vol(dl["Close"])
    s20 = sma(cu, 20)
    diff = (cu - s20).dropna().iloc[-20:]
    crosses = int((np.sign(diff).diff().abs() > 0).sum())
    st = analyze_structure(du)
    clear_up = st.uptrend and float(cu.iloc[-1]) > float(s20.iloc[-1])
    drag = 0.5 * lev * (lev - 1) * (vol_u ** 2) if not math.isnan(vol_u) else float("nan")

    shortfall = None
    if len(cu) > 61 and len(dl) > 61:
        ret_u = float(cu.iloc[-1] / cu.iloc[-61] - 1)
        ret_l = float(dl["Close"].iloc[-1] / dl["Close"].iloc[-61] - 1)
        shortfall = ret_l - lev * ret_u

    choppy = (not clear_up) and (crosses >= LEV_CROSS_WARN or (not math.isnan(vol_u) and vol_u >= LEV_VOL_WARN))
    warn = choppy or st.downtrend
    return LevStatus(symbol, under, lev, vol_u, vol_l, crosses, clear_up, st.downtrend, drag, shortfall, warn)


# ==========================================================================
# 9. NOTIFIER 通知系統
# ==========================================================================

@dataclass
class Alert:
    key: str      # 去重用的唯一鍵
    level: str    # danger / warn / info / opportunity
    text: str


def _chunks(text: str, size: int) -> list[str]:
    return [text[i:i + size] for i in range(0, len(text), size)] or [""]


def send_telegram(text: str) -> None:
    if not (TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID):
        return
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    for part in _chunks(text, 3800):
        try:
            requests.post(url, json={"chat_id": TELEGRAM_CHAT_ID, "text": part}, timeout=15).raise_for_status()
        except Exception as exc:
            print(f"[警告] Telegram 發送失敗: {exc}")


def send_discord(text: str) -> None:
    if not DISCORD_WEBHOOK_URL:
        return
    for part in _chunks(text, 1900):
        try:
            requests.post(DISCORD_WEBHOOK_URL, json={"content": part}, timeout=15).raise_for_status()
        except Exception as exc:
            print(f"[警告] Discord 發送失敗: {exc}")


def filter_new_alerts(alerts: list[Alert]) -> list[Alert]:
    """同一天內相同 key 的警報只推送一次，避免洗版。"""
    today = dt.date.today().isoformat()
    state = {"date": today, "keys": []}
    try:
        with open(ALERT_STATE_FILE, "r", encoding="utf-8") as f:
            loaded = json.load(f)
        if loaded.get("date") == today:
            state = loaded
    except Exception:
        pass
    fresh = [a for a in alerts if a.key not in state["keys"]]
    state["keys"].extend(a.key for a in fresh)
    try:
        with open(ALERT_STATE_FILE, "w", encoding="utf-8") as f:
            json.dump(state, f, ensure_ascii=False)
    except Exception as exc:
        print(f"[警告] 無法寫入狀態檔: {exc}")
    return fresh


def push_alerts(alerts: list[Alert]) -> None:
    fresh = filter_new_alerts(alerts)
    if not fresh:
        print("（沒有新的警報需要推送）")
        return
    if not (TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID) and not DISCORD_WEBHOOK_URL:
        print("（未設定 Telegram / Discord，略過推送）")
        return
    header = f"📡 智慧監控 {dt.datetime.now():%Y-%m-%d %H:%M}\n"
    msg = header + "\n".join(a.text for a in fresh)
    send_telegram(msg)
    send_discord(msg)
    print(f"已推送 {len(fresh)} 則新警報。")


# ==========================================================================
# 10. REPORT 報表輸出
# ==========================================================================

def money(x: float) -> str:
    return f"{x:,.2f}"


def section(title: str) -> None:
    print("\n" + "=" * 72)
    print(title)
    print("=" * 72)


def print_macro(snap: MacroSnapshot) -> None:
    section("📊 【宏觀與大盤體制狀態】")
    for s in snap.indices:
        st = s.structure
        hh = "HH" if st.hh else ("LH" if st.lh else "-")
        hl = "HL" if st.hl else ("LL" if st.ll else "-")
        print(f"  {s.symbol:<5} 收 {s.close:>9.2f} | 20MA {s.sma20:>9.2f} "
              f"({'上' if s.above20 else '下'}) | RSI {s.rsi:>5.1f} | 結構 {hh}/{hl} | {s.label}")
    vix_txt = f"{snap.vix:.2f}" if snap.vix is not None else "N/A"
    if snap.vix is not None and snap.vix_prev:
        vix_txt += f"（前日 {snap.vix_prev:.2f}）"
    fg_txt = f"{snap.fear_greed:.0f}" if snap.fear_greed is not None else "N/A"
    print(f"\n  VIX: {vix_txt}")
    print(f"  恐懼貪婪指數: {fg_txt}  [來源: {snap.fg_source}]")
    print(f"  大盤體制: {snap.regime}")
    print(f"  恐慌低接評分: {snap.panic_score}/3  "
          f"{' ; '.join(snap.panic_hits) if snap.panic_hits else '（目前無恐慌條件）'}")
    if snap.panic_alert:
        print("  🚨 熊市極度恐慌低接機會：條件越多越接近低點，適合分批、小注逐步加注。")


def print_sizing(sz: Sizing, indent: str = "      ") -> None:
    print(f"{indent}建議股數 {sz.shares} 股 | 下單金額 {money(sz.order_value)} | "
          f"最大虧損 {money(sz.risk_amount)} | 停損距離 {sz.stop_pct:.2%} | "
          f"到目標獲利約 {money(sz.reward_amount)}")
    print(f"{indent}參考：入場價下方 2% 硬停損價 = {sz.hard_stop_2pct:.2f}")
    for n in sz.notes:
        print(f"{indent}⚠️ {n}")


def print_scan(setups: list[BuySetup]) -> None:
    section("🔍 【觀察清單買點掃描】")
    hits = [s for s in setups if s.valid]
    print(f"  共 {len(setups)} 檔，符合買入條件 {len(hits)} 檔（RRR ≥ {MIN_RRR}）\n")
    for s in setups:
        if s.valid:
            print(f"  ✅ {s.symbol:<6} 現價 {s.price:.2f} | {s.trigger}")
            print(f"      停損 {s.stop:.2f} | 目標 {s.target:.2f}（{s.target_note}） | RRR {s.rrr:.2f} | RSI {s.rsi:.0f}")
            if s.sizing:
                print_sizing(s.sizing)
        else:
            px = f"{s.price:.2f}" if not math.isnan(s.price) else "N/A"
            print(f"  ·  {s.symbol:<6} 現價 {px:>9} | 未觸發：{s.reason}")


def print_portfolio(rows: list[HoldingStatus]) -> None:
    section("💼 【目前持倉風控與損益狀態】")
    if not rows:
        print("  （沒有持倉資料）")
        return
    print(f"  {'代號':<6}{'股數':>7}{'成本':>10}{'現價':>10}{'市值':>13}{'未實現損益':>14}{'%':>9}{'距停損':>9}{'RRR':>7}")
    tv = tc = 0.0
    for r in rows:
        rrr = f"{r.rrr_now:.2f}" if r.rrr_now is not None else "-"
        print(f"  {r.symbol:<6}{r.shares:>7.0f}{r.avg_cost:>10.2f}{r.price:>10.2f}{r.market_value:>13,.2f}"
              f"{r.pnl:>+14,.2f}{r.pnl_pct:>+8.2f}%{r.dist_to_stop_pct:>+8.2f}%{rrr:>7}")
        tv += r.market_value
        tc += r.cost_value
    pnl = tv - tc
    print("  " + "-" * 82)
    print(f"  整體：成本 {money(tc)} | 市值 {money(tv)} | 未實現損益 {pnl:+,.2f} ({(tv / tc - 1) * 100:+.2f}%)"
          if tc else "  整體：N/A")


def print_alerts(alerts: list[Alert], levs: list[LevStatus]) -> None:
    section("⚠️ 【各項風險與加碼警報】")
    if levs:
        print("  [槓桿 ETF 複利損耗監控]")
        for l in levs:
            sf = f"{l.shortfall_60d:+.1%}" if l.shortfall_60d is not None else "N/A"
            print(f"   {l.symbol}（{l.leverage}x {l.underlying}）：標的20日波動率 {l.vol_underlying:.1%} | "
                  f"穿越20MA {l.crosses} 次 | 理論年化損耗 ≈ {l.drag_annual:.1%} | "
                  f"近60日實際−理論 {sf} | {'清晰上升趨勢' if l.clear_uptrend else '無清晰上升趨勢'}")
        print()
    if not alerts:
        print("  目前沒有警報 ✅")
        return
    order = {"danger": 0, "warn": 1, "opportunity": 2, "info": 3}
    for a in sorted(alerts, key=lambda x: order.get(x.level, 9)):
        print(f"  {a.text}")


# ==========================================================================
# MAIN 主流程
# ==========================================================================

def ask_float(prompt: str, default: float) -> float:
    raw = input(f"{prompt} [預設 {default}]: ").strip()
    if not raw:
        return default
    try:
        return float(raw.replace(",", ""))
    except ValueError:
        print("輸入無效，使用預設值。")
        return default


@dataclass
class AnalysisResult:
    """一次完整分析的結果：終端機版與 Streamlit 網頁版共用。"""
    macro: MacroSnapshot
    setups: list[BuySetup]
    holdings: list[HoldingStatus]
    levs: list[LevStatus]
    alerts: list[Alert]
    account: float
    risk_pct: float


def run_analysis(account: Optional[float] = None, risk_pct: Optional[float] = None) -> AnalysisResult:
    """
    執行全部策略邏輯並回傳結構化結果（不做任何 print / 推送）。
    account / risk_pct 省略時使用 CONFIG 的 ACCOUNT_SIZE / MAX_RISK_PCT。
    """
    account = ACCOUNT_SIZE if account is None else account
    risk_pct = MAX_RISK_PCT if risk_pct is None else risk_pct
    alerts: list[Alert] = []

    # 1) 大盤與宏觀
    macro = build_macro_snapshot()
    if macro.panic_alert:
        alerts.append(Alert("panic", "opportunity",
                            f"🚨 熊市極度恐慌低接機會：{' ; '.join(macro.panic_hits)}（評分 {macro.panic_score}/3），建議分批加注"))
    if macro.regime.startswith(("震盪", "空頭")):
        alerts.append(Alert(f"regime:{macro.regime[:2]}", "warn", f"⚠️ 大盤體制：{macro.regime}"))

    # 2) 觀察清單掃描
    setups: list[BuySetup] = []
    for sym in WATCHLIST:
        df = get_history(sym)
        setups.append(gate_leveraged_setup(evaluate_buy_setup(sym, df)) if not df.empty
                      else BuySetup(sym, False, reason="無資料"))
    # 個股層級恐慌：VIX 高 + 該資產 RSI<30
    if macro.vix is not None and macro.vix >= VIX_PANIC:
        for s in setups:
            if not math.isnan(s.rsi) and s.rsi < RSI_OVERSOLD:
                alerts.append(Alert(f"oversold:{s.symbol}", "opportunity",
                                    f"🚨 {s.symbol} RSI {s.rsi:.0f} < {RSI_OVERSOLD} 且 VIX ≥ {VIX_PANIC}：超賣低接觀察"))

    # 3) 持倉
    holdings = evaluate_holdings(alerts)

    # 4) 槓桿 ETF
    lev_symbols = sorted({s for s in WATCHLIST + [h["symbol"] for h in HOLDINGS] if s in LEVERAGED_MAP})
    levs = [x for x in (check_leveraged(s) for s in lev_symbols) if x is not None]
    for l in levs:
        if l.warn:
            reason = "下降趨勢" if l.downtrend else "震盪/無清晰上升趨勢"
            alerts.append(Alert(f"lev:{l.symbol}", "warn",
                                f"⚠️ {l.symbol}：目前盤勢{reason}（{l.underlying} 20日波動率 {l.vol_underlying:.0%}，"
                                f"穿越20MA {l.crosses} 次），槓桿 ETF 存在複利損耗風險，建議降低部位或轉換回一倍 ETF"))

    # 5) 資金部位計算
    for s in setups:
        if s.valid:
            s.sizing = calc_position(s.price, s.stop, s.target, account, risk_pct, s.symbol in LEVERAGED_MAP)
            sz = s.sizing
            alerts.append(Alert(f"buy:{s.symbol}", "opportunity",
                                f"✅ 買入訊號 {s.symbol} @ {s.price:.2f}：{s.trigger}｜停損 {s.stop:.2f}｜"
                                f"目標 {s.target:.2f}｜RRR {s.rrr:.2f}｜建議 {sz.shares} 股（金額 {money(sz.order_value)}，"
                                f"最大虧損 {money(sz.risk_amount)}）"))
    for h in holdings:  # 加碼訊號也附上部位建議
        if h.setup and h.setup.valid and h.setup.sizing is None:
            h.setup.sizing = calc_position(h.price, h.setup.stop, h.setup.target, account, risk_pct,
                                           h.symbol in LEVERAGED_MAP)
    return AnalysisResult(macro, setups, holdings, levs, alerts, account, risk_pct)


def run_once(args: argparse.Namespace) -> None:
    print(f"\n執行時間：{dt.datetime.now():%Y-%m-%d %H:%M:%S}（資料為最新可得的日線收市價）")
    res = run_analysis()

    # 互動模式：有買入訊號時，詢問總資金與單筆風險比例後重算部位（資料已快取，不會重複下載）
    signals = [s for s in res.setups if s.valid]
    if args.interactive and signals:
        print(f"\n偵測到 {len(signals)} 個買入訊號：{', '.join(s.symbol for s in signals)}")
        account = ask_float("請輸入總資金規模", ACCOUNT_SIZE)
        risk_pct = ask_float("請輸入單筆最大容忍虧損比例（例如 1 代表 1%）", MAX_RISK_PCT * 100) / 100
        res = run_analysis(account, risk_pct)

    print_macro(res.macro)
    print_scan(res.setups)
    print_portfolio(res.holdings)
    for h in res.holdings:
        if h.setup and h.setup.valid and h.setup.sizing:
            print(f"\n  💡 {h.symbol} 加碼建議部位（總資金 {money(res.account)}，單筆風險 {res.risk_pct:.2%}）：")
            print_sizing(h.setup.sizing, indent="     ")
    print_alerts(res.alerts, res.levs)

    if args.notify:
        print()
        push_alerts(res.alerts)


def main() -> None:
    parser = argparse.ArgumentParser(description="股票與 ETF 智慧監控與風控助理")
    parser.add_argument("--notify", action="store_true", help="推送警報到 Telegram / Discord")
    parser.add_argument("--interactive", action="store_true", help="有買入訊號時互動輸入總資金與風險比例")
    parser.add_argument("--loop", type=int, default=0, metavar="MIN", help="每 MIN 分鐘重複執行")
    args = parser.parse_args()

    if args.loop <= 0:
        run_once(args)
        return
    while True:
        _HISTORY_CACHE.clear()  # 每輪重新抓取最新資料
        try:
            run_once(args)
        except Exception as exc:  # 單輪失敗不應中斷長期監控
            print(f"[錯誤] 本輪執行失敗: {exc}")
        print(f"\n下一輪將在 {args.loop} 分鐘後執行（Ctrl+C 結束）…")
        try:
            time.sleep(args.loop * 60)
        except KeyboardInterrupt:
            print("\n已停止。")
            break


if __name__ == "__main__":
    main()
