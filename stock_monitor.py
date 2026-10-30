#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
==========================================================================
 股票與 ETF 智慧監控與風控助理 (後端引擎)
 支援互惠基金、USD/CAD 匯率換算與持倉風控
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
class Alert:
    key: str
    level: str  # danger / warn / info / success
    text: str
    symbol: str = ""  # 所屬標的；留空時會依 key（如 stop_SPCX）自動推導，無對應標的者視為全域警報
    action: str = ""  # 僅「規則建議」使用：加碼 / 持有 / 減碼 / 出場 / 停利…

    def __post_init__(self) -> None:
        if not self.symbol and "_" in self.key:
            prefix, rest = self.key.split("_", 1)
            if prefix in {"buy", "stop", "nearstop", "target", "lev", "advice"}:
                self.symbol = rest

    @property
    def is_advice(self) -> bool:
        return self.key.startswith("advice_")


def money(cur: str, value: float) -> str:
    """依幣別格式化金額：USD → US$1,234.56；CAD → C$1,234.56。"""
    return f"US${value:,.2f}" if cur == "USD" else f"C${value:,.2f}"


ALERT_SEVERITY = {"danger": 0, "warn": 1, "info": 2, "success": 3}


def group_alerts(alerts: list["Alert"]) -> tuple[list["Alert"], dict[str, list["Alert"]]]:
    """回傳 (全域警報, {標的: [警報...]})；各標的內與標的之間皆依嚴重度排序（danger 最前）。"""
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
    """CNN 取不到時的估算值（非官方）：VIX 分數 + SPY 相對 125 日均線的動能分數，各佔一半。"""
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
    """順序：手動值 → CNN（瀏覽器標頭，含日期網址備援）→ 估算值 → 無法取得。"""
    if FEAR_GREED_MANUAL is not None:
        return float(FEAR_GREED_MANUAL), "手動設定"

    urls = [_CNN_FG_URL, f"{_CNN_FG_URL}/{dt.date.today().isoformat()}"]
    for url in urls:
        try:
            resp = requests.get(url, headers=_CNN_HEADERS, timeout=8)
            if resp.status_code == 200:
                score = resp.json()["fear_and_greed"]["score"]
                return float(score), "CNN"
            print(f"[提示] CNN 恐懼貪婪指數回應 {resp.status_code}（{url}）")
        except Exception as exc:
            print(f"[提示] CNN 恐懼貪婪指數取得失敗：{exc}")

    est = _estimate_fear_greed(vix)
    if est is not None:
        return est, "估算（VIX+SPY動能，非CNN官方）"
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
    return IndexStatus(sym, c, s20, above, st, r, label)


def build_macro_snapshot() -> MacroSnapshot:
    # VIX
    vix = vix_prev = None
    vdf = get_history("^VIX", "3mo")
    if has_enough_data(vdf, 2):
        vix = float(vdf["Close"].iloc[-1])
        vix_prev = float(vdf["Close"].iloc[-2])

    fg, fg_src = _fetch_fear_greed(vix)

    indices = [s for s in (_index_status(b) for b in BENCHMARKS) if s is not None]

    # 恐慌評分
    hits: list[str] = []
    if vix is not None and vix >= VIX_PANIC:
        hits.append(f"VIX {vix:.1f} ≥ {VIX_PANIC}")
    if fg is not None and fg <= FEAR_GREED_PANIC:
        hits.append(f"恐懼貪婪指數 {fg:.0f} ≤ {FEAR_GREED_PANIC}")
    oversold = [i.symbol for i in indices if i.rsi <= RSI_OVERSOLD]
    if oversold:
        hits.append(f"RSI 超賣：{', '.join(oversold)}")

    # 市場體制
    if indices:
        n_above = sum(1 for i in indices if i.above20)
        n_down = sum(1 for i in indices if i.structure.downtrend)
        if n_above == len(indices):
            regime = "多頭（偏好順勢回調買入）"
        elif n_above == 0 or n_down >= 2:
            regime = "空頭／高風險（降低倉位、嚴守停損）"
        else:
            regime = "震盪（縮小倉位、等待確認）"
    else:
        regime = "資料不足，無法判斷"

    return MacroSnapshot(
        vix=vix, vix_prev=vix_prev, fear_greed=fg, fg_source=fg_src,
        indices=indices, regime=regime,
        panic_score=len(hits), panic_hits=hits, panic_alert=len(hits) >= 2,
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


def send_telegram_alert(message: str) -> None:
    try:
        token = os.getenv("TELEGRAM_BOT_TOKEN")
        chat_id = os.getenv("TELEGRAM_CHAT_ID")
        if not token or not chat_id:
            return
        url = f"https://api.telegram.org/bot{token}/sendMessage"
        payload = {"chat_id": chat_id, "text": message, "parse_mode": "Markdown"}
        requests.post(url, json=payload, timeout=5)
    except Exception as exc:
        print(f"[警告] 發送 Telegram 警報失敗: {exc}")


def scan_setups(account_size: float, risk_pct: float, alerts: list[Alert]) -> list[BuySetup]:
    fx = get_usdcad_rate()
    setups = []
    for sym in WATCHLIST:
        df = get_history(sym)
        if not has_enough_data(df):
            setups.append(BuySetup(symbol=sym, valid=False, reason="歷史資料不足 60 根 K 線"))
            continue

        fx_mult = fx if not (sym.endswith(".TO") or sym.endswith(".V")) else 1.0
        price = float(df["Close"].iloc[-1]) * fx_mult
        st = analyze_structure(df)
        r = float(rsi(df["Close"]).iloc[-1])

        sup = find_support(df, st)
        tgt, _ = find_target(df, st)

        if sup is None:
            sma20 = float(sma(df["Close"], 20).iloc[-1])
            sup_price = sma20 * 0.98 * fx_mult
            sup_lbl = "20日均線附近"
        else:
            sup_price, sup_lbl = sup
            sup_price *= fx_mult

        if tgt:
            tgt *= fx_mult

        stop = sup_price * (1 - STOP_BUFFER)
        risk_dist = price - stop
        reward_dist = tgt - price if tgt else 0
        rrr = reward_dist / risk_dist if risk_dist > 0 and reward_dist > 0 else 0.0

        is_uptrend = st.uptrend or (price >= float(sma(df["Close"], 20).iloc[-1]) * fx_mult)
        is_near_support = (price - sup_price) / sup_price <= SUPPORT_TOLERANCE * 1.5
        valid = is_uptrend and is_near_support and (rrr >= MIN_RRR)

        risk_amt = account_size * risk_pct
        shares = int(risk_amt / risk_dist) if risk_dist > 0 else 0
        order_val = shares * price

        max_pos_val = account_size * MAX_POSITION_PCT
        if order_val > max_pos_val:
            shares = int(max_pos_val / price) if price > 0 else 0
            order_val = shares * price

        sizing = Sizing(
            shares=shares, order_value=order_val, risk_amount=shares * risk_dist if risk_dist > 0 else 0,
            stop_pct=(risk_dist / price) * 100 if price > 0 else 0, reward_amount=shares * reward_dist if reward_dist > 0 else 0,
            hard_stop_2pct=price * 0.98
        )
        reason_text = f"回調至{sup_lbl} {sup_price:.2f} CAD，RRR 達 {rrr:.2f}" if valid else f"目前價格未緊貼支撐或 RRR ({rrr:.2f}) 未達門檻"

        setups.append(BuySetup(
            symbol=sym, valid=valid, price=price, trigger=f"回調至{sup_lbl} {sup_price:.2f} CAD",
            support=sup_price, stop=stop, target=tgt, rrr=rrr, reason=reason_text, rsi=r, sizing=sizing
        ))

        if valid:
            cur_s = "CAD" if fx_mult == 1.0 else "USD"
            alerts.append(Alert(
                f"buy_{sym}", "success",
                f"🟢 {sym} 符合買點：回調至{sup_lbl} {money(cur_s, sup_price / fx_mult)}，"
                f"停損 {money(cur_s, stop / fx_mult)}，目標 {money(cur_s, (tgt or 0) / fx_mult)}，RRR 達 {rrr:.2f}"))
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
    # --- 原幣別資訊（avg_cost/price/stop/target 等欄位皆為 CAD；以下為原幣） ---
    currency: str = "CAD"           # "USD" 或 "CAD"
    fx_mult: float = 1.0            # 原幣 → CAD 的乘數（CAD 標的為 1）
    avg_cost_native: float = 0.0
    price_native: float = 0.0
    stop_native: float = 0.0
    target_native: float = 0.0
    has_live: bool = True           # 是否有即時報價（互惠基金為 False）
    rsi: Optional[float] = None
    trend: str = ""                 # 上升結構 / 下降結構 / 盤整
    above20: Optional[bool] = None
    weight_pct: float = 0.0         # 市值占總資金規模的比例


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
        if df.empty:
            price = avg_cost
            fx_mult = 1.0
        else:
            fx_mult = fx if not (sym.endswith(".TO") or sym.endswith(".V")) else 1.0
            price = float(df["Close"].iloc[-1]) * fx_mult

        st = analyze_structure(df) if has_enough_data(df) else Structure([], [])
        stop = data["stop"] * fx_mult if data["stop"] > 0 else 0
        if not stop or stop <= 0:
            sup = find_support(df, st) if not df.empty else None
            stop = (sup[0] * (1 - STOP_BUFFER) if sup else avg_cost * 0.90) * fx_mult

        target = data["target"] * fx_mult if data["target"] > 0 else 0
        if not target or target <= 0:
            t_val, _ = find_target(df, st) if not df.empty else (avg_cost * 1.20, "")
            target = t_val * fx_mult

        # 側邊欄輸入的平均成本、停損、目標價皆為「該標的原幣」（USD 標的填美元、.TO 填加幣）
        currency = "CAD" if (sym.endswith(".TO") or sym.endswith(".V") or not has_live) else "USD"
        avg_cost_cad = avg_cost * fx_mult          # 成本換算為 CAD，才能與 CAD 市值相比
        mv, cv = shares * price, shares * avg_cost_cad
        rrr_now = (target - price) / (price - stop) if stop < price < target else None
        dist_stop = (price / stop - 1) * 100 if stop > 0 else 0.0

        if has_enough_data(df):
            rsi_val: Optional[float] = float(rsi(df["Close"]).iloc[-1])
            above20: Optional[bool] = bool(float(df["Close"].iloc[-1]) >= float(sma(df["Close"], 20).iloc[-1]))
            trend = "上升結構" if st.uptrend else ("下降結構" if st.downtrend else "盤整")
        else:
            rsi_val, above20, trend = None, None, ""

        results.append(HoldingStatus(
            symbol=sym, shares=shares, avg_cost=avg_cost_cad, price=price,
            stop=stop, target=target, market_value=mv, cost_value=cv,
            pnl=mv - cv, pnl_pct=(price / avg_cost_cad - 1) * 100 if avg_cost_cad > 0 else 0.0,
            dist_to_stop_pct=dist_stop,
            rrr_now=rrr_now, risk_level=risk_lvl, account_type=acc_type, is_long_term=is_lt,
            currency=currency, fx_mult=fx_mult, avg_cost_native=avg_cost,
            price_native=price / fx_mult, stop_native=stop / fx_mult, target_native=target / fx_mult,
            has_live=has_live, rsi=rsi_val, trend=trend, above20=above20,
            weight_pct=(mv / account_size * 100) if account_size > 0 else 0.0,
        ))

        # 警報（僅針對有即時報價的標的；互惠基金無報價不觸發）；金額一律以該標的原幣顯示
        p_n, s_n, t_n = price / fx_mult, stop / fx_mult, target / fx_mult
        if has_live and not is_lt:
            if price <= stop:
                alerts.append(Alert(f"stop_{sym}", "danger",
                                    f"🔴 {sym} 現價 {money(currency, p_n)} 已跌破停損價 {money(currency, s_n)}，請檢視是否出場"))
            elif dist_stop <= NEAR_STOP_WARN_PCT:
                alerts.append(Alert(f"nearstop_{sym}", "warn",
                                    f"🟠 {sym} 距離停損僅 {dist_stop:.1f}%（現價 {money(currency, p_n)}／停損 {money(currency, s_n)}）"))
            if price >= target:
                alerts.append(Alert(f"target_{sym}", "info",
                                    f"🎯 {sym} 現價 {money(currency, p_n)} 已達目標價 {money(currency, t_n)}，可考慮分批停利"))
        if sym in LEVERAGED_MAP:
            ref, mult = LEVERAGED_MAP[sym]
            alerts.append(Alert(f"lev_{sym}", "warn",
                                f"⚡ {sym} 為 {mult} 倍槓桿 ETF（追蹤 {ref}），長期持有有波動耗損，建議短線操作並嚴設停損"))
    return results


def advise_holding(h: HoldingStatus, setup: Optional[BuySetup], account_size: float) -> Optional[tuple[str, str, str]]:
    """依使用者自訂的風控規則，對單一持倉給出「加碼 / 持有 / 減碼 / 出場」訊號。

    回傳 (動作, 警報等級, 理由)；無即時報價的資產（互惠基金等）回傳 None。
    這是規則式訊號，不是投資建議。優先順序：跌破停損 > 達目標 > 超過倉位上限 >
    接近停損且趨勢轉弱 > 出現買點可加碼 > 獲利後 RSI 偏高 > 持有。
    """
    if not h.has_live:
        return None
    cur = h.currency
    p, s, t = h.price_native, h.stop_native, h.target_native
    max_pct = MAX_POSITION_PCT * 100
    cap_val = account_size * MAX_POSITION_PCT
    w = h.weight_pct

    # 1) 跌破停損
    if h.price <= h.stop:
        if h.is_long_term:
            return ("檢視持有理由", "warn",
                    f"長線資產已跌破參考停損價 {money(cur, s)}（現價 {money(cur, p)}）；請重新評估持有理由，而非機械式停損")
        return ("減碼／出場", "danger",
                f"現價 {money(cur, p)} 已跌破停損價 {money(cur, s)}，依停損紀律應減碼或出場")

    # 2) 達目標價
    if h.price >= h.target:
        return ("分批停利", "warn",
                f"現價 {money(cur, p)} 已達目標價 {money(cur, t)}，可考慮分批賣出（如 1/3～1/2）並上移停損保護獲利")

    # 3) 單一標的倉位過大
    if w > max_pct and h.price > 0:
        trim = math.ceil((h.market_value - cap_val) / h.price)
        return ("減碼", "warn",
                f"倉位占總資金 {w:.1f}%，超過單一標的上限 {max_pct:.0f}%；"
                f"回到上限需減碼約 {trim} 股（約 {money(cur, trim * p)}）")

    # 4) 接近停損且趨勢轉弱
    if (not h.is_long_term) and h.dist_to_stop_pct <= NEAR_STOP_WARN_PCT and h.trend == "下降結構":
        return ("減碼／收緊停損", "warn",
                f"距停損僅 {h.dist_to_stop_pct:.1f}% 且價格結構為下降（高低點同步下移）；"
                f"不宜加碼，可先減碼或嚴守停損 {money(cur, s)}")

    # 5) 出現買點 → 可加碼（受倉位上限限制）
    if setup is not None and setup.valid:
        room_val = cap_val - h.market_value
        room_sh = int(room_val / h.price) if (room_val > 0 and h.price > 0) else 0
        rule_sh = setup.sizing.shares if setup.sizing else 0
        add = min(rule_sh, room_sh)
        if add >= 1:
            setup_stop_n = (setup.stop or h.stop) / h.fx_mult
            return ("可加碼", "success",
                    f"出現買點（回調至支撐 {money(cur, (setup.support or h.price) / h.fx_mult)}，RRR {setup.rrr:.2f}），倉位 {w:.1f}% 低於上限 {max_pct:.0f}%；"
                    f"依單筆風險規則與倉位上限，最多約加 {add} 股（約 {money(cur, add * p)}），停損 {money(cur, setup_stop_n)}")
        return ("持有", "info",
                f"雖有買點，但倉位 {w:.1f}% 已接近上限 {max_pct:.0f}%，沒有加碼空間")

    # 6) 大幅獲利且 RSI 過熱 → 考慮部分停利
    if h.rsi is not None and h.rsi >= 70 and h.pnl_pct >= 20:
        return ("持有（可部分停利）", "info",
                f"已獲利 {h.pnl_pct:.1f}% 且 RSI {h.rsi:.0f} 偏高（過熱），不宜追高加碼；可考慮部分停利，停損 {money(cur, s)}")

    # 7) 其餘：持有
    extra = f"；距停損 {h.dist_to_stop_pct:.1f}%（停損 {money(cur, s)}）"
    if h.dist_to_stop_pct <= NEAR_STOP_WARN_PCT:
        extra += "，接近停損，守住停損價"
    rsi_txt = f"，RSI {h.rsi:.0f}" if h.rsi is not None else ""
    return ("持有", "info", f"趨勢：{h.trend or '資料不足'}{rsi_txt}{extra}；目前沒有新的買點，不建議加碼")


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

    # 持倉規則建議（加碼 / 持有 / 減碼 / 出場）；需要同時用到持倉與買點掃描結果
    setup_map = {s.symbol: s for s in setups}
    for h in holdings:
        adv = advise_holding(h, setup_map.get(h.symbol), account_size)
        if adv is None:
            continue
        action, level, reason = adv
        alerts.append(Alert(f"advice_{h.symbol}", level, f"🧭 {h.symbol} 規則建議【{action}】{reason}", action=action))
    return AnalysisResult(macro=macro, setups=setups, holdings=holdings, alerts=alerts)
