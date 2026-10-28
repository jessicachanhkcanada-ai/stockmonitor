# -*- coding: utf-8 -*-
"""
==========================================================================
 📈 智慧股票監控與風控助理 — Streamlit 網頁版
==========================================================================

本檔負責「介面」；所有策略邏輯都在同資料夾的 stock_monitor.py（分析引擎）。
兩個檔案必須放在同一個資料夾。

【本機執行】
    pip install -r requirements.txt
    streamlit run streamlit_app.py
    → 瀏覽器開啟 http://localhost:8501

【機密設定（建議）】
    環境變數或 .streamlit/secrets.toml 皆可：
        TELEGRAM_BOT_TOKEN = "123456:ABC..."
        TELEGRAM_CHAT_ID   = "123456789"
        APP_PASSWORD       = "你的網頁密碼"   # 有設定才會啟用密碼保護

【注意】
    - 設定（自選股、持倉、風控參數）會存在 settings.json；Telegram Token 不會寫入檔案。
    - 網頁只有「開著」的時候才會分析與推送。要 24 小時自動推送，請另外執行：
          python stock_monitor.py --notify --loop 60
    - 技術訊號僅供參考，不構成投資建議。
"""

from __future__ import annotations

import datetime as dt
import hmac
import json
import os
import re
import time
from typing import Optional

import pandas as pd
import requests
import streamlit as st

import stock_monitor as sm

SETTINGS_FILE = "settings.json"
CACHE_TTL_SECONDS = 15 * 60   # 價格資料超過 15 分鐘自動重新抓取
HOLD_COLS = ["symbol", "shares", "avg_cost", "stop", "target"]
LEVEL_ORDER = {"danger": 0, "warn": 1, "opportunity": 2, "info": 3}

st.set_page_config(page_title="智慧股票監控與風控助理", page_icon="📈", layout="wide")


# ==========================================================================
# 工具：機密、設定檔、密碼保護
# ==========================================================================

def get_secret(name: str, default: str = "") -> str:
    """先讀環境變數，再讀 Streamlit secrets；都沒有就回傳預設值。"""
    val = os.getenv(name)
    if val:
        return val
    try:
        return str(st.secrets[name])
    except Exception:
        return default


def default_settings() -> dict:
    """預設設定：取自 stock_monitor.py 的 CONFIG 區。"""
    return {
        "watchlist": list(sm.WATCHLIST),
        "holdings": [dict(h) for h in sm.HOLDINGS],
        "account": float(sm.ACCOUNT_SIZE),
        "risk_pct": sm.MAX_RISK_PCT * 100,
        "max_pos_pct": sm.MAX_POSITION_PCT * 100,
        "min_rrr": float(sm.MIN_RRR),
        "vix_panic": float(sm.VIX_PANIC),
        "fg_panic": float(sm.FEAR_GREED_PANIC),
        "rsi_oversold": float(sm.RSI_OVERSOLD),
        "support_tol_pct": sm.SUPPORT_TOLERANCE * 100,
        "lev_vol_warn_pct": sm.LEV_VOL_WARN * 100,
        "fg_manual_on": False,
        "fg_manual": 50.0,
    }


def load_settings() -> dict:
    cfg = default_settings()
    try:
        with open(SETTINGS_FILE, "r", encoding="utf-8") as f:
            cfg.update(json.load(f))
    except Exception:
        pass
    return cfg


def save_settings(cfg: dict) -> Optional[str]:
    try:
        with open(SETTINGS_FILE, "w", encoding="utf-8") as f:
            json.dump(cfg, f, ensure_ascii=False, indent=2)
        return None
    except Exception as exc:
        return str(exc)


def require_password() -> None:
    """若設定了 APP_PASSWORD，必須先通過密碼才能看到畫面（個人財務資料請務必啟用）。"""
    pw = get_secret("APP_PASSWORD")
    if not pw or st.session_state.get("authed"):
        return
    st.title("🔒 請輸入密碼")
    entered = st.text_input("密碼", type="password")
    if entered:
        if hmac.compare_digest(entered, pw):
            st.session_state["authed"] = True
            st.rerun()
        else:
            st.error("密碼錯誤")
    st.stop()


# ==========================================================================
# 工具：解析輸入、套用設定到分析引擎
# ==========================================================================

def parse_watchlist(text: str) -> list[str]:
    seen, out = set(), []
    for tok in re.split(r"[,\s;，]+", text.upper()):
        if tok and tok not in seen:
            seen.add(tok)
            out.append(tok)
    return out


def holdings_to_df(holdings: list[dict]) -> pd.DataFrame:
    df = pd.DataFrame(holdings, columns=HOLD_COLS)
    return df.astype({"symbol": "object", "shares": "float64", "avg_cost": "float64",
                      "stop": "float64", "target": "float64"})


def clean_holdings(df: pd.DataFrame) -> list[dict]:
    """把表格編輯器的內容整理成引擎使用的持倉清單；允許 stop 與 target 留空。"""
    out = []
    for _, r in df.iterrows():
        sym = str(r["symbol"]).strip().upper() if pd.notna(r["symbol"]) else ""
        if not sym or sym == "NONE" or sym == "NAN":
            continue
        try:
            shares = float(r["shares"])
            cost = float(r["avg_cost"])
            # 允許 stop / target 留空或為 None，若沒填則設為 0.0，交由後端自動計算
            stop = float(r["stop"]) if pd.notna(r["stop"]) and str(r["stop"]).upper() != "NONE" else 0.0
            target = float(r["target"]) if pd.notna(r["target"]) and str(r["target"]).upper() != "NONE" else 0.0
        except (TypeError, ValueError):
            continue
            
        if pd.isna(shares) or pd.isna(cost) or shares <= 0 or cost <= 0:
            continue
            
        out.append({"symbol": sym, "shares": shares, "avg_cost": cost, "stop": stop, "target": target})
    return out


def apply_to_engine(cfg: dict) -> None:
    """把網頁上的設定寫入 stock_monitor 模組的全域參數，之後的分析就會採用。"""
    sm.WATCHLIST = cfg["watchlist"]
    sm.HOLDINGS = cfg["holdings"]
    sm.ACCOUNT_SIZE = cfg["account"]
    sm.MAX_RISK_PCT = cfg["risk_pct"] / 100
    sm.MAX_POSITION_PCT = cfg["max_pos_pct"] / 100
    sm.MIN_RRR = cfg["min_rrr"]
    sm.VIX_PANIC = cfg["vix_panic"]
    sm.FEAR_GREED_PANIC = cfg["fg_panic"]
    sm.RSI_OVERSOLD = cfg["rsi_oversold"]
    sm.SUPPORT_TOLERANCE = cfg["support_tol_pct"] / 100
    sm.LEV_VOL_WARN = cfg["lev_vol_warn_pct"] / 100
    sm.FEAR_GREED_MANUAL = cfg["fg_manual"] if cfg["fg_manual_on"] else None


# ==========================================================================
# Telegram
# ==========================================================================

def tg_send(token: str, chat_id: str, text: str) -> tuple[bool, str]:
    """發送 Telegram 訊息，回傳 (是否成功, 說明)。錯誤訊息會遮蔽 Token。"""
    token, chat_id = token.strip(), chat_id.strip()
    if not token or not chat_id:
        return False, "請先在側邊欄填寫 Bot Token 與 Chat ID"
    url = f"https://api.telegram.org/bot{token}/sendMessage"
    try:
        for i in range(0, max(len(text), 1), 3800):
            r = requests.post(url, json={"chat_id": chat_id, "text": text[i:i + 3800]}, timeout=15)
            data = r.json()
            if not data.get("ok"):
                return False, f"Telegram 回覆：{data.get('description', r.text)}"
        return True, "已送出，請到 Telegram 查看"
    except Exception as exc:
        return False, str(exc).replace(token, "***")


def build_summary(res: sm.AnalysisResult) -> str:
    m = res.macro
    lines = [f"📡 智慧監控 {dt.datetime.now():%Y-%m-%d %H:%M}", f"大盤體制：{m.regime}"]
    bits = []
    if m.vix is not None:
        bits.append(f"VIX {m.vix:.1f}")
    if m.fear_greed is not None:
        bits.append(f"恐懼貪婪 {m.fear_greed:.0f}（{m.fg_source}）")
    if bits:
        lines.append("｜".join(bits))
    lines.append("")
    if res.alerts:
        lines += [a.text for a in sorted(res.alerts, key=lambda a: LEVEL_ORDER.get(a.level, 9))]
    else:
        lines.append("目前沒有警報 ✅")
    return "\n".join(lines)


# ==========================================================================
# 版面：各區塊
# ==========================================================================

def structure_text(s) -> str:
    hh = "HH" if s.hh else ("LH" if s.lh else "-")
    hl = "HL" if s.hl else ("LL" if s.ll else "-")
    return f"{hh}/{hl}"


def render_macro(res: sm.AnalysisResult) -> None:
    m = res.macro
    c1, c2, c3, c4 = st.columns(4)
    if m.vix is not None:
        delta = f"{m.vix - m.vix_prev:+.2f}" if m.vix_prev else None
        c1.metric("VIX", f"{m.vix:.2f}", delta, delta_color="inverse")
    else:
        c1.metric("VIX", "N/A")
    c2.metric("恐懼貪婪指數", f"{m.fear_greed:.0f}" if m.fear_greed is not None else "N/A")
    c2.caption(f"來源：{m.fg_source}")
    c3.metric("恐慌低接評分", f"{m.panic_score} / 3")
    c4.metric("基準 ETF 上升趨勢", f"{sum(1 for i in m.indices if i.structure.uptrend and i.above20)} / {len(m.indices)}")

    if m.regime.startswith("多頭"):
        st.success(f"大盤體制：{m.regime}")
    elif m.regime.startswith("空頭"):
        st.error(f"大盤體制：{m.regime}")
    else:
        st.warning(f"大盤體制：{m.regime}")

    if m.panic_alert:
        st.error("🚨 熊市極度恐慌低接機會：" + " ； ".join(m.panic_hits) + "。條件越多越接近低點，適合分批小注加注。")
    elif m.panic_hits:
        st.info("恐慌條件部分出現：" + " ； ".join(m.panic_hits))

    rows = [{
        "代號": i.symbol, "收市價": i.close, "20MA": i.sma20,
        "位於20MA": "上方" if i.above20 else "下方", "RSI": i.rsi,
        "結構": structure_text(i.structure), "趨勢": i.label,
    } for i in m.indices]
    if rows:
        st.dataframe(pd.DataFrame(rows), hide_index=True, column_config={
            "收市價": st.column_config.NumberColumn(format="%.2f"),
            "20MA": st.column_config.NumberColumn(format="%.2f"),
            "RSI": st.column_config.NumberColumn(format="%.1f"),
        })
    st.caption("結構 HH/HL = 一頂高於一頂 / 一底高於一底（上升趨勢）；LH/LL = 一頂低於一頂 / 一底低於一底。")


def render_sizing(sz: sm.Sizing) -> None:
    d1, d2, d3, d4 = st.columns(4)
    d1.metric("建議股數", f"{sz.shares:,}")
    d2.metric("下單金額", f"{sz.order_value:,.2f}")
    d3.metric("最大虧損", f"{sz.risk_amount:,.2f}")
    d4.metric("停損距離", f"{sz.stop_pct:.2%}")
    st.caption(f"到目標價約獲利 {sz.reward_amount:,.2f}｜入場價下方 2% 硬停損價參考：{sz.hard_stop_2pct:.2f}")
    for n in sz.notes:
        st.warning(n)


def render_scan(res: sm.AnalysisResult) -> None:
    hits = [s for s in res.setups if s.valid]
    st.markdown(f"**共 {len(res.setups)} 檔，符合買入條件 {len(hits)} 檔**（RRR ≥ {sm.MIN_RRR:g}）｜"
                f"部位依總資金 {res.account:,.0f}、單筆風險 {res.risk_pct:.2%} 計算")
    for s in hits:
        with st.container(border=True):
            st.subheader(f"✅ {s.symbol}")
            st.caption(s.trigger)
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("現價", f"{s.price:.2f}")
            c2.metric("停損", f"{s.stop:.2f}", f"{s.stop / s.price - 1:.2%}", delta_color="off")
            c3.metric("目標", f"{s.target:.2f}", f"{s.target / s.price - 1:+.2%}", delta_color="off")
            c4.metric("RRR", f"{s.rrr:.2f}")
            if s.sizing:
                render_sizing(s.sizing)
            st.caption(f"目標來源：{s.target_note}｜RSI {s.rsi:.0f}")
    if not hits:
        st.info("目前沒有符合條件的買點。下方列出每檔未觸發的原因。")

    rows = []
    for s in res.setups:
        rows.append({
            "代號": s.symbol,
            "狀態": "✅ 買入訊號" if s.valid else "⏳ 未觸發",
            "現價": None if pd.isna(s.price) else s.price,
            "觸發／原因": s.trigger if s.valid else s.reason,
            "停損": s.stop, "目標": s.target, "RRR": s.rrr,
            "RSI": None if pd.isna(s.rsi) else s.rsi,
        })
    st.dataframe(pd.DataFrame(rows), hide_index=True, column_config={
        "現價": st.column_config.NumberColumn(format="%.2f"),
        "停損": st.column_config.NumberColumn(format="%.2f"),
        "目標": st.column_config.NumberColumn(format="%.2f"),
        "RRR": st.column_config.NumberColumn(format="%.2f"),
        "RSI": st.column_config.NumberColumn(format="%.0f"),
    })


def holding_flag(h: sm.HoldingStatus) -> str:
    if h.price <= h.stop:
        return "🔴 已跌穿停損"
    if h.price >= h.target:
        return "🎯 已達目標"
    if h.dist_to_stop_pct < 3:
        return "🟡 接近停損"
    return "🟢 正常"


def render_portfolio(res: sm.AnalysisResult) -> None:
    rows = res.holdings
    if not rows:
        st.info("尚未輸入持倉。請在左側「目前持倉」表格新增資料。")
        return
    tv = sum(r.market_value for r in rows)
    tc = sum(r.cost_value for r in rows)
    pnl = tv - tc
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("總成本", f"{tc:,.2f}")
    c2.metric("總市值", f"{tv:,.2f}")
    c3.metric("未實現損益", f"{pnl:+,.2f}", f"{(tv / tc - 1) * 100:+.2f}%" if tc else None)
    c4.metric("持倉檔數", len(rows))

    df = pd.DataFrame([{
        "代號": r.symbol, "狀態": holding_flag(r), "股數": r.shares, "成本": r.avg_cost, "現價": r.price,
        "市值": r.market_value, "未實現損益": r.pnl, "損益%": r.pnl_pct,
        "停損": r.stop, "距停損%": r.dist_to_stop_pct, "目標": r.target, "現時RRR": r.rrr_now,
        "加碼訊號": "💡 可分批加碼" if (r.setup and r.setup.valid and r.price > r.stop) else "",
    } for r in rows])
    st.dataframe(df, hide_index=True, column_config={
        "股數": st.column_config.NumberColumn(format="%.0f"),
        "成本": st.column_config.NumberColumn(format="%.2f"),
        "現價": st.column_config.NumberColumn(format="%.2f"),
        "市值": st.column_config.NumberColumn(format="%.2f"),
        "未實現損益": st.column_config.NumberColumn(format="%+.2f"),
        "損益%": st.column_config.NumberColumn(format="%+.2f"),
        "停損": st.column_config.NumberColumn(format="%.2f"),
        "距停損%": st.column_config.NumberColumn(format="%+.2f"),
        "目標": st.column_config.NumberColumn(format="%.2f"),
        "現時RRR": st.column_config.NumberColumn(format="%.2f"),
    })
    st.caption("停損以「收市價」跌穿為準；現時RRR 在現價已高於目標價或已跌穿停損時不顯示。")

    for r in rows:
        if r.setup and r.setup.valid and r.setup.sizing and r.price > r.stop:
            with st.expander(f"💡 {r.symbol} 加碼建議部位"):
                st.caption(r.setup.trigger)
                render_sizing(r.setup.sizing)


def render_alerts(res: sm.AnalysisResult) -> None:
    if res.levs:
        st.markdown("#### 槓桿 ETF 複利損耗監控")
        df = pd.DataFrame([{
            "代號": l.symbol, "對應標的": f"{l.leverage}x {l.underlying}",
            "標的20日波動率": l.vol_underlying * 100, "穿越20MA次數": l.crosses,
            "理論年化損耗%": l.drag_annual * 100,
            "近60日實際−理論%": None if l.shortfall_60d is None else l.shortfall_60d * 100,
            "趨勢": "清晰上升" if l.clear_uptrend else ("下降趨勢" if l.downtrend else "震盪／未明"),
            "狀態": "⚠️ 注意損耗" if l.warn else "✅ 暫無警告",
        } for l in res.levs])
        st.dataframe(df, hide_index=True, column_config={
            "標的20日波動率": st.column_config.NumberColumn("標的20日波動率 (%)", format="%.1f"),
            "理論年化損耗%": st.column_config.NumberColumn("理論年化損耗 (%)", format="%.1f"),
            "近60日實際−理論%": st.column_config.NumberColumn("近60日實際−理論 (%)", format="%+.1f"),
        })
        st.caption("理論年化損耗 ≈ 0.5 × 槓桿倍數 × (倍數−1) × 年化波動率²；實際−理論為粗略觀察，路徑不同結果會差很多。")

    st.markdown("#### 警報清單")
    if not res.alerts:
        st.success("目前沒有警報 ✅")
        return
    for a in sorted(res.alerts, key=lambda a: LEVEL_ORDER.get(a.level, 9)):
        if a.level == "danger":
            st.error(a.text)
        elif a.level == "warn":
            st.warning(a.text)
        elif a.level == "opportunity":
            st.success(a.text)
        else:
            st.info(a.text)


def render_calculator(default_account: float, default_risk_pct: float) -> None:
    st.markdown("手動輸入任何一筆交易，計算建議股數與風險。")
    c1, c2, c3 = st.columns(3)
    entry = c1.number_input("買入價", min_value=0.0, value=100.0, step=0.5, key="calc_entry")
    stop = c2.number_input("停損價", min_value=0.0, value=95.0, step=0.5, key="calc_stop")
    target = c3.number_input("目標價（可填 0 略過）", min_value=0.0, value=115.0, step=0.5, key="calc_target")
    d1, d2, d3 = st.columns(3)
    account = d1.number_input("總資金規模", min_value=0.0, value=float(default_account), step=1000.0, key="calc_account")
    risk_pct = d2.number_input("單筆最大容忍虧損 (%)", min_value=0.1, max_value=20.0,
                               value=float(default_risk_pct), step=0.1, key="calc_risk")
    leveraged = d3.checkbox("這是槓桿 ETF", key="calc_lev")

    if entry <= 0 or stop <= 0 or stop >= entry:
        st.warning("請輸入有效的價格：停損價必須低於買入價。")
        return
    sz = sm.calc_position(entry, stop, target if target > entry else None, account, risk_pct / 100, leveraged)
    render_sizing(sz)
    if target > entry:
        rrr = (target - entry) / (entry - stop)
        (st.success if rrr >= sm.MIN_RRR else st.warning)(
            f"風險報酬比 RRR = {rrr:.2f}（門檻 {sm.MIN_RRR:g}）" + ("" if rrr >= sm.MIN_RRR else "，低於門檻，不建議進場"))


def render_chart(res: sm.AnalysisResult) -> None:
    symbols = list(dict.fromkeys(sm.WATCHLIST + [h["symbol"] for h in sm.HOLDINGS] + sm.BENCHMARKS))
    sym = st.selectbox("選擇標的", symbols)
    df = sm.get_history(sym)
    if df.empty:
        st.warning("無法取得此標的資料。")
        return
    close = df["Close"]
    chart_df = pd.DataFrame({"收市價": close, "20MA": sm.sma(close, 20), "50MA": sm.sma(close, 50)}).tail(150)
    st.line_chart(chart_df)
    struct = sm.analyze_structure(df)
    hi = "、".join(f"{s.date:%m/%d} {s.price:.2f}" for s in struct.highs[-3:]) or "無"
    lo = "、".join(f"{s.date:%m/%d} {s.price:.2f}" for s in struct.lows[-3:]) or "無"
    st.caption(f"最近波段高點：{hi}")
    st.caption(f"最近波段低點：{lo}")


# ==========================================================================
# 主程式
# ==========================================================================

def main() -> None:
    require_password()
    cfg = load_settings()

    # ---------------- 側邊欄 ----------------
    with st.sidebar:
        st.title("⚙️ 設定")

        with st.expander("📋 自選股", expanded=True):
            wl_text = st.text_area("代號（用逗號、空格或換行分隔）", value=", ".join(cfg["watchlist"]), height=90)

        with st.expander("💼 目前持倉", expanded=True):
            st.caption("可直接在表格編輯；底部可新增／刪除列。停損以收市價為準。")
            hold_df = st.data_editor(
                holdings_to_df(cfg["holdings"]), num_rows="dynamic", key="holdings_editor",
                column_config={
                    "symbol": st.column_config.TextColumn("代號"),
                    "shares": st.column_config.NumberColumn("股數", min_value=0.0, step=1.0),
                    "avg_cost": st.column_config.NumberColumn("平均成本", min_value=0.0, format="%.2f"),
                    "stop": st.column_config.NumberColumn("停損/支撐", min_value=0.0, format="%.2f"),
                    "target": st.column_config.NumberColumn("目標價", min_value=0.0, format="%.2f"),
                })

        with st.expander("💰 資金與風控", expanded=True):
            account = st.number_input("總資金規模", min_value=0.0, value=float(cfg["account"]), step=1000.0)
            risk_pct = st.number_input("單筆最大容忍虧損 (%)", min_value=0.1, max_value=20.0,
                                       value=float(cfg["risk_pct"]), step=0.1)
            max_pos = st.number_input("單一標的最大倉位 (%)", min_value=1.0, max_value=100.0,
                                      value=float(cfg["max_pos_pct"]), step=1.0)
            min_rrr = st.number_input("最低風險報酬比 RRR", min_value=1.0, max_value=10.0,
                                      value=float(cfg["min_rrr"]), step=0.5)

        with st.expander("🔧 進階參數"):
            vix_panic = st.number_input("VIX 恐慌門檻 ≥", value=float(cfg["vix_panic"]), step=0.5)
            fg_panic = st.number_input("恐懼貪婪門檻 ≤", value=float(cfg["fg_panic"]), step=1.0)
            rsi_os = st.number_input("RSI 超賣門檻 <", value=float(cfg["rsi_oversold"]), step=1.0)
            sup_tol = st.number_input("支撐容忍距離 (%)", min_value=0.5, max_value=10.0,
                                      value=float(cfg["support_tol_pct"]), step=0.5)
            lev_vol = st.number_input("槓桿警告：標的20日波動率 ≥ (%)", min_value=5.0, max_value=100.0,
                                      value=float(cfg["lev_vol_warn_pct"]), step=1.0)
            fg_on = st.checkbox("手動指定恐懼貪婪值（CNN 抓不到時）", value=bool(cfg["fg_manual_on"]))
            fg_val = st.slider("恐懼貪婪值", 0.0, 100.0, float(cfg["fg_manual"]), disabled=not fg_on)

        with st.expander("📨 Telegram 推播", expanded=True):
            token = st.text_input("Bot Token", value=get_secret("TELEGRAM_BOT_TOKEN"), type="password",
                                  help="向 @BotFather 建立機器人後取得。不會寫入設定檔。")
            chat_id = st.text_input("Chat ID", value=get_secret("TELEGRAM_CHAT_ID"),
                                    help="先對你的機器人傳一句話，再開 api.telegram.org/bot<TOKEN>/getUpdates 查看 chat.id")
            auto_push = st.checkbox("每次分析後自動推送「新」警報（同日去重）", value=False)
            if st.button("📨 發送測試通知", type="primary"):
                with st.spinner("傳送中…"):
                    ok, msg = tg_send(token, chat_id, f"✅ 測試成功！智慧股票監控已連線。\n{dt.datetime.now():%Y-%m-%d %H:%M:%S}")
                (st.success if ok else st.error)(msg)

        if st.button("💾 儲存設定"):
            to_save = {
                "watchlist": parse_watchlist(wl_text), "holdings": clean_holdings(hold_df),
                "account": account, "risk_pct": risk_pct, "max_pos_pct": max_pos, "min_rrr": min_rrr,
                "vix_panic": vix_panic, "fg_panic": fg_panic, "rsi_oversold": rsi_os,
                "support_tol_pct": sup_tol, "lev_vol_warn_pct": lev_vol,
                "fg_manual_on": fg_on, "fg_manual": fg_val,
            }
            err = save_settings(to_save)
            st.success(f"已儲存到 {SETTINGS_FILE}（Token 不會被儲存）") if err is None else st.error(f"儲存失敗：{err}")

    live_cfg = {
        "watchlist": parse_watchlist(wl_text), "holdings": clean_holdings(hold_df),
        "account": account, "risk_pct": risk_pct, "max_pos_pct": max_pos, "min_rrr": min_rrr,
        "vix_panic": vix_panic, "fg_panic": fg_panic, "rsi_oversold": rsi_os,
        "support_tol_pct": sup_tol, "lev_vol_warn_pct": lev_vol, "fg_manual_on": fg_on, "fg_manual": fg_val,
    }
    apply_to_engine(live_cfg)

    # ---------------- 標題與操作列 ----------------
    st.title("📈 智慧股票監控與風控助理")
    st.caption("以日線收市價為基礎的趨勢與風控提示，僅供個人參考，不構成投資建議。")

    now = time.time()
    if "last_fetch" not in st.session_state:
        st.session_state["last_fetch"] = 0.0
    top1, top2, top3 = st.columns([1, 1, 3])
    force_refresh = top1.button("🔄 重新抓取數據")
    push_now = top2.button("📤 推送目前警報")
    if force_refresh or now - st.session_state["last_fetch"] > CACHE_TTL_SECONDS:
        sm._HISTORY_CACHE.clear()
        st.session_state["last_fetch"] = now
    top3.caption(f"數據更新時間：{dt.datetime.fromtimestamp(st.session_state['last_fetch']):%Y-%m-%d %H:%M:%S}"
                 f"（每 {CACHE_TTL_SECONDS // 60} 分鐘自動更新，或按左側按鈕）")

    # ---------------- 執行分析 ----------------
    try:
        with st.spinner("正在分析大盤、觀察清單與持倉…"):
            res = sm.run_analysis(account, risk_pct / 100)
    except Exception as exc:
        st.error(f"分析失敗：{exc}")
        st.stop()

    # ---------------- 推播 ----------------
    if push_now:
        ok, msg = tg_send(token, chat_id, build_summary(res))
        (st.success if ok else st.error)(f"推送：{msg}")
    elif auto_push and token and chat_id:
        fresh = sm.filter_new_alerts(res.alerts)
        if fresh:
            ok, msg = tg_send(token, chat_id, "📡 新警報\n" + "\n".join(a.text for a in fresh))
            (st.toast if ok else st.error)(f"已自動推送 {len(fresh)} 則新警報" if ok else f"自動推送失敗：{msg}")

    # ---------------- 主畫面 ----------------
    n_danger = sum(1 for a in res.alerts if a.level == "danger")
    t1, t2, t3, t4, t5, t6 = st.tabs([
        "📊 宏觀體制", "🔍 買點掃描", "💼 持倉風控",
        f"⚠️ 警報與槓桿{' (' + str(n_danger) + '🔴)' if n_danger else ''}", "🧮 風控計算機", "📉 圖表",
    ])
    with t1:
        render_macro(res)
    with t2:
        render_scan(res)
    with t3:
        render_portfolio(res)
    with t4:
        render_alerts(res)
    with t5:
        render_calculator(account, risk_pct)
    with t6:
        render_chart(res)


main()
