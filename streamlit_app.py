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

import stock_monitor_3 as sm

SETTINGS_FILE = "settings.json"
CACHE_TTL_SECONDS = 15 * 60   # 價格資料超過 15 分鐘自動重新抓取
HOLD_COLS = ["symbol", "shares", "avg_cost", "stop", "target", "is_long_term"]
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
    """預設設定：取自 stock_monitor_3.py 的 CONFIG 區。"""
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
    """若設定了 APP_PASSWORD，必須先通過密碼才能看到畫面。"""
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
    df = pd.DataFrame(holdings)
    # 確保所有必要欄位都存在
    for col in HOLD_COLS:
        if col not in df.columns:
            if col == "is_long_term":
                df[col] = False
            elif col in ["shares", "avg_cost", "stop", "target"]:
                df[col] = 0.0
            else:
                df[col] = ""
    df = df[HOLD_COLS]
    df["is_long_term"] = df["is_long_term"].fillna(False).astype(bool)
    return df.astype({
        "symbol": "object", "shares": "float64", "avg_cost": "float64",
        "stop": "float64", "target": "float64", "is_long_term": "bool"
    })


def clean_holdings(df: pd.DataFrame) -> list[dict]:
    """把表格編輯器的內容整理成引擎使用的持倉清單；支援長線投資標籤。"""
    out = []
    for _, r in df.iterrows():
        sym = str(r["symbol"]).strip().upper() if pd.notna(r["symbol"]) else ""
        if not sym or sym == "NONE" or sym == "NAN":
            continue
        try:
            shares = float(r["shares"])
            cost = float(r["avg_cost"])
            stop = float(r["stop"]) if pd.notna(r["stop"]) and str(r["stop"]).upper() != "NONE" else 0.0
            target = float(r["target"]) if pd.notna(r["target"]) and str(r["target"]).upper() != "NONE" else 0.0
            is_lt = bool(r.get("is_long_term", False)) if pd.notna(r.get("is_long_term")) else False
        except (TypeError, ValueError):
            continue
            
        if pd.isna(shares) or pd.isna(cost) or shares <= 0 or cost <= 0:
            continue
            
        out.append({
            "symbol": sym, 
            "shares": shares, 
            "avg_cost": cost, 
            "stop": stop, 
            "target": target, 
            "is_long_term": is_lt
        })
    return out


def apply_to_engine(cfg: dict) -> None:
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
# 版面渲染
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
        st.error("🚨 熊市極度恐慌低接機會：" + " ； ".join(m.panic_hits) + "。")


def render_sizing(sz: sm.Sizing) -> None:
    d1, d2, d3, d4 = st.columns(4)
    d1.metric("建議股數", f"{sz.shares:,}")
    d2.metric("下單金額", f"{sz.order_value:,.2f}")
    d3.metric("最大虧損", f"{sz.risk_amount:,.2f}")
    d4.metric("停損距離", f"{sz.stop_pct:.2%}")
    for n in sz.notes:
        st.warning(n)


def render_scan(res: sm.AnalysisResult) -> None:
    hits = [s for s in res.setups if s.valid]
    st.markdown(f"**共 {len(res.setups)} 檔，符合買入條件 {len(hits)} 檔**（RRR ≥ {sm.MIN_RRR:g}）")
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
    } for r in rows])
    st.dataframe(df, hide_index=True)


def render_alerts(res: sm.AnalysisResult) -> None:
    if not res.alerts:
        st.success("目前沒有警報 ✅")
        return
    for a in sorted(res.alerts, key=lambda a: LEVEL_ORDER.get(a.level, 9)):
        if a.level == "danger":
            st.error(a.text)
        elif a.level == "warn":
            st.warning(a.text)
        else:
            st.success(a.text)


def render_calculator(default_account: float, default_risk_pct: float) -> None:
    st.markdown("手動輸入任何一筆交易，計算建議股數與風險。")
    c1, c2, c3 = st.columns(3)
    entry = c1.number_input("買入價", min_value=0.0, value=100.0, step=0.5, key="calc_entry")
    stop = c2.number_input("停損價", min_value=0.0, value=95.0, step=0.5, key="calc_stop")
    target = c3.number_input("目標價（可填 0 略過）", min_value=0.0, value=115.0, step=0.5, key="calc_target")
    d1, d2, d3 = st.columns(3)
    account = d1.number_input("總資金規模", min_value=0.0, value=float(default_account), step=1000.0, key="calc_account")
    risk_pct = d2.number_input("單筆最大容忍虧損 (%)", min_value=0.1, max_value=20.0, value=float(default_risk_pct), step=0.1, key="calc_risk")
    leveraged = d3.checkbox("這是槓桿 ETF", key="calc_lev")

    if entry <= 0 or stop <= 0 or stop >= entry:
        st.warning("請輸入有效的價格：停損價必須低於買入價。")
        return
    sz = sm.calc_position(entry, stop, target if target > entry else None, account, risk_pct / 100, leveraged)
    render_sizing(sz)


def render_chart(res: sm.AnalysisResult) -> None:
    symbols = list(dict.fromkeys(sm.WATCHLIST + [h["symbol"] for h in sm.HOLDINGS] + sm.BENCHMARKS))
    sym = st.selectbox("選擇標的", symbols)
    df = sm.get_history(sym)
    if df.empty:
        st.warning("無法取得此標的資料。")
        return
    chart_df = pd.DataFrame({"收市價": df["Close"], "20MA": sm.sma(df["Close"], 20)}).tail(150)
    st.line_chart(chart_df)


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
            st.caption("勾選「長線投資」的標的（如 RRSP/TFSA 內長線資產）將不會觸發短線停損警報。")
            hold_df = st.data_editor(
                holdings_to_df(cfg["holdings"]), num_rows="dynamic", key="holdings_editor",
                column_config={
                    "symbol": st.column_config.TextColumn("代號"),
                    "shares": st.column_config.NumberColumn("股數", min_value=0.0, step=1.0),
                    "avg_cost": st.column_config.NumberColumn("平均成本", min_value=0.0, format="%.2f"),
                    "stop": st.column_config.NumberColumn("停損/支撐(留空自動算)", min_value=0.0, format="%.2f"),
                    "target": st.column_config.NumberColumn("目標價(留空自動算)", min_value=0.0, format="%.2f"),
                    "is_long_term": st.column_config.CheckboxColumn("長線投資 (略過短線停損)", default=False),
                }
            )

        with st.expander("💰 資金與風控", expanded=True):
            account = st.number_input("總資金規模", min_value=0.0, value=float(cfg["account"]), step=1000.0)
            risk_pct = st.number_input("單筆最大容忍虧損 (%)", min_value=0.1, max_value=20.0, value=float(cfg["risk_pct"]), step=0.1)
            max_pos = st.number_input("單一標的最大倉位 (%)", min_value=1.0, max_value=100.0, value=float(cfg["max_pos_pct"]), step=1.0)
            min_rrr = st.number_input("最低風險報酬比 RRR", min_value=1.0, max_value=10.0, value=float(cfg["min_rrr"]), step=0.5)

        with st.expander("📨 Telegram 推播", expanded=True):
            token = st.text_input("Bot Token", value=get_secret("TELEGRAM_BOT_TOKEN"), type="password")
            chat_id = st.text_input("Chat ID", value=get_secret("TELEGRAM_CHAT_ID"))
            auto_push = st.checkbox("每次分析後自動推送新警報", value=False)
            if st.button("📨 發送測試通知", type="primary"):
                with st.spinner("傳送中…"):
                    ok, msg = tg_send(token, chat_id, f"✅ 測試成功！智慧股票監控已連線。\n{dt.datetime.now():%Y-%m-%d %H:%M:%S}")
                (st.success if ok else st.error)(msg)

        if st.button("💾 儲存設定"):
            to_save = {
                "watchlist": parse_watchlist(wl_text), 
                "holdings": clean_holdings(hold_df),
                "account": account, "risk_pct": risk_pct, "max_pos_pct": max_pos, "min_rrr": min_rrr,
            }
            err = save_settings(to_save)
            if err is None:
                st.success(f"已成功儲存設定！")
                st.rerun()
            else:
                st.error(f"儲存失敗：{err}")

    # 套用設定到引擎
    live_cfg = {
        "watchlist": parse_watchlist(wl_text), 
        "holdings": clean_holdings(hold_df),
        "account": account, "risk_pct": risk_pct, "max_pos_pct": max_pos, "min_rrr": min_rrr,
        "vix_panic": cfg["vix_panic"], "fg_panic": cfg["fg_panic"], "rsi_oversold": cfg["rsi_oversold"],
        "support_tol_pct": cfg["support_tol_pct"], "lev_vol_warn_pct": cfg["lev_vol_warn_pct"],
        "fg_manual_on": cfg["fg_manual_on"], "fg_manual": cfg["fg_manual"],
    }
    apply_to_engine(live_cfg)

    # ---------------- 主畫面與分析 ----------------
    st.title("📈 智慧股票監控與風控助理")
    st.caption("以日線收市價為基礎的趨勢與風控提示，支援 RRSP 長線資產過濾與自動補全。")

    try:
        with st.spinner("正在分析大盤、觀察清單與持倉…"):
            res = sm.run_analysis(account, risk_pct / 100)
    except Exception as exc:
        st.error(f"分析失敗：{exc}")
        st.stop()

    n_danger = sum(1 for a in res.alerts if a.level == "danger")
    t1, t2, t3, t4, t5, t6 = st.tabs([
        "📊 宏觀體制", "🔍 買點掃描", "💼 持倉風控",
        f"⚠️ 警報與槓桿{' (' + str(n_danger) + '🔴)' if n_danger else ''}", "🧮 風控計算機", "📉 圖表",
    ])
    with t1: render_macro(res)
    with t2: render_scan(res)
    with t3: render_portfolio(res)
    with t4: render_alerts(res)
    with t5: render_calculator(account, risk_pct)
    with t6: render_chart(res)


main()