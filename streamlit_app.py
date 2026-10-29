# -*- coding: utf-8 -*-
"""
==========================================================================
 📈 智慧股票監控與風控助理 — Streamlit 網頁版
 支援 Telegram 設定、買點原因解析、圖表顯示與手動持倉管理
==========================================================================
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
CACHE_TTL_SECONDS = 15 * 60
HOLD_COLS = ["symbol", "shares", "avg_cost", "stop", "target", "is_long_term"]
LEVEL_ORDER = {"danger": 0, "warn": 1, "opportunity": 2, "info": 3}

st.set_page_config(page_title="智慧股票監控與風控助理", page_icon="📈", layout="wide")


# ==========================================================================
# 工具：機密、設定檔、密碼保護
# ==========================================================================

def get_secret(name: str, default: str = "") -> str:
    val = os.getenv(name)
    if val:
        return val
    try:
        return str(st.secrets[name])
    except Exception:
        return default


def default_settings() -> dict:
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
        "tg_token": get_secret("TELEGRAM_BOT_TOKEN", ""),
        "tg_chat_id": get_secret("TELEGRAM_CHAT_ID", ""),
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


def parse_watchlist(text: str) -> list[str]:
    seen, out = set(), []
    for tok in re.split(r"[,\s;，]+", text.upper()):
        if tok and tok not in seen:
            seen.add(tok)
            out.append(tok)
    return out


def holdings_to_df(holdings: list[dict]) -> pd.DataFrame:
    df = pd.DataFrame(holdings)
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
    
    if cfg.get("tg_token"):
        os.environ["TELEGRAM_BOT_TOKEN"] = cfg["tg_token"]
    if cfg.get("tg_chat_id"):
        os.environ["TELEGRAM_CHAT_ID"] = str(cfg["tg_chat_id"])


# ==========================================================================
# 主程式 (Main)
# ==========================================================================

def main() -> None:
    require_password()
    cfg = load_settings()

    # ---------------- 側邊欄 ----------------
    with st.sidebar:
        st.title("⚙️ 設定")

        with st.expander("📋 自選股", expanded=False):
            wl_text = st.text_area("代號（用逗號或換行分隔）", value=", ".join(cfg["watchlist"]), height=90)

        # 📲 Telegram 設定區
        with st.expander("📲 Telegram 警報推播設定", expanded=True):
            st.caption("填入 Bot Token 與 Chat ID 以接收推播。")
            tg_token = st.text_input("Telegram Bot Token", value=cfg.get("tg_token", ""), type="password")
            tg_chat_id = st.text_input("Telegram Chat ID", value=str(cfg.get("tg_chat_id", "")))
            
            if st.button("🧪 測試發送 Telegram 訊息"):
                if tg_token and tg_chat_id:
                    os.environ["TELEGRAM_BOT_TOKEN"] = tg_token
                    os.environ["TELEGRAM_CHAT_ID"] = tg_chat_id
                    sm.send_telegram_alert("🔔 【測試訊息】Telegram 推播設定成功！")
                    st.success("已發送測試訊息，請檢查 Telegram！")
                else:
                    st.warning("請先填寫 Bot Token 與 Chat ID。")

        # 💼 持倉表格
        with st.expander("💼 目前持倉（手動微調/預覽）", expanded=True):
            hold_df = st.data_editor(
                holdings_to_df(cfg["holdings"]), num_rows="dynamic", key="holdings_editor",
                column_config={
                    "symbol": st.column_config.TextColumn("代號"),
                    "shares": st.column_config.NumberColumn("股數", min_value=0.0, step=0.0001, format="%.4f"),
                    "avg_cost": st.column_config.NumberColumn("平均成本", min_value=0.0, format="%.2f"),
                    "stop": st.column_config.NumberColumn("停損/支撐(留空自動算)", min_value=0.0, format="%.2f"),
                    "target": st.column_config.NumberColumn("目標價(留空自動算)", min_value=0.0, format="%.2f"),
                    "is_long_term": st.column_config.CheckboxColumn("長線投資", default=False),
                }
            )

        with st.expander("💰 資金與風控", expanded=False):
            account = st.number_input("總資金規模", min_value=0.0, value=float(cfg["account"]), step=1000.0)
            risk_pct = st.number_input("單筆最大容忍虧損 (%)", min_value=0.1, max_value=20.0, value=float(cfg["risk_pct"]), step=0.1)
            max_pos = st.number_input("單一標的最大倉位 (%)", min_value=1.0, max_value=100.0, value=float(cfg["max_pos_pct"]), step=1.0)
            min_rrr = st.number_input("最低風險報酬比 RRR", min_value=1.0, max_value=10.0, value=float(cfg["min_rrr"]), step=0.5)

        if st.button("💾 儲存設定"):
            to_save = {
                "watchlist": parse_watchlist(wl_text), 
                "holdings": clean_holdings(hold_df),
                "account": account, "risk_pct": risk_pct, "max_pos_pct": max_pos, "min_rrr": min_rrr,
                "tg_token": tg_token, "tg_chat_id": tg_chat_id,
            }
            err = save_settings(to_save)
            if err is None:
                st.success("已成功儲存設定！")
                st.rerun()
            else:
                st.error(f"儲存失敗：{err}")

    live_cfg = {
        "watchlist": parse_watchlist(wl_text), 
        "holdings": clean_holdings(hold_df),
        "account": account, "risk_pct": risk_pct, "max_pos_pct": max_pos, "min_rrr": min_rrr,
        "vix_panic": cfg["vix_panic"], "fg_panic": cfg["fg_panic"], "rsi_oversold": cfg["rsi_oversold"],
        "support_tol_pct": cfg["support_tol_pct"], "lev_vol_warn_pct": cfg["lev_vol_warn_pct"],
        "fg_manual_on": cfg["fg_manual_on"], "fg_manual": cfg["fg_manual"],
        "tg_token": tg_token, "tg_chat_id": tg_chat_id,
    }
    apply_to_engine(live_cfg)

    # ---------------- 主畫面與分析 ----------------
    st.title("📈 智慧股票監控與風控助理")
    st.caption("以日線收市價為基礎的趨勢與風控提示，支援 Telegram 自動推播與技術圖表。")

    try:
        with st.spinner("正在分析大盤、觀察清單與持倉…"):
            res = sm.run_analysis(account, risk_pct / 100)
    except Exception as exc:
        st.error(f"分析失敗：{exc}")
        st.stop()

    n_danger = sum(1 for a in res.alerts if a.level == "danger")
    t1, t2, t3, t4, t5 = st.tabs([
        "📊 宏觀體制", "🔍 買點掃描", "💼 持倉風控",
        f"⚠️ 警報與槓桿{' (' + str(n_danger) + '🔴)' if n_danger else ''}",
        "📈 互動圖表"
    ])
    
    with t1:
        st.subheader("大盤與市場體制")
        m = res.macro
        st.info(f"當前大盤體制：{m.regime}")
        
    with t2:
        st.subheader("觀察清單買點掃描")
        hits = [s for s in res.setups if s.valid]
        st.write(f"符合條件標的數：{len(hits)}")
        
        for setup in res.setups:
            with st.container():
                icon = "✅" if setup.valid else "⏳"
                rrr_val = setup.rrr if setup.rrr is not None else 0.0
                expander_label = f"{icon} {setup.symbol} - 現價: {setup.price:.2f} | RRR: {rrr_val:.2f}"
                with st.expander(expander_label, expanded=setup.valid):
                    col1, col2, col3, col4 = st.columns(4)
                    col1.metric("觸發條件", setup.trigger if setup.trigger else "無")
                    col2.metric("建議停損價", f"{setup.stop:.2f}" if setup.stop else "N/A")
                    col3.metric("目標價", f"{setup.target:.2f}" if setup.target else "N/A")
                    col4.metric("RSI (14)", f"{setup.rsi:.1f}")
                    
                    st.markdown(f"**💡 判定原因 / 說明**：{setup.reason}")
                    
                    if setup.valid and setup.sizing:
                        st.success(f"📌 **風控建議**：建議買入股數 `{setup.sizing.shares}` 股 | 下單金額 `$${setup.sizing.order_value:,.2f}` | 最大風險 `$${setup.sizing.risk_amount:,.2f}`")
        
    with t3:
        st.subheader("持倉綜合損益與風控（已自動跨 App 合併）")
        rows = res.holdings
        if rows:
            tv = sum(r.market_value for r in rows)
            tc = sum(r.cost_value for r in rows)
            pnl = tv - tc
            
            c1, c2, c3 = st.columns(3)
            c1.metric("總成本", f"${tc:,.2f}")
            c2.metric("總市值", f"${tv:,.2f}")
            c3.metric("未實現損益", f"${pnl:+,.2f}", f"{(tv / tc - 1) * 100:+.2f}%" if tc else None)
            
            df_disp = pd.DataFrame([{
                "代號": r.symbol,
                "總股數": r.shares,
                "加權平均成本": r.avg_cost,
                "現價": r.price,
                "總市值": r.market_value,
                "未實現損益": r.pnl,
                "損益%": r.pnl_pct,
                "自動停損價": r.stop,
                "自動目標價": r.target,
                "長線資產": "是" if r.is_long_term else "否"
            } for r in rows])
            
            st.dataframe(df_disp, hide_index=True)
        else:
            st.info("尚無持倉資料，請從左側欄手動輸入。")

    with t4:
        st.subheader("風險與加碼警報")
        if res.alerts:
            for a in res.alerts:
                if a.level == "danger":
                    st.error(a.text)
                elif a.level == "warn":
                    st.warning(a.text)
                else:
                    st.info(a.text)
        else:
            st.success("目前無短線警報 ✅")

    with t5:
        st.subheader("📈 標的走勢與均線圖表")
        selected_sym = st.selectbox("選擇要檢視的標的", sm.WATCHLIST + [h.symbol for h in res.holdings])
        if selected_sym:
            df_chart = sm.get_history(selected_sym, period="6mo")
            if not df_chart.empty:
                df_chart["SMA20"] = df_chart["Close"].rolling(20).mean()
                df_chart["SMA50"] = df_chart["Close"].rolling(50).mean()
                st.line_chart(df_chart[["Close", "SMA20", "SMA50"]])
            else:
                st.warning(f"無法取得 {selected_sym} 的圖表資料。")


if __name__ == "__main__":
    main()