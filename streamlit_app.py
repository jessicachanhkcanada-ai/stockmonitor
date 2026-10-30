# -*- coding: utf-8 -*-
"""
==========================================================================
 📈 智慧股票監控與個人財務資產管理 — Streamlit 網頁版
 支援互惠基金、USD/CAD 匯率、房貸與 4 個月開銷低風險基準配置分析
==========================================================================
"""

from __future__ import annotations

import datetime as dt
import hmac
import json
import os
import re
from typing import Optional

import pandas as pd
import requests
import streamlit as st

import stock_monitor as sm

SETTINGS_FILE = "settings.json"
HOLD_COLS = ["symbol", "shares", "avg_cost", "stop", "target", "risk_level", "account_type", "is_long_term"]

st.set_page_config(page_title="智慧股票監控與風控助理", page_icon="📈", layout="wide")


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
        "tg_token": get_secret("TELEGRAM_BOT_TOKEN", ""),
        "tg_chat_id": get_secret("TELEGRAM_CHAT_ID", ""),
        "mortgage_principal": 520000.0,
        "mortgage_rate": 3.70,
        "mortgage_years": 28.0,
        "weekly_payment": 650.0,
        "cash_savings": 68032.0,
        "annual_income_manual": 77766.46,
        "annual_expense_manual": 87626.10,
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


def auto_detect_risk(sym: str) -> str:
    sym = sym.upper()
    if any(k in sym for k in ["SGOV", "CBIL", "CASH", "PSA"]):
        return "低風險 (Low)"
    elif any(k in sym for k in ["XEQT", "CM.TO", "XIU"]):
        return "中風險 (Medium)"
    else:
        return "高風險 (High)"


def holdings_to_df(holdings: list[dict]) -> pd.DataFrame:
    for h in holdings:
        if "risk_level" not in h or not h["risk_level"]:
            h["risk_level"] = auto_detect_risk(h.get("symbol", ""))
        if "account_type" not in h or not h["account_type"]:
            h["account_type"] = "TFSA"
            
    df = pd.DataFrame(holdings)
    for col in HOLD_COLS:
        if col not in df.columns:
            if col == "is_long_term":
                df[col] = False
            elif col in ["shares", "avg_cost", "stop", "target"]:
                df[col] = 0.0
            elif col == "risk_level":
                df[col] = "高風險 (High)"
            elif col == "account_type":
                df[col] = "TFSA"
            else:
                df[col] = ""
    df = df[HOLD_COLS]
    df["is_long_term"] = df["is_long_term"].fillna(False).astype(bool)
    return df.astype({
        "symbol": "object", "shares": "float64", "avg_cost": "float64",
        "stop": "float64", "target": "float64", "risk_level": "object",
        "account_type": "object", "is_long_term": "bool"
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
            risk_level = str(r.get("risk_level", "高風險 (High)"))
            account_type = str(r.get("account_type", "TFSA"))
            is_lt = bool(r.get("is_long_term", False)) if pd.notna(r.get("is_long_term")) else False
        except (TypeError, ValueError):
            continue
            
        if pd.isna(shares) or pd.isna(cost) or shares <= 0 or cost <= 0:
            continue
            
        out.append({
            "symbol": sym, "shares": shares, "avg_cost": cost, "stop": stop, 
            "target": target, "risk_level": risk_level, "account_type": account_type, "is_long_term": is_lt
        })
    return out


def apply_to_engine(cfg: dict) -> None:
    sm.WATCHLIST = cfg["watchlist"]
    sm.HOLDINGS = cfg["holdings"]
    sm.ACCOUNT_SIZE = cfg["account"]
    sm.MAX_RISK_PCT = cfg["risk_pct"] / 100
    sm.MAX_POSITION_PCT = cfg["max_pos_pct"] / 100
    sm.MIN_RRR = cfg["min_rrr"]


def main() -> None:
    require_password()
    cfg = load_settings()

    with st.sidebar:
        st.title("⚙️ 設定")

        with st.expander("📋 自選股", expanded=False):
            wl_text = st.text_area("代號（用逗號或換行分隔）", value=", ".join(cfg["watchlist"]), height=90)

        with st.expander("💼 目前持倉（含互惠基金、風險與帳戶）", expanded=True):
            st.caption("💡 提示：互惠基金代號可自訂（如 MF_1），股數填 1，平均成本填總市值金額。")
            hold_df = st.data_editor(
                holdings_to_df(cfg["holdings"]), num_rows="dynamic", key="holdings_editor",
                column_config={
                    "symbol": st.column_config.TextColumn("代號 / 基金名稱"),
                    "shares": st.column_config.NumberColumn("股數 / 單位", min_value=0.0, step=0.0001, format="%.4f"),
                    "avg_cost": st.column_config.NumberColumn("平均成本 / 總市值", min_value=0.0, format="%.2f"),
                    "stop": st.column_config.NumberColumn("停損價(留空自動算)", min_value=0.0, format="%.2f"),
                    "target": st.column_config.NumberColumn("目標價(留空自動算)", min_value=0.0, format="%.2f"),
                    "risk_level": st.column_config.SelectboxColumn(
                        "風險等級", options=["低風險 (Low)", "中風險 (Medium)", "高風險 (High)"], required=True
                    ),
                    "account_type": st.column_config.SelectboxColumn(
                        "帳戶類型", options=["TFSA", "RRSP", "Non-Registered"], required=True
                    ),
                    "is_long_term": st.column_config.CheckboxColumn("長線", default=False),
                }
            )

        with st.expander("💰 資金與風控", expanded=False):
            account = st.number_input("總資金規模", min_value=0.0, value=float(cfg["account"]), step=1000.0)
            risk_pct = st.number_input("單筆最大容忍虧損 (%)", min_value=0.1, max_value=20.0, value=float(cfg["risk_pct"]), step=0.1)
            max_pos = st.number_input("單一標的最大倉位 (%)", min_value=1.0, max_value=100.0, value=float(cfg["max_pos_pct"]), step=1.0)
            min_rrr = st.number_input("最低風險報酬比 RRR", min_value=1.0, max_value=10.0, value=float(cfg["min_rrr"]), step=0.5)

        if st.button("💾 儲存側邊欄設定"):
            to_save = cfg.copy()
            to_save.update({
                "watchlist": parse_watchlist(wl_text), "holdings": clean_holdings(hold_df),
                "account": account, "risk_pct": risk_pct, "max_pos_pct": max_pos, "min_rrr": min_rrr,
            })
            err = save_settings(to_save)
            if err is None:
                st.success("已成功儲存設定！")
                st.rerun()
            else:
                st.error(f"儲存失敗：{err}")

    live_cfg = load_settings()
    live_cfg.update({
        "watchlist": parse_watchlist(wl_text), "holdings": clean_holdings(hold_df),
        "account": account, "risk_pct": risk_pct, "max_pos_pct": max_pos, "min_rrr": min_rrr,
    })
    apply_to_engine(live_cfg)

    fx_current = sm.get_usdcad_rate()
    st.title("📈 智慧股票監控與個人財務資產管理")

    try:
        with st.spinner("正在分析資產與持倉…"):
            res = sm.run_analysis(account, risk_pct / 100)
    except Exception as exc:
        st.error(f"分析失敗：{exc}")
        st.stop()

    t1, t2, t3, t4, t5, t6 = st.tabs([
        "📊 宏觀體制", "🔍 買點掃描", "💼 持倉風控", "⚠️ 警報", "📈 互動圖表", "🏠 房貸與財務管理"
    ])
    
    with t3:
        st.subheader("💼 持倉綜合損益與風控（已自動換算為 CAD 合併）")
        rows = res.holdings
        if rows:
            tv = sum(r.market_value for r in rows)
            tc = sum(r.cost_value for r in rows)
            pnl = tv - tc
            
            c1, c2, c3 = st.columns(3)
            c1.metric("總成本 (CAD)", f"${tc:,.2f}")
            c2.metric("總市值 (CAD)", f"${tv:,.2f}")
            c3.metric("未