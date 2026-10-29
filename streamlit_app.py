# -*- coding: utf-8 -*-
"""
==========================================================================
 📈 智慧股票監控與個人財務資產管理 — Streamlit 網頁版
 支援 Telegram 設定、持倉風險自動分類與手動調整、帳戶類型標記 (TFSA/RRSP)、
 美加匯率自動換算、房貸動態模擬與淨資產追蹤。
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
HOLD_COLS = ["symbol", "shares", "avg_cost", "stop", "target", "risk_level", "account_type", "is_long_term"]
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
        # 房貸與手動年度收支預設值
        "mortgage_principal": 530000.0,
        "mortgage_rate": 3.45,
        "mortgage_years": 23.0,
        "weekly_payment": 650.0,
        "cash_savings": 50000.0,
        "annual_income_manual": 97600.0,
        "annual_expense_manual": 92860.0,
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
            "symbol": sym, 
            "shares": shares, 
            "avg_cost": cost, 
            "stop": stop, 
            "target": target, 
            "risk_level": risk_level,
            "account_type": account_type,
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

        with st.expander("📲 Telegram 警報推播設定", expanded=False):
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

        with st.expander("💼 目前持倉（含風險與帳戶分類）", expanded=True):
            hold_df = st.data_editor(
                holdings_to_df(cfg["holdings"]), num_rows="dynamic", key="holdings_editor",
                column_config={
                    "symbol": st.column_config.TextColumn("代號"),
                    "shares": st.column_config.NumberColumn("股數", min_value=0.0, step=0.0001, format="%.4f"),
                    "avg_cost": st.column_config.NumberColumn("平均成本", min_value=0.0, format="%.2f"),
                    "stop": st.column_config.NumberColumn("停損價(留空自動算)", min_value=0.0, format="%.2f"),
                    "target": st.column_config.NumberColumn("目標價(留空自動算)", min_value=0.0, format="%.2f"),
                    "risk_level": st.column_config.SelectboxColumn(
                        "風險等級", 
                        options=["低風險 (Low)", "中風險 (Medium)", "高風險 (High)"],
                        required=True
                    ),
                    "account_type": st.column_config.SelectboxColumn(
                        "帳戶類型", 
                        options=["TFSA", "RRSP", "Non-Registered"],
                        required=True
                    ),
                    "is_long_term": st.column_config.CheckboxColumn("長線", default=False),
                }
            )

        with st.expander("💰 資金與風控", expanded=False):
            account = st.number_input("總資金規模", min_value=0.0, value=float(cfg["account"]), step=1000.0)
            risk_pct = st.number_input("單筆最大容忍虧損 (%)", min_value=0.1, max_value=20.0, value=float(cfg["risk_pct"]), step=0.1)
            max_pos = st.number_input("單一標的最大倉位 (%)", min_value=1.0, max_value=100.0, value=float(cfg["max_pos_pct"]), step=1.0)
            min_rrr = st.number_input("最低風險報酬比 RRR", min_value=1.0, max_value=10.0, value=float(cfg["min_rrr"]), step=0.5)

        if st.button("💾 儲存設定"):
            to_save = cfg.copy()
            to_save.update({
                "watchlist": parse_watchlist(wl_text), 
                "holdings": clean_holdings(hold_df),
                "account": account, "risk_pct": risk_pct, "max_pos_pct": max_pos, "min_rrr": min_rrr,
                "tg_token": tg_token, "tg_chat_id": tg_chat_id,
            })
            err = save_settings(to_save)
            if err is None:
                st.success("已成功儲存設定！")
                st.rerun()
            else:
                st.error(f"儲存失敗：{err}")

    live_cfg = load_settings()
    live_cfg.update({
        "watchlist": parse_watchlist(wl_text), 
        "holdings": clean_holdings(hold_df),
        "account": account, "risk_pct": risk_pct, "max_pos_pct": max_pos, "min_rrr": min_rrr,
        "tg_token": tg_token, "tg_chat_id": tg_chat_id,
    })
    apply_to_engine(live_cfg)

    # ---------------- 主畫面與分析 ----------------
    fx_current = sm.get_usdcad_rate()
    st.title("📈 智慧股票監控與個人財務資產管理")
    st.caption(f"支援 Telegram 背景排程推播、技術分析、持倉風控，以及美加匯率自動換算 (當前 USD/CAD 匯率: `{fx_current:.4f}` 統一以 CAD 結算)。")

    try:
        with st.spinner("正在分析大盤、觀察清單與持倉…"):
            res = sm.run_analysis(account, risk_pct / 100)
    except Exception as exc:
        st.error(f"分析失敗：{exc}")
        st.stop()

    n_danger = sum(1 for a in res.alerts if a.level == "danger")
    t1, t2, t3, t4, t5, t6 = st.tabs([
        "📊 宏觀體制", "🔍 買點掃描", "💼 持倉風控",
        f"⚠️ 警報與槓桿{' (' + str(n_danger) + '🔴)' if n_danger else ''}",
        "📈 互動圖表", "🏠 房貸與財務管理"
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
                expander_label = f"{icon} {setup.symbol} - 現價: {setup.price:.2f} CAD | RRR: {rrr_val:.2f}"
                with st.expander(expander_label, expanded=setup.valid):
                    col1, col2, col3, col4 = st.columns(4)
                    col1.metric("觸發條件", setup.trigger if setup.trigger else "無")
                    col2.metric("建議停損價", f"{setup.stop:.2f} CAD" if setup.stop else "N/A")
                    col3.metric("目標價", f"{setup.target:.2f} CAD" if setup.target else "N/A")
                    col4.metric("RSI (14)", f"{setup.rsi:.1f}")
                    
                    st.markdown(f"**💡 判定原因 / 說明**：{setup.reason}")
                    
                    if setup.valid and setup.sizing:
                        st.success(f"📌 **風控建議**：建議買入股數 `{setup.sizing.shares}` 股 | 下單金額 `$${setup.sizing.order_value:,.2f} CAD` | 最大風險 `$${setup.sizing.risk_amount:,.2f} CAD`")
        
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
            c3.metric("未實現損益 (CAD)", f"${pnl:+,.2f}", f"{(tv / tc - 1) * 100:+.2f}%" if tc else None)
            
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
                "風險等級": r.risk_level,
                "帳戶類型": r.account_type,
                "長線資產": "是" if r.is_long_term else "否"
            } for r in rows])
            
            st.dataframe(df_disp, hide_index=True)
            
            st.markdown("---")
            st.subheader("📊 持倉結構與權重分析 (Allocation & Concentration)")

            df_alloc = pd.DataFrame([{
                "Symbol": r.symbol,
                "MarketValue": r.market_value,
            } for r in rows])
            
            total_mv = df_alloc["MarketValue"].sum()
            if total_mv > 0:
                df_alloc["Weight"] = df_alloc["MarketValue"] / total_mv * 100
                
                col_chart, col_health = st.columns([1.3, 0.7])
                
                with col_chart:
                    import altair as alt
                    chart = alt.Chart(df_alloc).mark_arc(innerRadius=60).encode(
                        theta=alt.Theta(field="MarketValue", type="quantitative"),
                        color=alt.Color(field="Symbol", type="nominal", legend=alt.Legend(title="標的代號")),
                        tooltip=[
                            "Symbol", 
                            alt.Tooltip("MarketValue", format="$,.2f", title="市值 (CAD)"), 
                            alt.Tooltip("Weight", format=".1f%", title="佔比")
                        ]
                    ).properties(title="資產配置權重佔比 (Market Value in CAD)", height=320)
                    st.altair_chart(chart, use_container_width=True)
                    
                with col_health:
                    st.markdown("### 🔍 集中度診斷")
                    max_row = df_alloc.loc[df_alloc["Weight"].idxmax()]
                    max_sym = max_row["Symbol"]
                    max_weight = max_row["Weight"]
                    
                    if max_weight > 40:
                        st.warning(f"⚠️ **重壓警示**：`{max_sym}` 佔比高達 **{max_weight:.1f}%**，單一資產集中風險偏高，建議留意產業或市場波動。")
                    elif max_weight > 25:
                        st.info(f"ℹ️ **核心持倉**：`{max_sym}` 為目前第一大重倉，佔比 **{max_weight:.1f}%**，配置符合主流核心資產特性。")
                    else:
                        st.success("🟢 **配置均衡**：投資組合分散良好，無單一標的過度集中。")
                        
                    small_ones = df_alloc[df_alloc["Weight"] < 3]["Symbol"].tolist()
                    if small_ones:
                        st.caption(f"💡 佔比低於 3% 的標的：`{', '.join(small_ones)}`（個別漲跌對整體資產影響有限）。")
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

    with t6:
        st.subheader("🏠 房貸追蹤與年度收支手動填寫")
        st.caption("您可以隨時在此更新年度總收入與總支出，系統會自動換算每月平均、儲蓄率並與美加換算後的股票資產合併計算淨資產。")
        
        st.markdown("### 📊 年度收支快速輸入 (Annual Income & Expense in CAD)")
        col_in_man, col_ex_man, col_save_btn = st.columns([1.5, 1.5, 1])
        
        with col_in_man:
            manual_annual_inc = st.number_input("今年累積總收入 (CAD)", value=float(cfg.get("annual_income_manual", 97600.0)), step=1000.0, key="man_inc")
        with col_ex_man:
            manual_annual_exp = st.number_input("今年累積總支出 (CAD，含房貸)", value=float(cfg.get("annual_expense_manual", 92860.0)), step=1000.0, key="man_exp")
            
        with col_save_btn:
            st.write("")
            if st.button("💾 儲存收支數據"):
                cfg["annual_income_manual"] = manual_annual_inc
                cfg["annual_expense_manual"] = manual_annual_exp
                err = save_settings(cfg)
                if err is None:
                    st.success("已儲存！")
                    st.rerun()
                else:
                    st.error(f"儲存失敗：{err}")

        m_inc = manual_annual_inc / 12.0
        m_exp = manual_annual_exp / 12.0
        m_surplus = m_inc - m_exp
        s_rate = (m_surplus / m_inc * 100) if m_inc > 0 else 0.0
        
        st.markdown("---")
        c_i, c_e, c_s, c_r = st.columns(4)
        c_i.metric("平均每月收入", f"CAD ${m_inc:,.2f}", f"年度: ${manual_annual_inc:,.2f}")
        c_e.metric("平均每月支出", f"CAD ${m_exp:,.2f}", f"年度: ${manual_annual_exp:,.2f}")
        c_s.metric("平均每月可支配結餘", f"CAD ${m_surplus:,.2f}", f"{m_surplus * 12:+,.2f} /年")
        c_r.metric("動態儲蓄率", f"{s_rate:.1f}%")

        st.markdown("---")

        col_m1, col_m2 = st.columns(2)
        
        with col_m1:
            st.markdown("### 📋 房貸動態模擬（利率可調）")
            mortgage_principal = st.number_input("剩餘房貸本金 (CAD)", value=float(cfg.get("mortgage_principal", 530000.0)), step=5000.0, key="m_prin")
            mortgage_rate = st.number_input("目前房貸年利率 (%)", value=float(cfg.get("mortgage_rate", 3.45)), step=0.05, key="m_rate") / 100.0
            mortgage_years = st.number_input("剩餘還款年限 (年)", value=float(cfg.get("mortgage_years", 23.0)), step=1.0, key="m_yrs")
            weekly_payment = st.number_input("每週還款額 (CAD)", value=float(cfg.get("weekly_payment", 650.0)), step=10.0, key="m_wk")
            
            annual_interest_approx = mortgage_principal * mortgage_rate
            weekly_interest_approx = annual_interest_approx / 52
            weekly_principal_approx = weekly_payment - weekly_interest_approx
            
            st.info(
                f"💡 **目前房貸狀況解析**：\n\n"
                f"- 每年產生利息約：`CAD ${annual_interest_approx:,.2f}`\n"
                f"- 每週還款 `${weekly_payment:.0f}` 中：約 `CAD ${weekly_interest_approx:.2f}` 為利息，`CAD ${weekly_principal_approx:.2f}` 為本金。\n"
                f"- 預計約 **{mortgage_years:.1f} 年後**（約 52 歲）清償，比 55 歲目標提早 3 年！✅"
            )

        with col_m2:
            st.markdown("### 💰 全面淨資產總覽 (Net Worth in CAD)")
            total_stock_value = sum(r.market_value for r in res.holdings) if res.holdings else 0.0
            cash_savings = st.number_input("現金與高利活存/緊急預備金 (CAD)", value=float(cfg.get("cash_savings", 50000.0)), step=5000.0, key="c_sav")
            
            total_assets = total_stock_value + cash_savings
            net_worth = total_assets - mortgage_principal
            
            st.metric("總資產 (換算CAD美加股票 + 現金)", f"CAD ${total_assets:,.2f}")
            st.metric("總負債 (房貸本金)", f"CAD ${mortgage_principal:,.2f}", delta="-負債", delta_color="inverse")
            st.metric("預估個人淨資產 (Net Worth)", f"CAD ${net_worth:,.2f}", delta="安全防線")

        st.markdown("---")
        
        st.markdown("### 🎯 資產配置與風險目標對照 (20 / 40 / 40 比例)")
        low_target = total_assets * 0.20
        med_target = total_assets * 0.40
        high_target = total_assets * 0.40
        
        cr1, cr2, cr3 = st.columns(3)
        cr1.metric("低風險目標 (20%)", f"CAD ${low_target:,.2f}", "HISA / 保本 / CASH.TO")
        cr2.metric("中風險目標 (40%)", f"CAD ${med_target:,.2f}", "XEQT.TO / CM.TO 藍籌")
        cr3.metric("高風險目標 (40%)", f"CAD ${high_target:,.2f}", "VOO / QQC.TO 科技增長")


if __name__ == "__main__":
    main()