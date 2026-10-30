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
import math
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


def fmt_signed(value: float) -> str:
    """格式化正負金額，例如 +$1,234.00 / -$1,234.00"""
    sign = "+" if value >= 0 else "-"
    return f"{sign}${abs(value):,.2f}"


USD_BG = "background-color: rgba(59,130,246,0.22); font-weight: 600;"   # 藍色 = USD
CAD_BG = "background-color: rgba(239,68,68,0.20); font-weight: 600;"    # 紅色 = CAD


def symbol_currency(sym: str) -> str:
    """回傳標的原始交易幣別：.TO / .V 為 CAD，其餘為 USD；無報價的互惠基金視為 CAD(手動)。"""
    sym = sym.upper()
    if sym.endswith(".TO") or sym.endswith(".V"):
        return "CAD"
    if sm.get_history(sym).empty:
        return "CAD(手動)"
    return "USD"


def currency_badge(cur: str) -> str:
    return "🇺🇸 USD" if cur == "USD" else ("🇨🇦 CAD" if cur == "CAD" else "🇨🇦 CAD（手動）")


def native_price_text(cur: str, cad_price: float, fx: float) -> str:
    """把 CAD 價格還原成原幣文字，例如 US$698.12 / C$47.13。"""
    if cad_price is None or (isinstance(cad_price, float) and math.isnan(cad_price)):
        return "—"
    if cur == "USD":
        return f"US${cad_price / fx:,.2f}"
    return f"C${cad_price:,.2f}"


def style_currency_cols(df: pd.DataFrame, cur_series: list[str], cols: list[str], fmt: dict):
    """幣別相關欄位依 USD/CAD 上色；數字統一格式化。"""
    def _tint(col: pd.Series):
        return [USD_BG if c == "USD" else CAD_BG for c in cur_series]
    styler = df.style.apply(_tint, subset=cols, axis=0)
    return styler.format(fmt, na_rep="—")


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
            st.caption("💡 平均成本、停損價、目標價請填該標的「原幣」：USD 標的（如 PTRN、VOO）填美元，.TO 標的填加幣。互惠基金代號可自訂（如 MF_1），股數填 1，平均成本填總市值金額（CAD）。")
            hold_df = st.data_editor(
                holdings_to_df(cfg["holdings"]), num_rows="dynamic", key="holdings_editor",
                column_config={
                    "symbol": st.column_config.TextColumn("代號 / 基金名稱"),
                    "shares": st.column_config.NumberColumn("股數 / 單位", min_value=0.0, step=0.0001, format="%.4f"),
                    "avg_cost": st.column_config.NumberColumn("平均成本（原幣）/ 總市值", min_value=0.0, format="%.2f"),
                    "stop": st.column_config.NumberColumn("停損價（原幣，留空自動算）", min_value=0.0, format="%.2f"),
                    "target": st.column_config.NumberColumn("目標價（原幣，留空自動算）", min_value=0.0, format="%.2f"),
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

        with st.expander("😱 恐懼貪婪指數（手動覆蓋）", expanded=False):
            st.caption("CNN 無法連線時可自行輸入（到 cnn.com/markets/fear-and-greed 查看）。")
            use_manual_fg = st.checkbox("使用手動數值", value=False)
            manual_fg = st.number_input("恐懼貪婪指數 (0–100)", min_value=0, max_value=100, value=50, step=1)

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
    sm.FEAR_GREED_MANUAL = float(manual_fg) if use_manual_fg else None

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

    with t1:
        st.subheader("📊 宏觀市場體制")
        m = res.macro
        a1, a2, a3, a4 = st.columns(4)
        if m.vix is not None:
            vd = m.vix - m.vix_prev if m.vix_prev is not None else None
            a1.metric("VIX 恐慌指數", f"{m.vix:.2f}", f"{vd:+.2f}" if vd is not None else None, delta_color="inverse")
        else:
            a1.metric("VIX 恐慌指數", "N/A")
        a2.metric("恐懼貪婪指數", f"{m.fear_greed:.0f}" if m.fear_greed is not None else "N/A", m.fg_source, delta_color="off")
        a3.metric("恐慌評分", f"{m.panic_score} / 3")
        a4.metric("USD/CAD", f"{fx_current:.4f}")
        st.info(f"**目前市場體制：** {m.regime}")
        if m.panic_alert:
            st.warning("😱 恐慌訊號觸發：" + "；".join(m.panic_hits))
        elif m.panic_hits:
            st.caption("⚠️ 部分恐慌條件：" + "；".join(m.panic_hits))
        if m.indices:
            st.dataframe(pd.DataFrame([{
                "指數/ETF": i.symbol, "收盤價 (USD)": round(i.close, 2), "20日均線": round(i.sma20, 2),
                "站上20日線": "是" if i.above20 else "否", "RSI(14)": round(i.rsi, 1), "狀態": i.label,
            } for i in m.indices]), hide_index=True)
        else:
            st.warning("無法取得指數資料（可能是 yfinance 暫時連線失敗），請稍後重新整理。")

    with t2:
        st.subheader("🔍 自選股買點掃描")
        if not res.setups:
            st.info("自選股清單是空的，請在側邊欄加入代號。")
        else:
            n_valid = sum(1 for s in res.setups if s.valid)
            st.caption(
                f"共掃描 {len(res.setups)} 檔，符合買點 {n_valid} 檔｜"
                f"🇺🇸 藍色 = USD 標的（已按 USD/CAD {fx_current:.4f} 換算）｜🇨🇦 紅色 = CAD 標的｜"
                "標示 (CAD) 的欄位一律為加幣；「現價（原幣）」為該標的實際交易幣別的價格"
            )
            cur_s = [("CAD" if (s.symbol.endswith(".TO") or s.symbol.endswith(".V")) else "USD") for s in res.setups]
            rows_s = []
            for s, c in zip(res.setups, cur_s):
                sz = s.sizing
                rows_s.append({
                    "代號": s.symbol,
                    "原幣別": currency_badge(c),
                    "狀態": "✅ 符合" if s.valid else "⏳ 等待",
                    "現價（原幣）": native_price_text(c, s.price, fx_current),
                    "現價 (CAD)": None if math.isnan(s.price) else s.price,
                    "支撐 (CAD)": s.support,
                    "停損 (CAD)": s.stop,
                    "目標 (CAD)": s.target,
                    "RRR": s.rrr,
                    "RSI": None if math.isnan(s.rsi) else s.rsi,
                    "建議股數": sz.shares if sz else None,
                    "下單金額 (CAD)": None if not sz else sz.order_value,
                    "最大虧損 (CAD)": None if not sz else sz.risk_amount,
                    "說明": s.reason,
                })
            fmt_s = {
                "現價 (CAD)": "{:,.2f}", "支撐 (CAD)": "{:,.2f}", "停損 (CAD)": "{:,.2f}",
                "目標 (CAD)": "{:,.2f}", "RRR": "{:.2f}", "RSI": "{:.1f}", "建議股數": "{:,.0f}",
                "下單金額 (CAD)": "{:,.2f}", "最大虧損 (CAD)": "{:,.2f}",
            }
            sty_s = style_currency_cols(pd.DataFrame(rows_s), cur_s, ["原幣別", "現價（原幣）"], fmt_s)
            st.dataframe(sty_s, hide_index=True)

    with t4:
        st.subheader("⚠️ 警報")
        if not res.alerts:
            st.success("✅ 目前沒有觸發任何警報。")
        else:
            global_alerts, by_sym = sm.group_alerts(res.alerts)
            held_syms = {r.symbol for r in res.holdings}

            plain = [a for a in res.alerts if not a.is_advice]
            advices = [a for a in res.alerts if a.is_advice]
            n_danger = sum(1 for a in plain if a.level == "danger")
            n_warn = sum(1 for a in plain if a.level == "warn")
            n_buy = sum(1 for a in plain if a.level == "success")
            n_add = sum(1 for a in advices if "加碼" in a.action)
            n_cut = sum(1 for a in advices if any(k in a.action for k in ("減碼", "出場", "停利", "檢視")))
            n_hold = len(advices) - n_add - n_cut
            st.caption(
                f"共 {len(by_sym)} 檔標的｜🔴 危險 {n_danger}｜🟠 預警 {n_warn}｜🟢 買點 {n_buy}｜"
                f"🧭 持倉建議：可加碼 {n_add}、減碼／出場／停利 {n_cut}、持有 {n_hold}"
                "（同一標的的警報已合併顯示；建議為規則訊號，非投資建議）"
            )

            for a in global_alerts:
                (st.warning if a.level == "warn" else st.error if a.level == "danger" else st.info)(a.text)

            box = {"danger": st.error, "warn": st.warning, "info": st.info, "success": st.success}
            for sym, items in by_sym.items():
                advice = next((a for a in items if a.is_advice), None)
                # 已有持倉建議時，買點訊息會併入建議內，不重複顯示
                shown = [a for a in items if not a.is_advice and not (advice and a.key.startswith("buy_"))]
                worst = items[0].level  # 已依嚴重度排序（含建議）
                cur = symbol_currency(sym)
                role = "💼 持倉" if sym in held_syms else "👀 自選"
                lines = [f"**{sym}**　{currency_badge(cur)}　·　{role}"]
                for a in shown:
                    lines.append("- " + a.text.replace(f"{sym} ", "", 1))
                if advice is not None:
                    icon = {"danger": "🔴", "warn": "🟠", "success": "🟢"}.get(advice.level, "⚪")
                    body = advice.text.split("】", 1)[-1]
                    lines.append(f"\n**🧭 規則建議：{icon} {advice.action}**\n\n{body}")
                box.get(worst, st.info)("\n".join(lines))

    with t5:
        st.subheader("📈 互動圖表")
        choices = list(dict.fromkeys(list(live_cfg["watchlist"]) + [h["symbol"] for h in live_cfg["holdings"]]))
        if not choices:
            st.info("請先在側邊欄加入自選股或持倉。")
        else:
            cc1, cc2 = st.columns([2, 1])
            sel = cc1.selectbox("選擇標的", choices)
            per = cc2.selectbox("期間", ["3mo", "6mo", "1y", "2y"], index=2)
            hdf = sm.get_history(sel, per)
            if hdf.empty:
                st.warning(f"{sel} 沒有可用的價格資料（互惠基金無法繪圖）。")
            else:
                chart_df = pd.DataFrame({
                    "收盤價": hdf["Close"],
                    "SMA20": sm.sma(hdf["Close"], 20),
                    "SMA50": sm.sma(hdf["Close"], 50),
                })
                st.line_chart(chart_df)
                st.caption(f"{sel} 最新收盤：{float(hdf['Close'].iloc[-1]):,.2f}（原幣別）｜RSI(14)：{float(sm.rsi(hdf['Close']).iloc[-1]):.1f}")

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

            st.caption(
                f"🇺🇸 藍色 = USD 標的（已按 USD/CAD {fx_current:.4f} 換算）｜🇨🇦 紅色 = CAD 標的｜"
                "「平均成本／現價／停損／目標」為該標的原幣（US$ 或 C$）；"
                "標示 (CAD) 的欄位為加幣合併數字"
            )
            advice_map = {a.symbol: a.action for a in res.alerts if a.is_advice}
            cur_list = [r.currency for r in rows]
            native_cols = ["原幣別", "平均成本（原幣）", "現價（原幣）", "停損價（原幣）", "目標價（原幣）"]
            df_disp = pd.DataFrame([{
                "代號": r.symbol,
                "原幣別": currency_badge(r.currency if r.has_live else "CAD(手動)"),
                "總股數": r.shares,
                "平均成本（原幣）": sm.money(r.currency, r.avg_cost_native),
                "現價（原幣）": sm.money(r.currency, r.price_native),
                "停損價（原幣）": sm.money(r.currency, r.stop_native),
                "目標價（原幣）": sm.money(r.currency, r.target_native),
                "損益 %": r.pnl_pct,
                "總成本 (CAD)": r.cost_value,
                "總市值 (CAD)": r.market_value,
                "未實現損益 (CAD)": r.pnl,
                "倉位占比 %": r.weight_pct,
                "規則建議": advice_map.get(r.symbol, "—"),
                "風險等級": r.risk_level,
                "帳戶類型": r.account_type,
                "長線資產": "是" if r.is_long_term else "否",
            } for r in rows])
            fmt_h = {
                "總股數": "{:,.4f}", "損益 %": "{:+.2f}%", "總成本 (CAD)": "{:,.2f}",
                "總市值 (CAD)": "{:,.2f}", "未實現損益 (CAD)": "{:+,.2f}", "倉位占比 %": "{:.1f}%",
            }
            sty = style_currency_cols(df_disp, cur_list, native_cols, fmt_h)
            sty = sty.map(lambda v: "color: #22c55e;" if isinstance(v, (int, float)) and v > 0
                          else ("color: #ef4444;" if isinstance(v, (int, float)) and v < 0 else ""),
                          subset=["未實現損益 (CAD)", "損益 %"])
            st.dataframe(sty, hide_index=True)
            st.caption("🧭「規則建議」依你設定的停損、目標價、倉位上限與買點規則產生，詳細理由請看「⚠️ 警報」分頁；僅供參考，非投資建議。")
        else:
            st.info("目前沒有持倉，請在側邊欄「目前持倉」新增資料後按「儲存側邊欄設定」。")

    with t6:
        st.subheader("🏠 房貸追蹤與年度收支手動填寫")

        col_in_man, col_ex_man, col_save_btn = st.columns([1.5, 1.5, 1])
        with col_in_man:
            manual_annual_inc = st.number_input("今年累積總收入 (CAD)", value=float(cfg.get("annual_income_manual", 77766.46)), step=1000.0, key="man_inc")
        with col_ex_man:
            manual_annual_exp = st.number_input("今年累積總支出 (CAD，含房貸)", value=float(cfg.get("annual_expense_manual", 87626.10)), step=1000.0, key="man_exp")
        with col_save_btn:
            st.write("")
            if st.button("💾 儲存收支數據"):
                cfg["annual_income_manual"] = manual_annual_inc
                cfg["annual_expense_manual"] = manual_annual_exp
                save_settings(cfg)
                st.success("已儲存收支！")
                st.rerun()

        m_inc = manual_annual_inc / 12.0
        m_exp = manual_annual_exp / 12.0
        m_surplus = m_inc - m_exp

        st.markdown("---")
        c_i, c_e, c_s = st.columns(3)
        c_i.metric("平均每月收入", f"CAD ${m_inc:,.2f}")
        c_e.metric("平均每月支出", f"CAD ${m_exp:,.2f}")
        c_s.metric("平均每月可支配結餘", f"CAD ${m_surplus:,.2f}")

        st.markdown("---")

        col_m1, col_m2 = st.columns(2)
        with col_m1:
            st.markdown("### 📋 房貸動態模擬")
            mortgage_principal = st.number_input("剩餘房貸本金 (CAD)", value=float(cfg.get("mortgage_principal", 520000.0)), step=5000.0, key="m_prin")
            mortgage_rate = st.number_input("目前房貸年利率 (%)", value=float(cfg.get("mortgage_rate", 3.70)), step=0.05, key="m_rate")
            mortgage_years = st.number_input("剩餘還款年限 (年)", value=float(cfg.get("mortgage_years", 28.0)), step=1.0, key="m_yrs")
            weekly_payment = st.number_input("每週還款額 (CAD)", value=float(cfg.get("weekly_payment", 650.0)), step=10.0, key="m_wk")

        with col_m2:
            st.markdown("### 💰 全面淨資產總覽 (Net Worth in CAD)")
            total_stock_value = sum(r.market_value for r in res.holdings) if res.holdings else 0.0
            cash_savings = st.number_input("現金與高利活存/緊急預備金 (CAD)", value=float(cfg.get("cash_savings", 68032.0)), step=5000.0, key="c_sav")
            total_assets = total_stock_value + cash_savings
            net_worth = total_assets - mortgage_principal
            st.metric("總資產", f"CAD ${total_assets:,.2f}")
            st.metric("總負債 (房貸)", f"CAD ${mortgage_principal:,.2f}")
            st.metric("預估個人淨資產", f"CAD ${net_worth:,.2f}")

        if st.button("💾 儲存房貸與現金設定", type="primary"):
            cfg["mortgage_principal"] = mortgage_principal
            cfg["mortgage_rate"] = mortgage_rate
            cfg["mortgage_years"] = mortgage_years
            cfg["weekly_payment"] = weekly_payment
            cfg["cash_savings"] = cash_savings
            save_settings(cfg)
            st.success("房貸與現金設定已成功儲存！")
            st.rerun()

        st.markdown("---")

        # --- 動態風險配置比例對照（低風險 = 4個月開銷基準） ---
        st.markdown("### 🎯 風險資產配置對照（低風險 = 4 個月開銷基準）")

        monthly_exp = manual_annual_exp / 12.0
        low_target = monthly_exp * 4.0  # 4個月開銷

        actual_low = sum(r.market_value for r in res.holdings if "低" in r.risk_level or "Low" in r.risk_level) + cash_savings
        actual_med = sum(r.market_value for r in res.holdings if "中" in r.risk_level or "Medium" in r.risk_level)
        actual_high = sum(r.market_value for r in res.holdings if "高" in r.risk_level or "High" in r.risk_level)

        remaining_balance = max(0.0, total_assets - low_target)
        med_target = remaining_balance * 0.50
        high_target = remaining_balance * 0.50

        total_portfolio = total_assets
        low_pct = (actual_low / total_portfolio * 100) if total_portfolio > 0 else 0.0
        med_pct = (actual_med / total_portfolio * 100) if total_portfolio > 0 else 0.0
        high_pct = (actual_high / total_portfolio * 100) if total_portfolio > 0 else 0.0

        diff_low = actual_low - low_target
        diff_med = actual_med - med_target
        diff_high = actual_high - high_target

        # 修正：st.metric 第三個位置參數就是 delta，
        # 原本同時傳入位置參數與 delta= 關鍵字造成 TypeError。
        # 現在只傳一個 delta，目標金額改用 caption 顯示。
        cr1, cr2, cr3 = st.columns(3)
        with cr1:
            st.metric("低風險 (4個月開銷基準)", f"CAD ${actual_low:,.2f} ({low_pct:.1f}%)", delta=f"{fmt_signed(diff_low)} vs 目標")
            st.caption(f"目標：${low_target:,.2f}（4 個月開銷）")
        with cr2:
            st.metric("中風險 (剩餘資產 50%)", f"CAD ${actual_med:,.2f} ({med_pct:.1f}%)", delta=f"{fmt_signed(diff_med)} vs 目標")
            st.caption(f"目標：${med_target:,.2f}")
        with cr3:
            st.metric("高風險 (剩餘資產 50%)", f"CAD ${actual_high:,.2f} ({high_pct:.1f}%)", delta=f"{fmt_signed(diff_high)} vs 目標")
            st.caption(f"目標：${high_target:,.2f}")


if __name__ == "__main__":
    main()
