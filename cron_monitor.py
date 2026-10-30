#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
==========================================================================
 📈 股票與 ETF 24H 背景自動監控與推播腳本
 無需開啟網頁，專供 Cron / GitHub Actions 自動定時排程執行
==========================================================================
"""

from __future__ import annotations

import os
import json
import requests
import yfinance as yf
import stock_monitor as sm

SETTINGS_FILE = "settings.json"


def load_settings() -> dict:
    """讀取設定檔，若不存在則載入預設值"""
    cfg = sm.default_settings() if hasattr(sm, "default_settings") else {
        "watchlist": sm.WATCHLIST,
        "holdings": sm.HOLDINGS,
        "account": sm.ACCOUNT_SIZE,
        "risk_pct": sm.MAX_RISK_PCT * 100,
        "tg_token": os.getenv("TELEGRAM_BOT_TOKEN", ""),
        "tg_chat_id": os.getenv("TELEGRAM_CHAT_ID", "")
    }
    try:
        if os.path.exists(SETTINGS_FILE):
            with open(SETTINGS_FILE, "r", encoding="utf-8") as f:
                cfg.update(json.load(f))
    except Exception as exc:
        print(f"[提示] 讀取 settings.json 失敗或不存在，使用環境變數：{exc}")
    return cfg


def send_telegram_alert(message: str, token: str, chat_id: str) -> None:
    """發送去敏感化的安全 Telegram 通知"""
    if not token or not chat_id:
        print("[警告] 未設定 Telegram Token 或 Chat ID，跳過推播。")
        return
        
    url = f"https://api.telegram.org/bot{token}/sendMessage"
    payload = {
        "chat_id": chat_id,
        "text": message,
        "parse_mode": "Markdown"
    }
    try:
        resp = requests.post(url, json=payload, timeout=10)
        if resp.status_code == 200:
            print(f"[成功] 已成功推播訊息至 Telegram：{message}")
        else:
            print(f"[失敗] Telegram API 回應錯誤：{resp.text}")
    except Exception as exc:
        print(f"[錯誤] 發送 Telegram 警報失敗：{exc}")


def main():
    print("🚀 開始執行背景股票監控與掃描...")
    
    cfg = load_settings()
    
    # 優先從環境變數讀取金鑰 (Secrets)，若無則從設定檔讀取
    token = os.getenv("TELEGRAM_BOT_TOKEN") or cfg.get("tg_token", "")
    chat_id = os.getenv("TELEGRAM_CHAT_ID") or cfg.get("tg_chat_id", "")
    
    # 將設定套用至核心引擎
    sm.WATCHLIST = cfg.get("watchlist", sm.WATCHLIST)
    sm.HOLDINGS = cfg.get("holdings", sm.HOLDINGS)
    account_size = float(cfg.get("account", sm.ACCOUNT_SIZE))
    risk_pct = float(cfg.get("risk_pct", sm.MAX_RISK_PCT * 100)) / 100.0

    # 執行技術面與風控分析
    res = sm.run_analysis(account_size, risk_pct)
    
    # 統計發送數量
    sent_count = 0

    # 依標的合併：同一檔股票的停損警報與買點訊號合成一則訊息
    _, by_sym = sm.group_alerts(res.alerts)
    valid_setups = {s.symbol: s for s in res.setups if s.valid}
    held_syms = {h.symbol for h in res.holdings}

    for sym, items in by_sym.items():
        has_danger = any(a.level == "danger" and not a.is_advice for a in items)
        advice = next((a for a in items if a.is_advice), None)
        advice_actionable = advice is not None and advice.level in ("danger", "warn")
        setup = valid_setups.get(sym)
        if not has_danger and setup is None and not advice_actionable:
            continue  # 只推播停損、買點與需要動作的建議（減碼／停利／出場），其餘僅顯示於網頁

        role = "持倉" if sym in held_syms else "自選"
        lines = [f"📌 *{sym}*（{role}）"]
        for a in items:
            if a.level == "danger" and not a.is_advice:
                lines.append("• " + a.text.replace(f"{sym} ", "", 1))
        if setup is not None:
            lines.append(
                f"• 🟢 買點訊號：現價 `${setup.price:.2f}` CAD，{setup.trigger}\n"
                f"   建議停損 `${setup.stop:.2f}`｜RRR `{setup.rrr:.2f}`"
            )
        if advice is not None and (advice_actionable or sym in held_syms):
            lines.append("• " + advice.text.replace(f"{sym} ", "", 1))
        send_telegram_alert("\n".join(lines), token, chat_id)
        sent_count += 1

    if sent_count == 0:
        print("✅ 監控完成，目前無觸發任何停損或買點警報。")
    else:
        print(f"🎉 監控完成，共發送 {sent_count} 則 Telegram 警報。")


if __name__ == "__main__":
    main()
