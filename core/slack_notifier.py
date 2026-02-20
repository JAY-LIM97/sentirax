"""
Sentirax Slack 알림 모듈

환경변수 SLACK_WEBHOOK_URL을 설정하면 모든 매매·재학습 이벤트를 Slack으로 수신.
GitHub Secrets에 SLACK_WEBHOOK_URL 추가 권장.
"""

import os
import json
import requests
from datetime import datetime

# 환경변수 우선, 없으면 빈 문자열 (알림 비활성)
SLACK_WEBHOOK_URL = os.environ.get('SLACK_WEBHOOK_URL', '')


def send_slack_message(text: str) -> bool:
    """기본 Slack 메시지 전송"""
    url = SLACK_WEBHOOK_URL
    if not url:
        return False
    try:
        resp = requests.post(
            url,
            json={"text": text},
            headers={"Content-Type": "application/json"},
            timeout=10,
        )
        return resp.status_code == 200 and resp.text.strip().lower() == "ok"
    except Exception as e:
        print(f"  Slack 전송 실패: {e}")
        return False


def _send_blocks(blocks: list) -> bool:
    """Block Kit 형식 메시지 전송"""
    url = SLACK_WEBHOOK_URL
    if not url:
        return False
    try:
        resp = requests.post(
            url,
            json={"blocks": blocks},
            headers={"Content-Type": "application/json"},
            timeout=10,
        )
        return resp.status_code == 200 and resp.text.strip().lower() == "ok"
    except Exception as e:
        print(f"  Slack 전송 실패: {e}")
        return False


def _mode_tag(paper_trading: bool) -> str:
    return "모의" if paper_trading else "실전"


# ──────────────────────────────────────────────────────────────────────────────
# 매수 진입 알림
# ──────────────────────────────────────────────────────────────────────────────

def notify_trade_open(
    ticker: str,
    price,
    qty: int,
    tp: float,
    sl: float,
    tags: list,
    market: str = "US",
    paper_trading: bool = True,
):
    """매수 주문 체결 알림"""
    now  = datetime.now().strftime('%H:%M:%S')
    mode = _mode_tag(paper_trading)
    tag_str = ' '.join(f'`{t}`' for t in tags) if tags else ''

    if market == "KR":
        price_str = f"{int(price):,}원"
    else:
        price_str = f"${float(price):.2f}"

    lines = [
        f"*[{market}·{mode}] 매수 체결* — {now}",
        f">종목: *{ticker}*  |  가격: {price_str}  |  수량: {qty}주",
        f">TP: +{tp}%  |  SL: -{sl}%",
    ]
    if tag_str:
        lines.append(f">전략: {tag_str}")

    send_slack_message('\n'.join(lines))


# ──────────────────────────────────────────────────────────────────────────────
# 포지션 청산 알림
# ──────────────────────────────────────────────────────────────────────────────

def notify_trade_close(
    ticker: str,
    entry_price,
    exit_price,
    pnl_pct: float,
    reason: str,
    hold_minutes: float = 0,
    market: str = "US",
    paper_trading: bool = True,
):
    """포지션 청산 (TP/SL/TIMEOUT) 알림"""
    now  = datetime.now().strftime('%H:%M:%S')
    mode = _mode_tag(paper_trading)
    sign = "+" if pnl_pct >= 0 else ""
    emoji = "✅" if pnl_pct > 0 else ("⏱" if reason == "TIMEOUT" else "❌")

    if market == "KR":
        entry_str = f"{int(entry_price):,}원"
        exit_str  = f"{int(exit_price):,}원"
    else:
        entry_str = f"${float(entry_price):.2f}"
        exit_str  = f"${float(exit_price):.2f}"

    lines = [
        f"{emoji} *[{market}·{mode}] 포지션 청산* — {now}",
        f">종목: *{ticker}*  |  사유: `{reason}`",
        f">매수: {entry_str}  →  매도: {exit_str}",
        f">*손익: {sign}{pnl_pct:.2f}%*  (보유: {hold_minutes:.0f}분)",
    ]
    send_slack_message('\n'.join(lines))


# ──────────────────────────────────────────────────────────────────────────────
# 일일 거래 요약
# ──────────────────────────────────────────────────────────────────────────────

def notify_daily_summary(
    trade_log: list,
    open_positions: dict,
    market: str = "US",
    paper_trading: bool = True,
):
    """일일 스캘핑 요약 알림"""
    mode = _mode_tag(paper_trading)
    today = datetime.now().strftime('%Y-%m-%d')

    if not trade_log:
        send_slack_message(
            f"*[{market}·{mode}] 일일 요약 — {today}*\n"
            f">오늘 체결된 거래 없음"
        )
        return

    total   = len(trade_log)
    wins    = sum(1 for t in trade_log if t['pnl_pct'] > 0)
    losses  = total - wins
    win_pct = wins / total * 100 if total else 0
    total_pnl = sum(t['pnl_pct'] for t in trade_log)
    avg_pnl   = total_pnl / total if total else 0

    pnl_sign = "+" if total_pnl >= 0 else ""
    result_emoji = "📈" if total_pnl > 0 else "📉"

    # 거래 내역 (최대 10건)
    trade_lines = []
    for t in trade_log[-10:]:
        ticker = t.get('api_ticker') or t.get('ticker', '-')
        pnl    = t['pnl_pct']
        reason = t.get('reason', '-')
        s      = "+" if pnl >= 0 else ""
        icon   = "✅" if pnl > 0 else ("⏱" if reason == "TIMEOUT" else "❌")
        trade_lines.append(f"  {icon} {ticker:8s} {s}{pnl:.2f}% [{reason}]")

    lines = [
        f"{result_emoji} *[{market}·{mode}] 일일 요약 — {today}*",
        f">총 거래: {total}건  |  승: {wins}  |  패: {losses}  |  승률: {win_pct:.1f}%",
        f">*누적 손익: {pnl_sign}{total_pnl:.2f}%*  (평균: {pnl_sign}{avg_pnl:.2f}%/건)",
    ]
    if open_positions:
        lines.append(f">미결 포지션: {len(open_positions)}개 ({', '.join(open_positions.keys())})")
    if trade_lines:
        lines.append(f">```\n거래 내역 (최근 {len(trade_lines)}건)\n" + '\n'.join(trade_lines) + "\n```")

    send_slack_message('\n'.join(lines))


# ──────────────────────────────────────────────────────────────────────────────
# 모델 재학습 알림
# ──────────────────────────────────────────────────────────────────────────────

def notify_retrain_complete(
    results: list,
    market: str = "US",
):
    """스캘핑 모델 재학습 완료 알림"""
    today = datetime.now().strftime('%Y-%m-%d')

    saved   = [r for r in results if r.get('saved')]
    skipped = len(results) - len(saved)

    if not saved:
        send_slack_message(
            f"*[{market}] 스캘핑 모델 재학습 완료 — {today}*\n"
            f">저장된 모델 없음 (총 {len(results)}개 시도, {skipped}개 조건 미달)"
        )
        return

    avg_wr  = sum(r['win_rate'] for r in saved) / len(saved)
    avg_ret = sum(r['avg_return'] for r in saved) / len(saved)
    ret_sign = "+" if avg_ret >= 0 else ""

    # 상위 3개 모델
    top3 = sorted(saved, key=lambda x: x['win_rate'], reverse=True)[:3]
    top_lines = []
    for r in top3:
        name   = r.get('name') or r.get('ticker', '-')
        wr     = r['win_rate']
        ar     = r['avg_return']
        s      = "+" if ar >= 0 else ""
        top_lines.append(f"  {name:12s}  WinR {wr:.1f}%  AvgRet {s}{ar:.2f}%")

    lines = [
        f"*[{market}] 스캘핑 모델 재학습 완료 — {today}*",
        f">저장: {len(saved)}개  |  미달 스킵: {skipped}개",
        f">평균 승률: {avg_wr:.1f}%  |  평균 수익/건: {ret_sign}{avg_ret:.2f}%",
    ]
    if top_lines:
        lines.append(f">```\nTop 모델\n" + '\n'.join(top_lines) + "\n```")

    send_slack_message('\n'.join(lines))


# ──────────────────────────────────────────────────────────────────────────────
# 워크플로우 시작/종료 알림
# ──────────────────────────────────────────────────────────────────────────────

def notify_workflow_start(mode: str, market: str = "US", paper_trading: bool = True):
    """워크플로우 시작 알림"""
    now  = datetime.now().strftime('%H:%M KST')
    ptag = _mode_tag(paper_trading)
    send_slack_message(f"▶ *[{market}·{ptag}] {mode} 시작* — {now}")


def notify_workflow_end(mode: str, results: dict, market: str = "US", paper_trading: bool = True):
    """워크플로우 종료 알림"""
    now  = datetime.now().strftime('%H:%M KST')
    ptag = _mode_tag(paper_trading)
    ok   = sum(1 for v in results.values() if v)
    fail = len(results) - ok
    emoji = "✅" if fail == 0 else "⚠️"
    lines = [
        f"{emoji} *[{market}·{ptag}] {mode} 완료* — {now}",
        f">성공: {ok}  |  실패: {fail}",
    ]
    for task, success in results.items():
        icon = "✅" if success else "❌"
        lines.append(f"  {icon} {task}")
    send_slack_message('\n'.join(lines))
