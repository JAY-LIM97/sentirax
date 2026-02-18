#!/usr/bin/env python3
"""
Sentirax Cloud Runner - GitHub Actions / 클라우드 헤드리스 실행

Usage:
    python cloud/run_cloud.py --all                  # 스윙 + 스캘핑 1회
    python cloud/run_cloud.py --swing                # 스윙만
    python cloud/run_cloud.py --scalping-continuous   # 스캘핑 2시간 연속
    python cloud/run_cloud.py --refresh              # 급등주 스캔 + 스캘핑 모델 갱신
    python cloud/run_cloud.py --retrain              # 스윙 모델 일일 재학습
    python cloud/run_cloud.py --performance          # 성과 추적 + 모델 자동 교체
    python cloud/run_cloud.py --dashboard            # 대시보드 출력

    # 국내주식 모드
    python cloud/run_cloud.py --kr-swing             # 국내 스윙 (RSI+MA)
    python cloud/run_cloud.py --kr-scalping-opening  # 국내 오프닝 서지 + 학습 + 스캘핑
    python cloud/run_cloud.py --kr-scalping-continuous # 국내 스캘핑 연속 (체크포인트 복원)
    python cloud/run_cloud.py --kr-retrain           # 국내 스캘핑 모델 일일 재학습
"""

import sys
import os
import io
import json
import argparse
import logging
from datetime import datetime, date

# 프로젝트 루트
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)
os.chdir(PROJECT_ROOT)

# UTF-8 (Windows cp949 환경에서만 필요)
import platform
if platform.system() == 'Windows' and hasattr(sys.stdout, 'buffer'):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

# 로깅
log_dir = os.path.join(PROJECT_ROOT, 'logs')
os.makedirs(log_dir, exist_ok=True)
log_file = os.path.join(log_dir, f"trading_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.FileHandler(log_file, encoding='utf-8'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger('sentirax')


def is_us_market_holiday(check_date: date = None) -> bool:
    """미국 증시 휴장일 체크 (NYSE/NASDAQ)"""
    if check_date is None:
        check_date = date.today()

    # 2025-2026 NYSE 휴장일 (고정)
    holidays = {
        # 2025
        date(2025, 1, 1),    # New Year's Day
        date(2025, 1, 20),   # MLK Day
        date(2025, 2, 17),   # Presidents' Day
        date(2025, 4, 18),   # Good Friday
        date(2025, 5, 26),   # Memorial Day
        date(2025, 6, 19),   # Juneteenth
        date(2025, 7, 4),    # Independence Day
        date(2025, 9, 1),    # Labor Day
        date(2025, 11, 27),  # Thanksgiving
        date(2025, 12, 25),  # Christmas
        # 2026
        date(2026, 1, 1),    # New Year's Day
        date(2026, 1, 19),   # MLK Day
        date(2026, 2, 16),   # Presidents' Day
        date(2026, 4, 3),    # Good Friday
        date(2026, 5, 25),   # Memorial Day
        date(2026, 6, 19),   # Juneteenth
        date(2026, 7, 3),    # Independence Day (observed)
        date(2026, 9, 7),    # Labor Day
        date(2026, 11, 26),  # Thanksgiving
        date(2026, 12, 25),  # Christmas
    }

    return check_date in holidays


def is_kr_market_holiday(check_date: date = None) -> bool:
    """한국 증시 휴장일 체크 (KOSPI/KOSDAQ) — 토/일 + 공휴일"""
    if check_date is None:
        check_date = date.today()

    # 주말은 무조건 휴장
    if check_date.weekday() >= 5:
        return True

    # 2025-2026 한국 증시 공휴일
    holidays = {
        # 2025
        date(2025, 1, 1),    # 신정
        date(2025, 1, 28),   # 설날 연휴
        date(2025, 1, 29),   # 설날
        date(2025, 1, 30),   # 설날 연휴
        date(2025, 3, 1),    # 삼일절
        date(2025, 5, 5),    # 어린이날
        date(2025, 5, 6),    # 어린이날 대체
        date(2025, 6, 6),    # 현충일
        date(2025, 8, 15),   # 광복절
        date(2025, 10, 3),   # 개천절
        date(2025, 10, 6),   # 추석 연휴
        date(2025, 10, 7),   # 추석
        date(2025, 10, 8),   # 추석 연휴
        date(2025, 10, 9),   # 한글날
        date(2025, 12, 25),  # 성탄절
        # 2026
        date(2026, 1, 1),    # 신정
        date(2026, 2, 16),   # 설날 연휴
        date(2026, 2, 17),   # 설날
        date(2026, 2, 18),   # 설날 연휴
        date(2026, 3, 1),    # 삼일절
        date(2026, 5, 5),    # 어린이날
        date(2026, 6, 6),    # 현충일
        date(2026, 8, 15),   # 광복절
        date(2026, 9, 24),   # 추석 연휴
        date(2026, 9, 25),   # 추석
        date(2026, 9, 26),   # 추석 연휴
        date(2026, 10, 3),   # 개천절
        date(2026, 10, 9),   # 한글날
        date(2026, 12, 25),  # 성탄절
    }

    return check_date in holidays


def run_kr_swing_trading():
    """국내주식 스윙 봇 (RSI + MA 기반)"""
    logger.info("=" * 70)
    logger.info("KR SWING TRADING BOT")
    logger.info("=" * 70)

    try:
        from scripts.domestic_swing_bot import DomesticSwingBot
        bot = DomesticSwingBot(paper_trading=True, account_no="50163140")
        bot.run_once(execute=True)
        logger.info("KR Swing trading completed")
        return True
    except Exception as e:
        logger.error(f"KR Swing trading error: {e}", exc_info=True)
        return False


def run_kr_scalping(continuous: bool = False):
    """국내주식 스캘핑 봇"""
    logger.info("=" * 70)
    logger.info(f"KR SCALPING BOT ({'continuous' if continuous else 'single scan'})")
    logger.info("=" * 70)

    try:
        from scripts.domestic_scalping_bot import DomesticScalpingBot

        bot = DomesticScalpingBot(paper_trading=True, account_no="50163140")

        logger.info("Loading KR scalping models...")
        models = bot.load_kr_models()

        if not models:
            logger.warning("No KR scalping models found!")
            return False

        logger.info(f"Loaded {len(models)} KR models")

        if continuous:
            duration = int(os.environ.get('SCALPING_DURATION', '120'))
            bot.run_continuous(models, duration_minutes=duration,
                               interval_seconds=60, execute=True)
        else:
            bot.run_scalping_cycle(models, execute=True)

        logger.info("KR Scalping completed")
        return True

    except Exception as e:
        logger.error(f"KR Scalping error: {e}", exc_info=True)
        return False


def run_kr_opening_scan_and_trade():
    """국내 장 시작 30분 대기 → 오프닝 서지 스캔 → 모델 학습 → 스캘핑"""
    import time as _time
    from datetime import datetime, timezone

    logger.info("=" * 70)
    logger.info("KR OPENING SURGE SCALPING")
    logger.info("=" * 70)

    results = {}

    # 1. 장 시작(UTC 00:00) + 30분 = UTC 00:30까지 대기
    now_utc = datetime.now(timezone.utc)
    market_open_30m = now_utc.replace(hour=0, minute=30, second=0, microsecond=0)
    # 이미 지났으면 다음 날 00:30이 아닌 즉시 진행
    if now_utc >= market_open_30m:
        logger.info(f"Already past KR 30min mark ({now_utc.strftime('%H:%M')} UTC). Proceeding.")
    else:
        wait_sec = (market_open_30m - now_utc).total_seconds()
        logger.info(f"Waiting {wait_sec/60:.1f}min until KR 30min after open (UTC 00:30)...")
        _time.sleep(wait_sec)

    # 2. 국내 오프닝 서지 스캔
    logger.info("=" * 70)
    logger.info("KR OPENING SURGE SCAN")
    logger.info("=" * 70)
    try:
        sys.path.insert(0, os.path.join(PROJECT_ROOT, 'scripts'))
        from get_domestic_surging_stocks import scan_domestic_surging_stocks
        top_stocks = scan_domestic_surging_stocks(top_n=20, opening_surge=True)

        if top_stocks is not None and not top_stocks.empty:
            tickers = top_stocks['api_ticker'].tolist()
            logger.info(f"KR Opening surge: {len(tickers)} stocks: {tickers}")
        else:
            logger.warning("KR Opening surge found no stocks. Using standard scan.")
            scan_domestic_surging_stocks(top_n=20, opening_surge=False)
        results['kr_opening_scan'] = True
    except Exception as e:
        logger.error(f"KR Opening scan error: {e}", exc_info=True)
        results['kr_opening_scan'] = False

    # 3. 국내 스캘핑 모델 학습
    logger.info("=" * 70)
    logger.info("KR SCALPING MODEL TRAINING")
    logger.info("=" * 70)
    try:
        from train_domestic_scalping_model import main as train_kr_scalping
        train_kr_scalping()
        results['kr_model_train'] = True
        logger.info("KR Scalping model training completed")
    except Exception as e:
        logger.error(f"KR Model training error: {e}", exc_info=True)
        results['kr_model_train'] = False

    # 4. 스캘핑 연속 실행
    results['kr_scalping'] = run_kr_scalping(continuous=True)

    return results


def run_kr_retrain():
    """국내 스캘핑 모델 일일 재학습 (장 시작 전 최신 데이터 반영)"""
    logger.info("=" * 70)
    logger.info("KR SCALPING MODEL DAILY RETRAIN")
    logger.info("=" * 70)

    try:
        sys.path.insert(0, os.path.join(PROJECT_ROOT, 'scripts'))
        from get_domestic_surging_stocks import scan_domestic_surging_stocks
        scan_domestic_surging_stocks(top_n=20)

        from train_domestic_scalping_model import main as train_kr
        train_kr()

        logger.info("KR daily retrain completed")
        return True
    except Exception as e:
        logger.error(f"KR retrain error: {e}", exc_info=True)
        return False


def save_daily_report(results: dict, mode: str):
    """일일 거래 요약 리포트 저장"""
    report_dir = os.path.join(PROJECT_ROOT, 'results', 'daily_reports')
    os.makedirs(report_dir, exist_ok=True)

    today_str = datetime.now().strftime('%Y%m%d')
    report_path = os.path.join(report_dir, f'report_{today_str}.json')

    report = {
        'date': datetime.now().strftime('%Y-%m-%d'),
        'timestamp': datetime.now().isoformat(),
        'mode': mode,
        'results': {k: ('OK' if v else 'FAILED') for k, v in results.items()},
        'success_count': sum(1 for v in results.values() if v),
        'fail_count': sum(1 for v in results.values() if not v),
    }

    # 기존 리포트가 있으면 병합 (같은 날 여러 번 실행 시)
    if os.path.exists(report_path):
        try:
            with open(report_path) as f:
                existing = json.load(f)
            if isinstance(existing, list):
                existing.append(report)
            else:
                existing = [existing, report]
            report = existing
        except Exception:
            report = [report]
    else:
        report = [report]

    with open(report_path, 'w') as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    logger.info(f"Daily report saved: {report_path}")


def write_github_summary(results: dict, mode: str):
    """GitHub Actions Job Summary 출력"""
    summary_file = os.environ.get('GITHUB_STEP_SUMMARY')
    if not summary_file:
        return

    lines = [
        f"## Sentirax {mode} - {datetime.now().strftime('%Y-%m-%d %H:%M')} UTC",
        "",
        "| Task | Status |",
        "|------|--------|",
    ]
    for task, success in results.items():
        status = "OK" if success else "FAILED"
        lines.append(f"| {task} | {status} |")

    lines.append("")
    ok = sum(1 for v in results.values() if v)
    fail = sum(1 for v in results.values() if not v)
    lines.append(f"**Result:** {ok} succeeded, {fail} failed")

    with open(summary_file, 'a') as f:
        f.write('\n'.join(lines) + '\n')


def load_env_from_github():
    """GitHub Actions 환경변수에서 .env 파일 생성"""
    env_path = os.path.join(PROJECT_ROOT, '.env')
    if os.path.exists(env_path):
        return

    # GitHub Actions secrets → .env
    env_vars = ['HT_API_KEY', 'HT_API_SECRET_KEY', 'HT_API_FK_KEY', 'HT_API_FK_SECRET_KEY']
    lines = []
    for var in env_vars:
        val = os.environ.get(var, '')
        if val:
            lines.append(f"{var}={val}")

    if lines:
        with open(env_path, 'w') as f:
            f.write('\n'.join(lines) + '\n')
        logger.info(f"Created .env from environment variables ({len(lines)} keys)")


def run_swing_trading():
    """스윙 트레이딩 봇 (헤드리스)"""
    logger.info("=" * 70)
    logger.info("SWING TRADING BOT")
    logger.info("=" * 70)

    try:
        from core.model_manager import ModelAutoSwitcher
        switcher = ModelAutoSwitcher(PROJECT_ROOT)
        active_tickers = switcher.get_active_tickers('swing')

        from scripts.auto_trading_bot import AutoTradingBot, TOP20_TICKERS
        tickers = active_tickers if active_tickers else TOP20_TICKERS

        logger.info(f"Active tickers: {len(tickers)}")

        bot = AutoTradingBot(paper_trading=True, account_no="50163140")
        bot.run_once(tickers=tickers, execute=True)

        # 거래 결과 기록
        try:
            from core.model_manager import PerformanceTracker
            pt = PerformanceTracker(PROJECT_ROOT)
            pt.calculate_daily_performance()
            logger.info("Trade results logged to performance tracker")
        except Exception as e:
            logger.warning(f"Performance logging skipped: {e}")

        logger.info("Swing trading completed")
        return True

    except Exception as e:
        logger.error(f"Swing trading error: {e}", exc_info=True)
        return False


def run_scalping(continuous: bool = False):
    """스캘핑 봇 (헤드리스)"""
    logger.info("=" * 70)
    logger.info(f"SCALPING BOT ({'continuous 2hr' if continuous else 'single scan'})")
    logger.info("=" * 70)

    try:
        from scripts.scalping_bot import ScalpingBot

        bot = ScalpingBot(paper_trading=True, account_no="50163140")

        logger.info("Loading scalping models...")
        models = bot.load_scalping_models()

        if not models:
            logger.warning("No scalping models found!")
            return False

        logger.info(f"Loaded {len(models)} models")

        if continuous:
            duration = int(os.environ.get('SCALPING_DURATION', '120'))
            bot.run_continuous(models, duration_minutes=duration, interval_seconds=60, execute=True)
        else:
            bot.run_once(models, execute=True)

        logger.info("Scalping completed")
        return True

    except Exception as e:
        logger.error(f"Scalping error: {e}", exc_info=True)
        return False


def run_model_refresh():
    """급등주 스캔 + 스캘핑 모델 재학습 (기존 방식)"""
    logger.info("=" * 70)
    logger.info("MODEL REFRESH - surging stocks + scalping retrain")
    logger.info("=" * 70)

    try:
        sys.path.insert(0, os.path.join(PROJECT_ROOT, 'scripts'))
        from get_surging_stocks import scan_surging_stocks
        scan_surging_stocks(top_n=20)

        from train_scalping_model import main as train_scalping
        train_scalping()

        logger.info("Model refresh completed")
        return True

    except Exception as e:
        logger.error(f"Model refresh error: {e}", exc_info=True)
        return False


def run_opening_scan_and_trade():
    """장 시작 30분 대기 → 오프닝 서지 스캔 → 모델 학습 → 스캘핑 연속 실행"""
    import time as _time
    from datetime import datetime, timezone

    logger.info("=" * 70)
    logger.info("OPENING SURGE SCALPING")
    logger.info("=" * 70)

    results = {}

    # 1. 장 시작 후 30분까지 대기
    # 미국 장 시작 = UTC 14:30 → 30분 후 = UTC 15:00
    now_utc = datetime.now(timezone.utc)
    market_open_30m = now_utc.replace(hour=15, minute=0, second=0, microsecond=0)

    if now_utc < market_open_30m:
        wait_seconds = (market_open_30m - now_utc).total_seconds()
        logger.info(f"Waiting {wait_seconds/60:.1f}min until 30min after market open (UTC 15:00)...")
        _time.sleep(wait_seconds)
    else:
        logger.info(f"Already past 30min mark (now={now_utc.strftime('%H:%M')} UTC). Proceeding immediately.")

    # 2. 오프닝 서지 스캔
    logger.info("=" * 70)
    logger.info("OPENING SURGE SCAN")
    logger.info("=" * 70)
    try:
        sys.path.insert(0, os.path.join(PROJECT_ROOT, 'scripts'))
        from get_surging_stocks import scan_surging_stocks
        top_stocks = scan_surging_stocks(top_n=20, opening_surge=True)

        if top_stocks is not None and not top_stocks.empty:
            tickers = top_stocks['ticker'].tolist()
            logger.info(f"Opening surge: {len(tickers)} stocks selected: {tickers}")
            results['opening_scan'] = True
        else:
            logger.warning("Opening surge scan found no stocks. Falling back to standard scan.")
            top_stocks = scan_surging_stocks(top_n=20, opening_surge=False)
            results['opening_scan'] = True
    except Exception as e:
        logger.error(f"Opening scan error: {e}", exc_info=True)
        results['opening_scan'] = False
        # 실패해도 기존 방식으로 대체 시도
        try:
            from get_surging_stocks import scan_surging_stocks
            scan_surging_stocks(top_n=20, opening_surge=False)
        except Exception:
            pass

    # 3. 스캘핑 모델 학습 (오프닝 서지 결과 또는 기존 surging_stocks_today.csv 사용)
    logger.info("=" * 70)
    logger.info("SCALPING MODEL TRAINING")
    logger.info("=" * 70)
    try:
        from train_scalping_model import main as train_scalping
        train_scalping()
        results['model_train'] = True
        logger.info("Scalping model training completed")
    except Exception as e:
        logger.error(f"Model training error: {e}", exc_info=True)
        results['model_train'] = False

    # 4. 스캘핑 연속 실행 (남은 시간)
    results['scalping'] = run_scalping(continuous=True)

    return results


def run_swing_retrain():
    """스윙 모델 일일 재학습 (최신 데이터 반영, 성과 기반)"""
    logger.info("=" * 70)
    logger.info("SWING MODEL DAILY RETRAIN")
    logger.info("=" * 70)

    try:
        from core.model_manager import SwingModelRetrainer, PerformanceTracker

        tracker = PerformanceTracker(PROJECT_ROOT)
        retrainer = SwingModelRetrainer(PROJECT_ROOT)
        results = retrainer.retrain_all(tracker=tracker)

        replaced = sum(1 for r in results if r.get('replaced', False))
        logger.info(f"Retrain done: {replaced} models replaced")
        return True

    except Exception as e:
        logger.error(f"Retrain error: {e}", exc_info=True)
        return False


def run_performance_tracking():
    """성과 추적 + 모델 자동 교체 판단"""
    logger.info("=" * 70)
    logger.info("PERFORMANCE TRACKING + AUTO-SWITCH")
    logger.info("=" * 70)

    try:
        from core.model_manager import PerformanceTracker, ModelAutoSwitcher

        tracker = PerformanceTracker(PROJECT_ROOT)

        # 일별 성과 계산
        summary = tracker.calculate_daily_performance()
        if summary:
            logger.info(f"Daily: {summary['total_positions']} positions, "
                        f"avg P&L={summary['avg_pnl']:+.2f}%, "
                        f"win rate={summary['win_rate']:.1f}%")

        # 모델 점수 업데이트 + 자동 교체
        switcher = ModelAutoSwitcher(PROJECT_ROOT)
        switcher.evaluate_and_switch(tracker)

        logger.info("Performance tracking completed")
        return True

    except Exception as e:
        logger.error(f"Performance tracking error: {e}", exc_info=True)
        return False


def run_dashboard():
    """대시보드 출력"""
    try:
        from core.model_manager import PerformanceTracker
        tracker = PerformanceTracker(PROJECT_ROOT)
        print(tracker.get_dashboard())
        return True
    except Exception as e:
        logger.error(f"Dashboard error: {e}", exc_info=True)
        return False


def main():
    parser = argparse.ArgumentParser(description='Sentirax Cloud Trading Bot')
    parser.add_argument('--swing', action='store_true', help='Swing trading')
    parser.add_argument('--scalping', action='store_true', help='Scalping single scan')
    parser.add_argument('--scalping-continuous', action='store_true', help='Scalping 2hr continuous')
    parser.add_argument('--scalping-opening', action='store_true', help='Opening surge scan + train + scalping')
    parser.add_argument('--refresh', action='store_true', help='Surging stocks scan + scalping retrain')
    parser.add_argument('--retrain', action='store_true', help='Swing model daily retrain')
    parser.add_argument('--performance', action='store_true', help='Performance tracking + auto-switch')
    parser.add_argument('--dashboard', action='store_true', help='Print dashboard')
    parser.add_argument('--all', action='store_true', help='Swing + scalping single scan')
    # 국내주식 모드
    parser.add_argument('--kr-swing', action='store_true', help='KR domestic swing trading')
    parser.add_argument('--kr-scalping-continuous', action='store_true', help='KR scalping continuous')
    parser.add_argument('--kr-scalping-opening', action='store_true', help='KR opening surge + train + scalping')
    parser.add_argument('--kr-retrain', action='store_true', help='KR scalping model daily retrain')

    args = parser.parse_args()

    if not any(vars(args).values()):
        args.all = True

    logger.info(f"Sentirax started at {datetime.now()}")
    logger.info(f"Project: {PROJECT_ROOT}")

    # 국내 모드 여부 판별
    kr_mode = any([args.kr_swing, args.kr_scalping_continuous,
                   args.kr_scalping_opening, args.kr_retrain])

    if kr_mode:
        # 한국 휴장일 체크
        if is_kr_market_holiday():
            logger.info(f"KR market is CLOSED today ({date.today()}). Skipping.")
            write_github_summary({'holiday_skip': True}, 'KR Holiday')
            return
    else:
        # 미국 휴장일 체크
        if is_us_market_holiday():
            logger.info(f"US market is CLOSED today ({date.today()}). Skipping all trading.")
            write_github_summary({'holiday_skip': True}, 'Holiday')
            return

    # GitHub Actions에서 실행 시 .env 생성
    load_env_from_github()

    # 실행 모드 판별
    mode_parts = [k for k, v in vars(args).items() if v and k != 'all']
    mode = '+'.join(mode_parts) if mode_parts else 'all'

    results = {}

    if args.refresh:
        results['refresh'] = run_model_refresh()

    if args.retrain:
        results['retrain'] = run_swing_retrain()

    if args.performance:
        results['performance'] = run_performance_tracking()

    if args.swing or args.all:
        results['swing'] = run_swing_trading()

    if args.scalping or args.all:
        results['scalping'] = run_scalping(continuous=False)

    if args.scalping_continuous:
        results['scalping'] = run_scalping(continuous=True)

    if args.scalping_opening:
        opening_results = run_opening_scan_and_trade()
        results.update(opening_results)

    if args.dashboard:
        results['dashboard'] = run_dashboard()

    # 국내주식 모드
    if args.kr_swing:
        results['kr_swing'] = run_kr_swing_trading()

    if args.kr_scalping_continuous:
        results['kr_scalping'] = run_kr_scalping(continuous=True)

    if args.kr_scalping_opening:
        kr_results = run_kr_opening_scan_and_trade()
        results.update(kr_results)

    if args.kr_retrain:
        results['kr_retrain'] = run_kr_retrain()

    # 요약
    logger.info("=" * 70)
    logger.info("SUMMARY")
    logger.info("=" * 70)
    for task, success in results.items():
        logger.info(f"  {task}: {'OK' if success else 'FAILED'}")
    logger.info(f"Finished at {datetime.now()}")

    # 일일 리포트 저장 + GitHub Summary
    save_daily_report(results, mode)
    write_github_summary(results, mode)


if __name__ == "__main__":
    main()
