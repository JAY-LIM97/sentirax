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
"""

import sys
import os
import io
import argparse
import logging
from datetime import datetime

# 프로젝트 루트
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)
os.chdir(PROJECT_ROOT)

# UTF-8
if hasattr(sys.stdout, 'buffer'):
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
    """급등주 스캔 + 스캘핑 모델 재학습"""
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
    parser.add_argument('--refresh', action='store_true', help='Surging stocks scan + scalping retrain')
    parser.add_argument('--retrain', action='store_true', help='Swing model daily retrain')
    parser.add_argument('--performance', action='store_true', help='Performance tracking + auto-switch')
    parser.add_argument('--dashboard', action='store_true', help='Print dashboard')
    parser.add_argument('--all', action='store_true', help='Swing + scalping single scan')

    args = parser.parse_args()

    if not any(vars(args).values()):
        args.all = True

    logger.info(f"Sentirax started at {datetime.now()}")
    logger.info(f"Project: {PROJECT_ROOT}")

    # GitHub Actions에서 실행 시 .env 생성
    load_env_from_github()

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

    if args.dashboard:
        results['dashboard'] = run_dashboard()

    # 요약
    logger.info("=" * 70)
    logger.info("SUMMARY")
    logger.info("=" * 70)
    for task, success in results.items():
        logger.info(f"  {task}: {'OK' if success else 'FAILED'}")
    logger.info(f"Finished at {datetime.now()}")


if __name__ == "__main__":
    main()
