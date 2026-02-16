#!/usr/bin/env python3
"""
Sentirax Cloud Runner - 클라우드 환경 헤드리스 실행

Oracle Cloud / Linux 서버에서 cron으로 자동 실행되는 스크립트.
input() 없이 자동으로 스윙 트레이딩 + 스캘핑을 순차 실행한다.

Usage:
    python cloud/run_cloud.py                    # 둘 다 실행
    python cloud/run_cloud.py --swing            # 스윙만
    python cloud/run_cloud.py --scalping         # 스캘핑만
    python cloud/run_cloud.py --scalping-continuous  # 스캘핑 연속 모드 (2시간)
"""

import sys
import os
import io
import argparse
import logging
from datetime import datetime

# 프로젝트 루트를 path에 추가
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)
os.chdir(PROJECT_ROOT)

# UTF-8 설정
if hasattr(sys.stdout, 'buffer'):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

# 로깅 설정
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


def run_swing_trading():
    """스윙 트레이딩 봇 실행 (헤드리스)"""
    logger.info("=" * 70)
    logger.info("SWING TRADING BOT - Cloud Mode")
    logger.info("=" * 70)

    try:
        from scripts.auto_trading_bot import AutoTradingBot, TOP20_TICKERS

        bot = AutoTradingBot(paper_trading=True, account_no="50163140")
        bot.run_once(tickers=TOP20_TICKERS, execute=True)

        logger.info("Swing trading completed successfully")
        return True

    except Exception as e:
        logger.error(f"Swing trading error: {e}", exc_info=True)
        return False


def run_scalping(continuous: bool = False):
    """스캘핑 봇 실행 (헤드리스)"""
    logger.info("=" * 70)
    logger.info("SCALPING BOT - Cloud Mode")
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
            # 2시간 연속 실행 (1분 간격)
            bot.run_continuous(models, duration_minutes=120, interval_seconds=60, execute=True)
        else:
            # 1회 스캔
            bot.run_once(models, execute=True)

        logger.info("Scalping completed successfully")
        return True

    except Exception as e:
        logger.error(f"Scalping error: {e}", exc_info=True)
        return False


def run_model_refresh():
    """매일 장 시작 전 급등주 스캔 및 스캘핑 모델 갱신"""
    logger.info("=" * 70)
    logger.info("MODEL REFRESH - Scanning surging stocks")
    logger.info("=" * 70)

    try:
        # 급등주 스캔
        sys.path.insert(0, os.path.join(PROJECT_ROOT, 'scripts'))
        from get_surging_stocks import scan_surging_stocks
        scan_surging_stocks(top_n=20)

        # 스캘핑 모델 재학습
        from train_scalping_model import main as train_scalping
        train_scalping()

        logger.info("Model refresh completed")
        return True

    except Exception as e:
        logger.error(f"Model refresh error: {e}", exc_info=True)
        return False


def main():
    parser = argparse.ArgumentParser(description='Sentirax Cloud Trading Bot')
    parser.add_argument('--swing', action='store_true', help='Run swing trading only')
    parser.add_argument('--scalping', action='store_true', help='Run scalping (single scan)')
    parser.add_argument('--scalping-continuous', action='store_true', help='Run scalping (2hr continuous)')
    parser.add_argument('--refresh', action='store_true', help='Refresh surging stocks & retrain scalping models')
    parser.add_argument('--all', action='store_true', help='Run swing + scalping single scan (default)')

    args = parser.parse_args()

    # 기본: --all
    if not any([args.swing, args.scalping, args.scalping_continuous, args.refresh, args.all]):
        args.all = True

    logger.info(f"Sentirax Cloud Runner started at {datetime.now()}")
    logger.info(f"Project root: {PROJECT_ROOT}")
    logger.info(f"Log file: {log_file}")

    results = {}

    if args.refresh:
        results['refresh'] = run_model_refresh()

    if args.swing or args.all:
        results['swing'] = run_swing_trading()

    if args.scalping or args.all:
        results['scalping'] = run_scalping(continuous=False)

    if args.scalping_continuous:
        results['scalping'] = run_scalping(continuous=True)

    # 결과 요약
    logger.info("=" * 70)
    logger.info("EXECUTION SUMMARY")
    logger.info("=" * 70)
    for task, success in results.items():
        status = "OK" if success else "FAILED"
        logger.info(f"  {task}: {status}")

    logger.info(f"Finished at {datetime.now()}")
    logger.info(f"Full log: {log_file}")


if __name__ == "__main__":
    main()
