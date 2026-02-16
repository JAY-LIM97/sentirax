"""
Sentirax Scalping Bot - 급등주 스캘핑 자동매매

- 급등주 TOP 20에서 ML 모델 통과한 종목 대상
- 1분 간격 실시간 모니터링
- 매수 신호 시 진입, TP/SL 자동 관리
- 모의투자 계좌로 실행
"""

import sys
import os
import io
import time
import json
import pickle

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta

from core.kis_trading_api import KISTradingAPI

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def load_strategy_config() -> dict:
    """strategy.json에서 실시간 전략 설정 로드 (GitHub에서 수정 가능)"""
    config_path = os.path.join(PROJECT_ROOT, 'config', 'strategy.json')

    # GitHub Actions에서는 git pull로 최신 설정 가져오기
    if os.environ.get('GITHUB_ACTIONS'):
        try:
            os.system(f'cd {PROJECT_ROOT} && git pull origin main --quiet 2>/dev/null')
        except Exception:
            pass

    if os.path.exists(config_path):
        with open(config_path) as f:
            return json.load(f)

    # 기본값
    return {
        'scalping': {
            'enabled': True,
            'max_positions': 5,
            'min_probability': 0.55,
            'order_quantity': 1,
            'scan_interval_seconds': 60,
            'tp_override': None,
            'sl_override': None,
            'disabled_tickers': [],
            'forced_buy_tickers': [],
            'only_tickers': []
        },
        'risk': {
            'max_daily_loss_pct': -10.0,
            'max_total_positions': 20,
            'stop_all_trading': False,
            'paper_trading': True
        }
    }


# 스캘핑 Feature 생성 함수 (train_scalping_model.py와 동일)
def create_scalping_features(df: pd.DataFrame) -> pd.DataFrame:
    """스캘핑용 Feature Engineering (1분봉)"""

    feat = pd.DataFrame(index=df.index)
    feat['close'] = df['Close']
    feat['volume'] = df['Volume']

    for period in [1, 3, 5, 10, 15, 30]:
        feat[f'return_{period}m'] = df['Close'].pct_change(period) * 100

    for period in [5, 10, 20, 60]:
        feat[f'ma_{period}m'] = df['Close'].rolling(period).mean()
        feat[f'price_to_ma_{period}m'] = (df['Close'] / feat[f'ma_{period}m'] - 1) * 100

    ma20 = df['Close'].rolling(20).mean()
    std20 = df['Close'].rolling(20).std()
    feat['bb_upper'] = (df['Close'] - (ma20 + 2 * std20)) / df['Close'] * 100
    feat['bb_lower'] = (df['Close'] - (ma20 - 2 * std20)) / df['Close'] * 100
    feat['bb_width'] = (4 * std20) / ma20 * 100

    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    rs = gain / loss
    feat['rsi_14m'] = 100 - (100 / (1 + rs))

    feat['volume_ratio_5m'] = df['Volume'] / df['Volume'].rolling(5).mean()
    feat['volume_ratio_20m'] = df['Volume'] / df['Volume'].rolling(20).mean()
    feat['volume_spike'] = (df['Volume'] > df['Volume'].rolling(20).mean() * 2).astype(int)

    feat['candle_body'] = (df['Close'] - df['Open']) / df['Open'] * 100
    feat['candle_wick_upper'] = (df['High'] - df[['Open', 'Close']].max(axis=1)) / df['Open'] * 100
    feat['candle_wick_lower'] = (df[['Open', 'Close']].min(axis=1) - df['Low']) / df['Open'] * 100

    day_high = df['High'].rolling(60).max()
    day_low = df['Low'].rolling(60).min()
    feat['price_position'] = (df['Close'] - day_low) / (day_high - day_low + 0.001) * 100

    ret_1m = df['Close'].pct_change() * 100
    feat['momentum_accel'] = ret_1m.diff()

    feat['volatility_5m'] = df['Close'].pct_change().rolling(5).std() * 100
    feat['volatility_20m'] = df['Close'].pct_change().rolling(20).std() * 100

    typical_price = (df['High'] + df['Low'] + df['Close']) / 3
    cum_vol = df['Volume'].cumsum()
    cum_vwap = (typical_price * df['Volume']).cumsum()
    feat['vwap_deviation'] = (df['Close'] / (cum_vwap / cum_vol) - 1) * 100

    return feat


class ScalpingBot:
    """스캘핑 자동매매 봇"""

    def __init__(self, paper_trading: bool = True, account_no: str = None):
        self.paper_trading = paper_trading
        self.models_dir = os.path.join(os.path.dirname(__file__), '..', 'models')

        # KIS API 초기화
        print("=" * 70)
        print("Sentirax Scalping Bot v1.0")
        print("=" * 70)

        self.api = KISTradingAPI(paper_trading=paper_trading)
        if not self.api.authenticate():
            raise Exception("API Authentication Failed!")

        if account_no:
            self.api.set_account(account_no, "01")

        # 포지션 추적
        self.positions = {}  # {ticker: {entry_price, quantity, entry_time, tp, sl}}
        self.trade_log = []

        print("  Bot initialized!")

    def load_scalping_models(self) -> dict:
        """저장된 스캘핑 모델 로드"""
        models = {}

        for f in os.listdir(self.models_dir):
            if f.endswith('_scalping.pkl'):
                ticker = f.replace('_scalping.pkl', '').upper()
                model_path = os.path.join(self.models_dir, f)

                with open(model_path, 'rb') as fp:
                    model_data = pickle.load(fp)

                models[ticker] = model_data
                perf = model_data['performance']
                params = model_data['params']
                print(f"  Loaded {ticker}: Acc={perf['accuracy']*100:.1f}%, "
                      f"WinR={perf['win_rate']:.1f}%, "
                      f"TP={params['take_profit']}%, SL={params['stop_loss']}%")

        return models

    def get_latest_1min_data(self, ticker: str) -> pd.DataFrame:
        """최신 1분봉 데이터 수집"""
        try:
            stock = yf.Ticker(ticker)
            df = stock.history(period='1d', interval='1m')
            if df.empty:
                # fallback to 2d
                df = stock.history(period='2d', interval='1m')
            return df
        except Exception as e:
            print(f"  Data error {ticker}: {e}")
            return None

    def predict_entry(self, ticker: str, model_data: dict) -> dict:
        """매수 진입 신호 예측"""

        df = self.get_latest_1min_data(ticker)
        if df is None or len(df) < 60:
            return None

        model = model_data['model']
        scaler = model_data['scaler']
        feature_names = model_data['feature_names']
        params = model_data['params']

        # Feature 생성
        features = create_scalping_features(df)
        X = features[feature_names]
        valid_idx = X.notna().all(axis=1)
        X_clean = X[valid_idx]

        if len(X_clean) == 0:
            return None

        # 최신 데이터로 예측
        X_latest = X_clean.iloc[-1:]
        X_scaled = scaler.transform(X_latest)

        prediction = model.predict(X_scaled)[0]
        probability = model.predict_proba(X_scaled)[0]

        current_price = df['Close'].iloc[-1]

        return {
            'ticker': ticker,
            'signal': int(prediction),
            'buy_prob': probability[1] if len(probability) > 1 else 0,
            'price': current_price,
            'tp': params['take_profit'],
            'sl': params['stop_loss'],
            'time': datetime.now()
        }

    def check_positions(self):
        """보유 포지션 TP/SL 체크"""

        closed_tickers = []

        for ticker, pos in self.positions.items():
            try:
                stock = yf.Ticker(ticker)
                df = stock.history(period='1d', interval='1m')
                if df.empty:
                    continue

                current_price = df['Close'].iloc[-1]
                pnl_pct = (current_price / pos['entry_price'] - 1) * 100
                hold_minutes = (datetime.now() - pos['entry_time']).total_seconds() / 60

                # TP/SL 체크
                if pnl_pct >= pos['tp']:
                    print(f"  TP HIT {ticker}: +{pnl_pct:.2f}% (target: +{pos['tp']}%)")
                    self._close_position(ticker, current_price, 'TP')
                    closed_tickers.append(ticker)

                elif pnl_pct <= -pos['sl']:
                    print(f"  SL HIT {ticker}: {pnl_pct:.2f}% (limit: -{pos['sl']}%)")
                    self._close_position(ticker, current_price, 'SL')
                    closed_tickers.append(ticker)

                elif hold_minutes >= 60:
                    print(f"  TIMEOUT {ticker}: {pnl_pct:+.2f}% after {hold_minutes:.0f}min")
                    self._close_position(ticker, current_price, 'TIMEOUT')
                    closed_tickers.append(ticker)

                else:
                    print(f"  HOLD {ticker}: {pnl_pct:+.2f}% ({hold_minutes:.0f}min)")

            except Exception as e:
                print(f"  Error checking {ticker}: {e}")

        for t in closed_tickers:
            del self.positions[t]

    def _close_position(self, ticker: str, exit_price: float, reason: str):
        """포지션 종료 (매도)"""
        pos = self.positions[ticker]
        pnl_pct = (exit_price / pos['entry_price'] - 1) * 100

        # 매도 주문
        result = self.api.order_sell(ticker, pos['quantity'], price=0)

        self.trade_log.append({
            'ticker': ticker,
            'entry_price': pos['entry_price'],
            'exit_price': exit_price,
            'pnl_pct': pnl_pct,
            'reason': reason,
            'entry_time': pos['entry_time'],
            'exit_time': datetime.now()
        })

        print(f"  CLOSED {ticker}: {pnl_pct:+.2f}% ({reason})")

    def run_scalping_cycle(self, models: dict, max_positions: int = 5, execute: bool = False):
        """
        스캘핑 1사이클 실행 (매 사이클마다 strategy.json 실시간 반영)
        """

        # 전략 설정 실시간 로드
        config = load_strategy_config()
        scalp_cfg = config.get('scalping', {})
        risk_cfg = config.get('risk', {})

        # 긴급 정지 체크
        if risk_cfg.get('stop_all_trading', False):
            print(f"\n  !! EMERGENCY STOP - stop_all_trading=true in strategy.json")
            return

        if not scalp_cfg.get('enabled', True):
            print(f"\n  !! Scalping disabled in strategy.json")
            return

        # 설정 반영
        max_positions = scalp_cfg.get('max_positions', max_positions)
        min_prob = scalp_cfg.get('min_probability', 0.55)
        order_qty = scalp_cfg.get('order_quantity', 1)
        tp_override = scalp_cfg.get('tp_override')
        sl_override = scalp_cfg.get('sl_override')
        disabled = [t.upper() for t in scalp_cfg.get('disabled_tickers', [])]
        forced_buy = [t.upper() for t in scalp_cfg.get('forced_buy_tickers', [])]
        only_tickers = [t.upper() for t in scalp_cfg.get('only_tickers', [])]

        print(f"\n{'='*70}")
        print(f"  Scalping Cycle - {datetime.now().strftime('%H:%M:%S')}")
        print(f"  Positions: {len(self.positions)}/{max_positions} | "
              f"MinProb: {min_prob} | Qty: {order_qty}")
        if disabled:
            print(f"  Disabled: {disabled}")
        if forced_buy:
            print(f"  Forced BUY: {forced_buy}")
        if only_tickers:
            print(f"  Only tickers: {only_tickers}")
        print(f"{'='*70}")

        # 일일 손실 체크
        total_pnl = sum(t['pnl_pct'] for t in self.trade_log) if self.trade_log else 0
        max_daily_loss = risk_cfg.get('max_daily_loss_pct', -10.0)
        if total_pnl <= max_daily_loss:
            print(f"\n  !! DAILY LOSS LIMIT HIT: {total_pnl:.2f}% <= {max_daily_loss}%")
            print(f"  !! No new entries allowed")
            # 포지션 체크만 진행
            if self.positions:
                self.check_positions()
            return

        # 1. 보유 포지션 체크
        if self.positions:
            print(f"\n  [Position Check]")
            self.check_positions()

        # 2. 새 매수 신호 탐색
        if len(self.positions) < max_positions:
            print(f"\n  [Scanning for entries]")

            # 대상 모델 필터링
            scan_models = models
            if only_tickers:
                scan_models = {k: v for k, v in models.items() if k in only_tickers}

            for ticker, model_data in scan_models.items():
                if ticker in self.positions:
                    continue
                if ticker in disabled:
                    continue
                if len(self.positions) >= max_positions:
                    break

                # 강제 매수
                if ticker in forced_buy:
                    print(f"\n  FORCED BUY: {ticker}")
                    if execute:
                        order_result = self.api.order_buy(ticker, order_qty, price=0)
                        if order_result:
                            result = self.predict_entry(ticker, model_data)
                            price = result['price'] if result else 0
                            self.positions[ticker] = {
                                'entry_price': price,
                                'quantity': order_qty,
                                'entry_time': datetime.now(),
                                'tp': tp_override or model_data['params']['take_profit'],
                                'sl': sl_override or model_data['params']['stop_loss']
                            }
                            print(f"    FORCED ORDER PLACED!")
                    continue

                result = self.predict_entry(ticker, model_data)
                if result is None:
                    continue

                # TP/SL 오버라이드
                if tp_override:
                    result['tp'] = tp_override
                if sl_override:
                    result['sl'] = sl_override

                if result['signal'] == 1 and result['buy_prob'] >= min_prob:
                    print(f"\n  BUY SIGNAL: {ticker}")
                    print(f"    Price: ${result['price']:.2f}")
                    print(f"    Probability: {result['buy_prob']*100:.1f}%")
                    print(f"    TP: +{result['tp']}%, SL: -{result['sl']}%")

                    if execute:
                        order_result = self.api.order_buy(ticker, order_qty, price=0)
                        if order_result:
                            self.positions[ticker] = {
                                'entry_price': result['price'],
                                'quantity': order_qty,
                                'entry_time': datetime.now(),
                                'tp': result['tp'],
                                'sl': result['sl']
                            }
                            print(f"    ORDER PLACED!")
                        else:
                            print(f"    ORDER FAILED")
                    else:
                        self.positions[ticker] = {
                            'entry_price': result['price'],
                            'quantity': order_qty,
                            'entry_time': datetime.now(),
                            'tp': result['tp'],
                            'sl': result['sl']
                        }
                        print(f"    [SIMULATED]")

                else:
                    signal_text = "BUY" if result['signal'] == 1 else "NO"
                    print(f"  {ticker}: {signal_text} (prob={result['buy_prob']*100:.1f}%)")

    def run_continuous(self, models: dict, duration_minutes: int = 120,
                       interval_seconds: int = 60, execute: bool = False):
        """
        연속 스캘핑 실행 (매 사이클마다 strategy.json 실시간 반영)
        """

        config = load_strategy_config()
        scalp_cfg = config.get('scalping', {})
        interval_seconds = scalp_cfg.get('scan_interval_seconds', interval_seconds)

        print(f"\n{'='*70}")
        print(f"  CONTINUOUS SCALPING MODE")
        print(f"  Duration: {duration_minutes}min, Interval: {interval_seconds}s")
        print(f"  Execute: {'YES' if execute else 'NO (simulation)'}")
        print(f"  Models: {len(models)} stocks")
        print(f"  Strategy: config/strategy.json (live reload every cycle)")
        print(f"{'='*70}")

        start_time = datetime.now()
        end_time = start_time + timedelta(minutes=duration_minutes)
        cycle = 0

        try:
            while datetime.now() < end_time:
                cycle += 1

                # 매 사이클마다 전략 설정 다시 로드 (실시간 반영)
                live_config = load_strategy_config()
                risk_cfg = live_config.get('risk', {})

                # 긴급 정지 체크
                if risk_cfg.get('stop_all_trading', False):
                    print(f"\n  !! EMERGENCY STOP activated. Closing all positions...")
                    for ticker in list(self.positions.keys()):
                        try:
                            stock = yf.Ticker(ticker)
                            df = stock.history(period='1d', interval='1m')
                            if not df.empty:
                                self._close_position(ticker, df['Close'].iloc[-1], 'EMERGENCY')
                        except Exception:
                            pass
                    self.positions.clear()
                    break

                live_interval = live_config.get('scalping', {}).get('scan_interval_seconds', interval_seconds)

                self.run_scalping_cycle(models, execute=execute)

                remaining = (end_time - datetime.now()).total_seconds() / 60
                print(f"\n  Cycle #{cycle} | Remaining: {remaining:.1f}min | "
                      f"Next in {live_interval}s...")

                if datetime.now() < end_time:
                    time.sleep(live_interval)

        except KeyboardInterrupt:
            print("\n\n  STOPPED by user (Ctrl+C)")

        # 최종 요약
        self._print_summary()

    def run_once(self, models: dict, execute: bool = False):
        """1회 실행 (모든 종목 스캔 후 매수/매도)"""

        print(f"\n{'='*70}")
        print(f"  SINGLE SCAN MODE")
        print(f"  Models: {len(models)} stocks")
        print(f"  Execute: {'YES' if execute else 'NO (prediction only)'}")
        print(f"{'='*70}")

        buy_signals = []
        no_signals = []

        for ticker, model_data in models.items():
            result = self.predict_entry(ticker, model_data)
            if result is None:
                continue

            if result['signal'] == 1:
                buy_signals.append(result)
                emoji = "BUY"
            else:
                no_signals.append(result)
                emoji = "---"

            print(f"  {emoji} {ticker}: ${result['price']:.2f} "
                  f"(prob={result['buy_prob']*100:.1f}%, "
                  f"TP={result['tp']}%/SL={result['sl']}%)")

        # 매수 신호 있는 종목에 주문
        if execute and buy_signals:
            print(f"\n  Executing {len(buy_signals)} buy orders...")
            for sig in buy_signals:
                if sig['buy_prob'] >= 0.55:
                    result = self.api.order_buy(sig['ticker'], 1, price=0)
                    status = "OK" if result else "FAIL"
                    print(f"  ORDER {sig['ticker']}: {status}")

        # 요약
        print(f"\n{'='*70}")
        print(f"  RESULTS")
        print(f"{'='*70}")
        print(f"  BUY signals: {len(buy_signals)}")
        for s in buy_signals:
            print(f"    {s['ticker']}: ${s['price']:.2f} (prob={s['buy_prob']*100:.1f}%)")
        print(f"  NO signals:  {len(no_signals)}")

    def _print_summary(self):
        """거래 요약 출력"""
        print(f"\n\n{'='*70}")
        print(f"  TRADING SUMMARY")
        print(f"{'='*70}")

        if self.trade_log:
            total_pnl = sum(t['pnl_pct'] for t in self.trade_log)
            wins = sum(1 for t in self.trade_log if t['pnl_pct'] > 0)
            losses = len(self.trade_log) - wins

            print(f"  Total trades: {len(self.trade_log)}")
            print(f"  Wins: {wins}, Losses: {losses}")
            print(f"  Win rate: {wins/len(self.trade_log)*100:.1f}%")
            print(f"  Total P&L: {total_pnl:+.2f}%")

            print(f"\n  {'Ticker':<8} {'Entry':>10} {'Exit':>10} {'P&L':>8} {'Reason'}")
            print(f"  {'-'*50}")
            for t in self.trade_log:
                print(f"  {t['ticker']:<8} ${t['entry_price']:>8.2f} ${t['exit_price']:>8.2f} "
                      f"{t['pnl_pct']:>+7.2f}% {t['reason']}")
        else:
            print(f"  No closed trades")

        if self.positions:
            print(f"\n  Open positions: {len(self.positions)}")
            for ticker, pos in self.positions.items():
                print(f"    {ticker}: entry=${pos['entry_price']:.2f}")


def main():
    print("\n" + "=" * 70)
    print("Sentirax Scalping Bot v1.0")
    print("=" * 70)
    print("\n  Mode: Paper Trading (Mock)")
    print("  Strategy: Surging stocks scalping")
    print("  TP/SL: Dynamic per model")
    print("  Interval: 1 minute")
    print()

    # 실행 모드 선택
    print("  [1] Single scan (1회 스캔)")
    print("  [2] Continuous mode (연속 실행 - 2시간)")
    print()

    mode = input("  Select mode (1/2): ").strip()

    if mode not in ['1', '2']:
        print("  Invalid selection. Defaulting to single scan.")
        mode = '1'

    confirm = input("  Execute real orders? (yes/no): ").strip().lower()
    execute = confirm == 'yes'

    print(f"\n  Starting...")

    # 봇 초기화
    bot = ScalpingBot(paper_trading=True, account_no="50163140")

    # 모델 로드
    print(f"\n  Loading scalping models...")
    models = bot.load_scalping_models()

    if not models:
        print("  No scalping models found! Run train_scalping_model.py first.")
        return

    print(f"\n  Loaded {len(models)} models")

    if mode == '1':
        bot.run_once(models, execute=execute)
    else:
        bot.run_continuous(models, duration_minutes=120, interval_seconds=60, execute=execute)

    # 로그 저장
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = os.path.join(os.path.dirname(__file__), '..', 'results')
    os.makedirs(results_dir, exist_ok=True)
    log_file = os.path.join(results_dir, f'scalping_log_{timestamp}.txt')

    print(f"\n  Log: {log_file}")


if __name__ == "__main__":
    main()
