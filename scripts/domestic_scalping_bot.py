"""
국내주식 스캘핑 봇

- KOSPI/KOSDAQ 급등주 ML 모델 기반
- 1분봉 실시간 모니터링 (KST 09:00~15:30)
- TP/SL 자동 관리, 포지션 사이징 (KRW 기반)
"""

import sys
import os
import io
import time
import json
import pickle
import platform

if platform.system() == 'Windows':
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
    config_path = os.path.join(PROJECT_ROOT, 'config', 'strategy.json')
    if os.environ.get('GITHUB_ACTIONS'):
        try:
            os.system(f'cd {PROJECT_ROOT} && git pull origin main --quiet 2>/dev/null')
        except Exception:
            pass
    if os.path.exists(config_path):
        with open(config_path) as f:
            return json.load(f)
    return {
        'domestic_scalping': {
            'enabled': True, 'max_positions': 5, 'min_probability': 0.55,
            'order_quantity': 1, 'scan_interval_seconds': 60,
            'tp_override': None, 'sl_override': None,
            'disabled_tickers': [], 'forced_buy_tickers': [], 'only_tickers': []
        },
        'risk': {'max_daily_loss_pct': -10.0, 'stop_all_trading': False}
    }


def create_scalping_features(df: pd.DataFrame) -> pd.DataFrame:
    """스캘핑 Feature Engineering (해외와 동일)"""
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


class DomesticScalpingBot:
    """국내주식 스캘핑 봇 (KRW 기반)"""

    def __init__(self, paper_trading: bool = True, account_no: str = None):
        self.paper_trading = paper_trading
        self.models_dir = os.path.join(PROJECT_ROOT, 'models')

        print("=" * 70)
        print("Sentirax 국내주식 스캘핑 봇 v1.0")
        print("=" * 70)

        self.api = KISTradingAPI(paper_trading=paper_trading)
        if not self.api.authenticate():
            raise Exception("API 인증 실패!")

        if account_no:
            self.api.set_account(account_no, "01")

        self.positions = {}    # {api_ticker: {entry_price, quantity, entry_time, tp, sl, yf_ticker}}
        self.trade_log = []
        self._balance_cache = None
        self.results_dir = os.path.join(PROJECT_ROOT, 'results')
        os.makedirs(self.results_dir, exist_ok=True)

        self._load_checkpoint()
        print("  봇 초기화 완료!")

    def load_kr_models(self) -> dict:
        """국내 스캘핑 모델 로드 (*_kr_scalping.pkl)"""
        models = {}

        for f in os.listdir(self.models_dir):
            if not f.endswith('_kr_scalping.pkl'):
                continue

            model_path = os.path.join(self.models_dir, f)
            with open(model_path, 'rb') as fp:
                model_data = pickle.load(fp)

            api_ticker = model_data.get('api_ticker', f.replace('_kr_scalping.pkl', ''))
            models[api_ticker] = model_data
            perf = model_data['performance']
            params = model_data['params']
            name = model_data.get('name', api_ticker)
            print(f"  [{api_ticker}] {name}: Acc={perf['accuracy']*100:.1f}%, "
                  f"WinR={perf['win_rate']:.1f}%, "
                  f"TP={params['take_profit']}%, SL={params['stop_loss']}%")

        return models

    def get_latest_1min_kr(self, yf_ticker: str, max_retries: int = 3) -> pd.DataFrame:
        """1분봉 데이터 수집 (KS/KQ 종목)"""
        for attempt in range(max_retries):
            try:
                stock = yf.Ticker(yf_ticker)
                df = stock.history(period='1d', interval='1m')
                if df.empty:
                    df = stock.history(period='2d', interval='1m')
                return df
            except Exception as e:
                if attempt < max_retries - 1:
                    time.sleep(2)
                else:
                    print(f"  데이터 오류 {yf_ticker} ({max_retries}회 재시도): {e}")
                    return None

    def _get_total_balance_krw(self, config: dict) -> float:
        """계좌 총 KRW 잔고 (캐시 → API → config 순서)"""
        if self._balance_cache is not None:
            return self._balance_cache

        summary = self.api.get_account_summary_domestic()
        if summary and summary.get('total_krw', 0) > 0:
            self._balance_cache = summary['total_krw']
            print(f"  Balance (API): {self._balance_cache:,.0f}원")
            return self._balance_cache

        alloc = config.get('allocation', {})
        fallback = float(alloc.get('account_balance_krw', 0))
        if fallback > 0:
            self._balance_cache = fallback
            print(f"  Balance (config): {self._balance_cache:,.0f}원")
            return self._balance_cache

        return 0.0

    def _calc_kr_qty(self, price: float, buy_prob: float, config: dict) -> int:
        """국내 스캘핑 매수 수량 계산 (KRW 기반)"""
        alloc = config.get('allocation', {})
        scalping_pct = alloc.get('scalping_pct', 0.70)
        per_trade_pct = alloc.get('scalping_per_trade_pct', 0.40)
        strong_threshold = alloc.get('strong_signal_threshold', 0.70)
        strong_pct = alloc.get('strong_signal_scalping_pct', 0.60)

        if buy_prob >= strong_threshold:
            per_trade_pct = min(strong_pct, 1.0)

        total_krw = self._get_total_balance_krw(config)

        if total_krw > 0 and price > 0:
            budget = total_krw * scalping_pct * per_trade_pct
            qty = max(1, int(budget / price))
            label = "강한" if buy_prob >= strong_threshold else "기본"
            print(f"    [{label}신호] {total_krw:,.0f}원×{scalping_pct:.0%}×{per_trade_pct:.0%}"
                  f"/{price:,.0f}원 = {qty}주")
            return qty

        return config.get('domestic_scalping', {}).get('order_quantity', 1)

    def _save_checkpoint(self):
        checkpoint = {
            'timestamp': datetime.now().isoformat(),
            'positions': {},
            'trade_log': []
        }
        for api_ticker, pos in self.positions.items():
            checkpoint['positions'][api_ticker] = {
                'entry_price': pos['entry_price'],
                'quantity': pos['quantity'],
                'entry_time': pos['entry_time'].isoformat(),
                'tp': pos['tp'],
                'sl': pos['sl'],
                'yf_ticker': pos.get('yf_ticker', ''),
                'name': pos.get('name', api_ticker)
            }
        for trade in self.trade_log:
            checkpoint['trade_log'].append({
                'api_ticker': trade['api_ticker'],
                'name': trade.get('name', ''),
                'entry_price': trade['entry_price'],
                'exit_price': trade['exit_price'],
                'pnl_pct': trade['pnl_pct'],
                'reason': trade['reason'],
                'entry_time': trade['entry_time'].isoformat(),
                'exit_time': trade['exit_time'].isoformat()
            })
        path = os.path.join(self.results_dir, 'domestic_scalping_checkpoint.json')
        with open(path, 'w') as f:
            json.dump(checkpoint, f, indent=2, ensure_ascii=False)

    def _load_checkpoint(self):
        path = os.path.join(self.results_dir, 'domestic_scalping_checkpoint.json')
        if not os.path.exists(path):
            return
        try:
            with open(path) as f:
                checkpoint = json.load(f)
            ts = datetime.fromisoformat(checkpoint['timestamp'])
            if (datetime.now() - ts).total_seconds() > 4 * 3600:
                print("  체크포인트 너무 오래됨, 건너뜀")
                return
            for api_ticker, pos in checkpoint.get('positions', {}).items():
                self.positions[api_ticker] = {
                    'entry_price': pos['entry_price'],
                    'quantity': pos['quantity'],
                    'entry_time': datetime.fromisoformat(pos['entry_time']),
                    'tp': pos['tp'],
                    'sl': pos['sl'],
                    'yf_ticker': pos.get('yf_ticker', ''),
                    'name': pos.get('name', api_ticker)
                }
            for trade in checkpoint.get('trade_log', []):
                self.trade_log.append({
                    'api_ticker': trade['api_ticker'],
                    'name': trade.get('name', ''),
                    'entry_price': trade['entry_price'],
                    'exit_price': trade['exit_price'],
                    'pnl_pct': trade['pnl_pct'],
                    'reason': trade['reason'],
                    'entry_time': datetime.fromisoformat(trade['entry_time']),
                    'exit_time': datetime.fromisoformat(trade['exit_time'])
                })
            if self.positions:
                print(f"  체크포인트 복원: {len(self.positions)}개 포지션")
        except Exception as e:
            print(f"  체크포인트 로드 오류: {e}")

    def _save_daily_summary(self):
        today_str = datetime.now().strftime('%Y%m%d')
        summary = {
            'date': datetime.now().strftime('%Y-%m-%d'),
            'total_trades': len(self.trade_log),
            'open_positions': len(self.positions),
            'trades': []
        }
        if self.trade_log:
            total_pnl = sum(t['pnl_pct'] for t in self.trade_log)
            wins = sum(1 for t in self.trade_log if t['pnl_pct'] > 0)
            summary['total_pnl_pct'] = round(total_pnl, 2)
            summary['win_rate'] = round(wins / len(self.trade_log) * 100, 1)
            for t in self.trade_log:
                summary['trades'].append({
                    'ticker': t['api_ticker'],
                    'name': t.get('name', ''),
                    'pnl_pct': round(t['pnl_pct'], 2),
                    'reason': t['reason']
                })
        path = os.path.join(self.results_dir, f'domestic_scalping_summary_{today_str}.json')
        with open(path, 'w') as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        print(f"\n  일일 요약 저장: {path}")

    def predict_entry(self, api_ticker: str, model_data: dict) -> dict:
        """매수 진입 신호 예측"""
        yf_ticker = model_data.get('yf_ticker', api_ticker + '.KS')
        df = self.get_latest_1min_kr(yf_ticker)
        if df is None or len(df) < 60:
            return None

        model = model_data['model']
        scaler = model_data['scaler']
        feature_names = model_data['feature_names']
        params = model_data['params']

        features = create_scalping_features(df)
        X = features[feature_names]
        valid_idx = X.notna().all(axis=1)
        X_clean = X[valid_idx]

        if len(X_clean) == 0:
            return None

        X_latest = X_clean.iloc[-1:]
        X_scaled = scaler.transform(X_latest)
        prediction = model.predict(X_scaled)[0]
        probability = model.predict_proba(X_scaled)[0]

        current_price = float(df['Close'].iloc[-1])

        return {
            'api_ticker': api_ticker,
            'yf_ticker': yf_ticker,
            'name': model_data.get('name', api_ticker),
            'signal': int(prediction),
            'buy_prob': float(probability[1]) if len(probability) > 1 else 0.0,
            'price': current_price,
            'price_int': int(round(current_price)),
            'tp': params['take_profit'],
            'sl': params['stop_loss'],
            'time': datetime.now()
        }

    def check_positions(self):
        """보유 포지션 TP/SL 체크"""
        closed = []

        for api_ticker, pos in self.positions.items():
            try:
                yf_ticker = pos.get('yf_ticker', api_ticker + '.KS')
                df = yf.Ticker(yf_ticker).history(period='1d', interval='1m')
                if df.empty:
                    continue

                current_price = float(df['Close'].iloc[-1])
                pnl_pct = (current_price / pos['entry_price'] - 1) * 100
                hold_min = (datetime.now() - pos['entry_time']).total_seconds() / 60

                if pnl_pct >= pos['tp']:
                    print(f"  TP HIT {api_ticker}({pos.get('name','')}): +{pnl_pct:.2f}%")
                    self._close_position(api_ticker, int(round(current_price)), 'TP')
                    closed.append(api_ticker)
                elif pnl_pct <= -pos['sl']:
                    print(f"  SL HIT {api_ticker}({pos.get('name','')}): {pnl_pct:.2f}%")
                    self._close_position(api_ticker, int(round(current_price)), 'SL')
                    closed.append(api_ticker)
                elif hold_min >= 60:
                    print(f"  TIMEOUT {api_ticker}: {pnl_pct:+.2f}% ({hold_min:.0f}분)")
                    self._close_position(api_ticker, int(round(current_price)), 'TIMEOUT')
                    closed.append(api_ticker)
                else:
                    print(f"  HOLD {api_ticker}: {pnl_pct:+.2f}% ({hold_min:.0f}분)")

            except Exception as e:
                print(f"  포지션 체크 오류 {api_ticker}: {e}")

        for t in closed:
            del self.positions[t]

    def _close_position(self, api_ticker: str, exit_price: int, reason: str):
        pos = self.positions[api_ticker]
        pnl_pct = (exit_price / pos['entry_price'] - 1) * 100
        self.api.order_sell_domestic(api_ticker, pos['quantity'], price=exit_price)
        self.trade_log.append({
            'api_ticker': api_ticker,
            'name': pos.get('name', api_ticker),
            'entry_price': pos['entry_price'],
            'exit_price': exit_price,
            'pnl_pct': pnl_pct,
            'reason': reason,
            'entry_time': pos['entry_time'],
            'exit_time': datetime.now()
        })
        print(f"  CLOSED {api_ticker}: {pnl_pct:+.2f}% ({reason})")

    def run_scalping_cycle(self, models: dict, execute: bool = False):
        """스캘핑 1사이클 실행"""
        self._balance_cache = None

        config = load_strategy_config()
        scalp_cfg = config.get('domestic_scalping', {})
        risk_cfg = config.get('risk', {})

        if risk_cfg.get('stop_all_trading', False):
            print(f"\n  !! EMERGENCY STOP")
            return

        if not scalp_cfg.get('enabled', True):
            print(f"\n  !! 국내 스캘핑 비활성화됨")
            return

        max_positions = scalp_cfg.get('max_positions', 5)
        min_prob = scalp_cfg.get('min_probability', 0.55)
        tp_override = scalp_cfg.get('tp_override')
        sl_override = scalp_cfg.get('sl_override')
        disabled = [t for t in scalp_cfg.get('disabled_tickers', [])]
        forced_buy = [t for t in scalp_cfg.get('forced_buy_tickers', [])]
        only_tickers = [t for t in scalp_cfg.get('only_tickers', [])]

        print(f"\n{'='*70}")
        print(f"  [국내] 스캘핑 사이클 - {datetime.now().strftime('%H:%M:%S')} KST")
        print(f"  포지션: {len(self.positions)}/{max_positions} | MinProb: {min_prob}")
        print(f"{'='*70}")

        total_pnl = sum(t['pnl_pct'] for t in self.trade_log) if self.trade_log else 0
        max_daily_loss = risk_cfg.get('max_daily_loss_pct', -10.0)
        if total_pnl <= max_daily_loss:
            print(f"  !! 일일 손실 한도 초과: {total_pnl:.2f}%")
            if self.positions:
                self.check_positions()
            return

        if self.positions:
            print(f"\n  [포지션 점검]")
            self.check_positions()

        if len(self.positions) < max_positions:
            print(f"\n  [매수 신호 탐색]")
            scan_models = models
            if only_tickers:
                scan_models = {k: v for k, v in models.items() if k in only_tickers}

            for api_ticker, model_data in scan_models.items():
                if api_ticker in self.positions or api_ticker in disabled:
                    continue
                if len(self.positions) >= max_positions:
                    break

                if api_ticker in forced_buy:
                    print(f"\n  FORCED BUY: {api_ticker}")
                    if execute:
                        result = self.predict_entry(api_ticker, model_data)
                        if result and result['price_int'] > 0:
                            forced_qty = self._calc_kr_qty(result['price'], 0.55, config)
                            order_result = self.api.order_buy_domestic(
                                api_ticker, forced_qty, price=result['price_int'])
                            if order_result:
                                self.positions[api_ticker] = {
                                    'entry_price': result['price_int'],
                                    'quantity': forced_qty,
                                    'entry_time': datetime.now(),
                                    'tp': tp_override or model_data['params']['take_profit'],
                                    'sl': sl_override or model_data['params']['stop_loss'],
                                    'yf_ticker': result['yf_ticker'],
                                    'name': result['name']
                                }
                                print(f"    FORCED ORDER PLACED! ({forced_qty}주)")
                    continue

                result = self.predict_entry(api_ticker, model_data)
                if result is None:
                    continue

                if tp_override:
                    result['tp'] = tp_override
                if sl_override:
                    result['sl'] = sl_override

                if result['signal'] == 1 and result['buy_prob'] >= min_prob:
                    name = result['name']
                    print(f"\n  BUY SIGNAL: {api_ticker}({name})")
                    print(f"    가격: {result['price_int']:,}원")
                    print(f"    확률: {result['buy_prob']*100:.1f}%")
                    print(f"    TP: +{result['tp']}%, SL: -{result['sl']}%")

                    buy_qty = self._calc_kr_qty(result['price'], result['buy_prob'], config)

                    if execute:
                        order_result = self.api.order_buy_domestic(
                            api_ticker, buy_qty, price=result['price_int'])
                        if order_result:
                            self.positions[api_ticker] = {
                                'entry_price': result['price_int'],
                                'quantity': buy_qty,
                                'entry_time': datetime.now(),
                                'tp': result['tp'],
                                'sl': result['sl'],
                                'yf_ticker': result['yf_ticker'],
                                'name': result['name']
                            }
                            print(f"    ORDER PLACED! ({buy_qty}주)")
                        else:
                            print(f"    ORDER FAILED")
                    else:
                        self.positions[api_ticker] = {
                            'entry_price': result['price_int'],
                            'quantity': buy_qty,
                            'entry_time': datetime.now(),
                            'tp': result['tp'],
                            'sl': result['sl'],
                            'yf_ticker': result['yf_ticker'],
                            'name': result['name']
                        }
                        print(f"    [SIMULATED] ({buy_qty}주)")
                else:
                    sig = "BUY" if result['signal'] == 1 else "NO"
                    print(f"  {api_ticker}: {sig} (prob={result['buy_prob']*100:.1f}%)")

    def run_continuous(self, models: dict, duration_minutes: int = 120,
                       interval_seconds: int = 60, execute: bool = False):
        """연속 스캘핑 실행"""
        print(f"\n{'='*70}")
        print(f"  [국내] 연속 스캘핑 모드")
        print(f"  실행시간: {duration_minutes}분 / 간격: {interval_seconds}초")
        print(f"  실행: {'YES' if execute else 'NO (시뮬레이션)'}")
        print(f"  모델: {len(models)}개")
        print(f"{'='*70}")

        start_time = datetime.now()
        end_time = start_time + timedelta(minutes=duration_minutes)
        cycle = 0

        self._save_checkpoint()

        try:
            while datetime.now() < end_time:
                cycle += 1
                live_config = load_strategy_config()
                risk_cfg = live_config.get('risk', {})

                if risk_cfg.get('stop_all_trading', False):
                    print(f"\n  !! EMERGENCY STOP. 모든 포지션 청산...")
                    for api_ticker in list(self.positions.keys()):
                        try:
                            pos = self.positions[api_ticker]
                            yf_ticker = pos.get('yf_ticker', api_ticker + '.KS')
                            df = yf.Ticker(yf_ticker).history(period='1d', interval='1m')
                            if not df.empty:
                                price = int(round(float(df['Close'].iloc[-1])))
                                self._close_position(api_ticker, price, 'EMERGENCY')
                        except Exception:
                            pass
                    self.positions.clear()
                    break

                live_interval = live_config.get('domestic_scalping', {}).get(
                    'scan_interval_seconds', interval_seconds)

                self.run_scalping_cycle(models, execute=execute)
                self._save_checkpoint()

                remaining = (end_time - datetime.now()).total_seconds() / 60
                print(f"\n  Cycle #{cycle} | 남은시간: {remaining:.1f}분 | "
                      f"다음: {live_interval}초 후...")

                if datetime.now() < end_time:
                    time.sleep(live_interval)

        except KeyboardInterrupt:
            print("\n\n  사용자 중단 (Ctrl+C)")

        self._print_summary()
        self._save_checkpoint()
        self._save_daily_summary()

    def _print_summary(self):
        print(f"\n\n{'='*70}")
        print(f"  [국내] 거래 요약")
        print(f"{'='*70}")

        if self.trade_log:
            total_pnl = sum(t['pnl_pct'] for t in self.trade_log)
            wins = sum(1 for t in self.trade_log if t['pnl_pct'] > 0)
            print(f"  총 거래: {len(self.trade_log)}건 | 승: {wins} | 패: {len(self.trade_log)-wins}")
            print(f"  승률: {wins/len(self.trade_log)*100:.1f}% | 총 P&L: {total_pnl:+.2f}%")
            print(f"\n  {'종목':8s} {'매수가':>10} {'매도가':>10} {'P&L':>8} {'사유'}")
            print(f"  {'-'*50}")
            for t in self.trade_log:
                print(f"  {t['api_ticker']:<8} {t['entry_price']:>10,} {t['exit_price']:>10,} "
                      f"{t['pnl_pct']:>+7.2f}% {t['reason']}")
        else:
            print(f"  청산된 거래 없음")

        if self.positions:
            print(f"\n  미청산 포지션: {len(self.positions)}개")
            for api_ticker, pos in self.positions.items():
                print(f"    {api_ticker}({pos.get('name','')}): {pos['entry_price']:,}원 x {pos['quantity']}주")
