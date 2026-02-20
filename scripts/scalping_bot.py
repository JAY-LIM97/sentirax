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
import platform

if platform.system() == 'Windows':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta, timezone

from core.kis_trading_api import KISTradingAPI
from core.scalping_signals import create_scalping_features, compute_rule_signals
from core.slack_notifier import (
    notify_trade_open, notify_trade_close, notify_daily_summary,
)

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
        self._balance_cache = None  # 사이클별 잔고 캐시 (매 사이클 초기화)
        self.results_dir = os.path.join(os.path.dirname(__file__), '..', 'results')
        os.makedirs(self.results_dir, exist_ok=True)

        # 체크포인트에서 포지션 복원 시도
        self._load_checkpoint()

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

    def get_latest_1min_data(self, ticker: str, max_retries: int = 3) -> pd.DataFrame:
        """최신 1분봉 데이터 수집 (재시도 포함)"""
        for attempt in range(max_retries):
            try:
                stock = yf.Ticker(ticker)
                df = stock.history(period='1d', interval='1m')
                if df.empty:
                    df = stock.history(period='2d', interval='1m')
                return df
            except Exception as e:
                if attempt < max_retries - 1:
                    time.sleep(2)
                else:
                    print(f"  Data error {ticker} (after {max_retries} retries): {e}")
                    return None

    def _save_checkpoint(self):
        """현재 포지션 + 거래 기록 체크포인트 저장"""
        checkpoint = {
            'timestamp': datetime.now().isoformat(),
            'positions': {},
            'trade_log': []
        }
        for ticker, pos in self.positions.items():
            checkpoint['positions'][ticker] = {
                'entry_price': pos['entry_price'],
                'quantity': pos['quantity'],
                'entry_time': pos['entry_time'].isoformat(),
                'tp': pos['tp'],
                'sl': pos['sl'],
                'step': pos.get('step', 2),
            }
        for trade in self.trade_log:
            checkpoint['trade_log'].append({
                'ticker': trade['ticker'],
                'entry_price': trade['entry_price'],
                'exit_price': trade['exit_price'],
                'pnl_pct': trade['pnl_pct'],
                'reason': trade['reason'],
                'entry_time': trade['entry_time'].isoformat(),
                'exit_time': trade['exit_time'].isoformat()
            })
        path = os.path.join(self.results_dir, 'scalping_checkpoint.json')
        with open(path, 'w') as f:
            json.dump(checkpoint, f, indent=2)

    def _load_checkpoint(self):
        """이전 체크포인트에서 포지션 복원"""
        path = os.path.join(self.results_dir, 'scalping_checkpoint.json')
        if not os.path.exists(path):
            return
        try:
            with open(path) as f:
                checkpoint = json.load(f)
            # 4시간 이내 체크포인트만 복원 (오래된 건 무시)
            ts = datetime.fromisoformat(checkpoint['timestamp'])
            if (datetime.now() - ts).total_seconds() > 4 * 3600:
                print("  Checkpoint too old, skipping restore")
                return
            for ticker, pos in checkpoint.get('positions', {}).items():
                self.positions[ticker] = {
                    'entry_price': pos['entry_price'],
                    'quantity': pos['quantity'],
                    'entry_time': datetime.fromisoformat(pos['entry_time']),
                    'tp': pos['tp'],
                    'sl': pos['sl'],
                    'step': pos.get('step', 2),
                }
            for trade in checkpoint.get('trade_log', []):
                self.trade_log.append({
                    'ticker': trade['ticker'],
                    'entry_price': trade['entry_price'],
                    'exit_price': trade['exit_price'],
                    'pnl_pct': trade['pnl_pct'],
                    'reason': trade['reason'],
                    'entry_time': datetime.fromisoformat(trade['entry_time']),
                    'exit_time': datetime.fromisoformat(trade['exit_time'])
                })
            if self.positions:
                print(f"  Restored {len(self.positions)} positions from checkpoint")
            if self.trade_log:
                print(f"  Restored {len(self.trade_log)} trade records from checkpoint")
        except Exception as e:
            print(f"  Checkpoint load error: {e}")

    def _get_total_balance_usd(self, config: dict) -> float:
        """계좌 총 USD 잔고 조회 (API 우선, 실패 시 설정값 fallback + 경고)"""
        if self._balance_cache is not None:
            return self._balance_cache

        summary = self.api.get_account_summary()
        if summary and summary.get('total_usd', 0) > 0:
            self._balance_cache = summary['total_usd']
            print(f"  [Balance] API 조회 성공: ${self._balance_cache:,.0f}")
            return self._balance_cache

        # API 실패 시 config 설정값 fallback — 경고 출력
        alloc = config.get('allocation', {})
        fallback = float(alloc.get('account_balance_usd', 0))
        if fallback > 0:
            print(f"  ⚠️  [Balance] API 조회 실패! config fallback ${fallback:,.0f} 사용")
            print(f"  ⚠️  실제 잔고와 다를 수 있음. API 키/토큰/계좌번호 확인 필요!")
            self._balance_cache = fallback
            return self._balance_cache

        print(f"  ❌ [Balance] 잔고 조회 완전 실패 — 매수 차단")
        return 0.0

    def _calc_step_qty(self, price: float, step: int, config: dict) -> int:
        """
        2단계 진입 수량 계산 (종목당 최대 전체 자산 12%)
          step 1: 스캘핑 자산(60%) × 10% = 전체 자산의 6%
          step 2: 동일 (수익 1% 이상일 때만 허용)
        """
        alloc = config.get('allocation', {})
        scalping_pct = alloc.get('scalping_pct', 0.60)
        step_pct = alloc.get('scalping_step_pct', 0.10)

        total_usd = self._get_total_balance_usd(config)
        if total_usd > 0 and price > 0:
            budget = total_usd * scalping_pct * step_pct
            qty = max(1, int(budget / price))
            eff_pct = scalping_pct * step_pct * 100
            print(f"    [Step{step}] ${total_usd:,.0f}×{scalping_pct:.0%}×{step_pct:.0%}"
                  f"(={eff_pct:.0f}%total)/${price:.2f} = {qty}주")
            return qty
        return config.get('scalping', {}).get('order_quantity', 1)

    def _calc_scalping_qty(self, price: float, _buy_prob: float, config: dict) -> int:
        """강제매수 등 레거시 호출용 (step1 수량 반환)"""
        return self._calc_step_qty(price, 1, config)

    def _select_top_volume_tickers(self, models: dict, max_n: int = 5) -> dict:
        """
        실시간 거래량 기준 상위 N개 종목 선정 (다이나믹 스캐너)
        fast_info.last_volume ÷ three_month_average_volume 비율 기준 정렬
        거래량 급감 종목은 자동 제외, 대기 종목으로 교체
        """
        volume_scores: dict[str, float] = {}
        for ticker in models:
            try:
                stock = yf.Ticker(ticker)
                fi = stock.fast_info
                today_vol = getattr(fi, 'last_volume', None) or 0
                avg_vol = getattr(fi, 'three_month_average_volume', None) or today_vol or 1
                volume_scores[ticker] = today_vol / avg_vol if avg_vol > 0 else 0.0
            except Exception:
                volume_scores[ticker] = 0.0

        sorted_tickers = sorted(volume_scores, key=lambda t: volume_scores[t], reverse=True)
        top = sorted_tickers[:max_n]

        print(f"\n  [Volume Scanner] Top {max_n} targets (of {len(models)}):")
        for i, t in enumerate(top):
            print(f"    {i+1}. {t}: ×{volume_scores.get(t, 0):.2f}")

        return {t: models[t] for t in top if t in models}

    def _save_daily_summary(self):
        """일일 스캘핑 요약 저장"""
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
            summary['win_count'] = wins
            summary['loss_count'] = len(self.trade_log) - wins
            summary['win_rate'] = round(wins / len(self.trade_log) * 100, 1)
            for t in self.trade_log:
                summary['trades'].append({
                    'ticker': t['ticker'],
                    'pnl_pct': round(t['pnl_pct'], 2),
                    'reason': t['reason']
                })
        path = os.path.join(self.results_dir, f'scalping_summary_{today_str}.json')
        with open(path, 'w') as f:
            json.dump(summary, f, indent=2)
        print(f"\n  Daily summary saved: {path}")

    def predict_entry(self, ticker: str, model_data: dict) -> dict:
        """
        매수 진입 신호 예측 (ML + Rule 앙상블)

        진입 조건 (OR):
          A) ML 신호(1) AND ML확률 >= min_prob  [기본]
          B) ML확률 >= min_prob-0.10 AND 룰점수 >= 2 AND 거짓돌파 아님  [룰 보조]
          C) 룰점수 == 3 AND ML확률 >= 0.40  [3전략 모두 일치]
        """
        df = self.get_latest_1min_data(ticker)
        if df is None or len(df) < 60:
            return None

        model        = model_data['model']
        scaler       = model_data['scaler']
        feature_names = model_data['feature_names']
        params       = model_data['params']

        # ML Feature — 기존 모델이 저장한 피처명만 사용 (하위 호환)
        features = create_scalping_features(df)
        # 모델에 없는 새 피처는 자동 무시
        available = [f for f in feature_names if f in features.columns]
        X = features[available]
        valid_idx = X.notna().all(axis=1)
        X_clean = X[valid_idx]

        if len(X_clean) == 0:
            return None

        X_latest = X_clean.iloc[-1:]
        X_scaled = scaler.transform(X_latest)

        prediction  = model.predict(X_scaled)[0]
        probability = model.predict_proba(X_scaled)[0]
        buy_prob    = float(probability[1]) if len(probability) > 1 else 0.0

        current_price = float(df['Close'].iloc[-1])

        # ── Rule-based 신호 (3전략) ──────────────────────────────────────────
        rule = compute_rule_signals(df)
        score = rule['signal_score']

        # 룰 신호가 2개 이상이면 동적 TP/SL 사용, 아니면 모델 기본값
        if score >= 2:
            tp = rule['dynamic_tp']
            sl = rule['dynamic_sl']
        else:
            tp = params['take_profit']
            sl = params['stop_loss']

        # 룰 점수만큼 ML 확률 보정 (최대 +30%)
        buy_prob_adj = min(buy_prob * (1 + 0.10 * score), 0.99)

        return {
            'ticker':        ticker,
            'signal':        int(prediction),
            'buy_prob':      buy_prob,
            'buy_prob_adj':  buy_prob_adj,
            'price':         current_price,
            'tp':            tp,
            'sl':            sl,
            's1':            rule['s1'],
            's2':            rule['s2'],
            's3':            rule['s3'],
            'signal_score':  score,
            'breakdown_risk': rule['breakdown_risk'],
            'time':          datetime.now()
        }

    def check_positions(self, execute: bool = True):
        """보유 포지션 TP/SL + Step-2 추가매수 체크"""

        config = load_strategy_config()
        alloc = config.get('allocation', {})
        step2_min_profit = alloc.get('scalping_step2_min_profit_pct', 1.0)

        closed_tickers = []

        for ticker, pos in self.positions.items():
            try:
                stock = yf.Ticker(ticker)
                df = stock.history(period='1d', interval='1m')
                if df.empty:
                    continue

                current_price = float(df['Close'].iloc[-1])
                pnl_pct = (current_price / pos['entry_price'] - 1) * 100
                hold_minutes = (datetime.now() - pos['entry_time']).total_seconds() / 60

                # TP/SL 체크
                if pnl_pct >= pos['tp']:
                    print(f"  TP HIT {ticker}: +{pnl_pct:.2f}% (target: +{pos['tp']}%)")
                    self._close_position(ticker, current_price, 'TP', execute=execute)
                    closed_tickers.append(ticker)

                elif pnl_pct <= -pos['sl']:
                    print(f"  SL HIT {ticker}: {pnl_pct:.2f}% (limit: -{pos['sl']}%)")
                    self._close_position(ticker, current_price, 'SL', execute=execute)
                    closed_tickers.append(ticker)

                elif hold_minutes >= 60:
                    print(f"  TIMEOUT {ticker}: {pnl_pct:+.2f}% after {hold_minutes:.0f}min")
                    self._close_position(ticker, current_price, 'TIMEOUT', execute=execute)
                    closed_tickers.append(ticker)

                elif pos.get('step', 2) == 1 and pnl_pct >= step2_min_profit:
                    # Step-2 추가매수 (수익 1% 이상 + 아직 1차 매수만 한 상태)
                    add_qty = self._calc_step_qty(current_price, 2, config)
                    print(f"  STEP-2 ADD {ticker}: +{pnl_pct:.2f}% >= {step2_min_profit}% → +{add_qty}주")
                    order_ok = True
                    if execute:
                        order_result = self.api.order_buy(ticker, add_qty, price=current_price)
                        order_ok = bool(order_result)
                        if order_ok:
                            print(f"    ✅ Step-2 매수 성공")
                        else:
                            print(f"    ❌ Step-2 매수 실패")
                    if order_ok:
                        total_qty = pos['quantity'] + add_qty
                        avg_price = (pos['entry_price'] * pos['quantity'] +
                                     current_price * add_qty) / total_qty
                        self.positions[ticker]['quantity'] = total_qty
                        self.positions[ticker]['entry_price'] = avg_price
                        self.positions[ticker]['step'] = 2
                        print(f"    avg=${avg_price:.2f}, total={total_qty}주 (12%까지 사용)")

                else:
                    step_txt = f"[S{pos.get('step',2)}]" if pos.get('step',2) == 1 else ""
                    print(f"  HOLD {ticker}{step_txt}: {pnl_pct:+.2f}% ({hold_minutes:.0f}min)")

            except Exception as e:
                print(f"  Error checking {ticker}: {e}")

        for t in closed_tickers:
            del self.positions[t]

    def _close_position(self, ticker: str, exit_price: float, reason: str, execute: bool = True):
        """포지션 종료 (매도) — execute=False 시 시뮬레이션만"""
        pos = self.positions[ticker]
        qty = pos['quantity']
        pnl_pct = (exit_price / pos['entry_price'] - 1) * 100
        hold_min = (datetime.now() - pos['entry_time']).total_seconds() / 60

        print(f"  [CLOSE] {ticker}: {pnl_pct:+.2f}% ({reason}) | "
              f"진입 ${pos['entry_price']:.2f} → 청산 ${exit_price:.2f} | {qty}주")

        if execute:
            # ── 실제 보유수량 API 검증 ──────────────────────────────────
            actual_qty = self.api.get_holding_qty(ticker)
            if actual_qty == 0:
                print(f"    ⚠️  실제 보유수량 0주 확인 — 매도 주문 취소 (잔고 없음)")
            else:
                if actual_qty < 0:
                    # API 조회 실패 → 내부 기록값 사용
                    print(f"    ⚠️  보유수량 API 조회 실패 — 내부 기록({qty}주)으로 매도 진행")
                    actual_qty = qty
                elif actual_qty != qty:
                    print(f"    ⚠️  수량 불일치: 내부={qty}주 vs API={actual_qty}주 → API 값 사용")
                    qty = actual_qty

                order_result = self.api.order_sell(ticker, qty, price=exit_price)
                if order_result:
                    print(f"    ✅ 매도 주문 성공 — {qty}주 @ ${exit_price:.2f}")
                else:
                    print(f"    ❌ 매도 주문 실패 — 수동 확인 필요! ({ticker} {qty}주)")
        else:
            print(f"    [SIMULATED] 매도 {qty}주 @ ${exit_price:.2f}")

        self.trade_log.append({
            'ticker': ticker,
            'entry_price': pos['entry_price'],
            'exit_price': exit_price,
            'pnl_pct': pnl_pct,
            'reason': reason,
            'entry_time': pos['entry_time'],
            'exit_time': datetime.now()
        })
        notify_trade_close(
            ticker, pos['entry_price'], exit_price,
            pnl_pct, reason, hold_min,
            market="US", paper_trading=self.paper_trading,
        )

    def _force_close_all(self, reason: str = 'MARKET_CLOSE', execute: bool = True):
        """장 마감 전 모든 포지션 시장가 강제 청산"""
        if not self.positions:
            print(f"  [{reason}] 청산할 포지션 없음")
            return
        print(f"\n{'='*70}")
        print(f"  [{reason}] 전체 {len(self.positions)}개 포지션 강제 청산 중...")
        print(f"{'='*70}")
        for ticker in list(self.positions.keys()):
            try:
                df = yf.Ticker(ticker).history(period='1d', interval='1m')
                price = float(df['Close'].iloc[-1]) if not df.empty else self.positions[ticker]['entry_price']
                self._close_position(ticker, price, reason, execute=execute)
            except Exception as e:
                print(f"  {ticker} 강제 청산 오류: {e}")
        self.positions.clear()

    def run_scalping_cycle(self, models: dict, max_positions: int = 5, execute: bool = False):
        """
        스캘핑 1사이클 실행 (매 사이클마다 strategy.json 실시간 반영)
        """

        # 매 사이클마다 잔고 캐시 리셋 (최신 잔고 반영)
        self._balance_cache = None

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

        # 1. 보유 포지션 체크 (TP/SL + Step-2)
        if self.positions:
            print(f"\n  [Position Check]")
            self.check_positions(execute=execute)

        # 2. 새 매수 신호 탐색
        if len(self.positions) < max_positions:
            print(f"\n  [Scanning for entries]")

            # 대상 모델 필터링
            if only_tickers:
                scan_models = {k: v for k, v in models.items() if k in only_tickers}
            else:
                # 볼륨 기반 상위 N개 다이나믹 선정
                max_targets = config.get('allocation', {}).get('scalping_max_volume_targets', max_positions)
                scan_models = self._select_top_volume_tickers(models, max_n=max_targets)

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
                        result = self.predict_entry(ticker, model_data)
                        current_price = result['price'] if result else 0
                        if current_price <= 0:
                            print(f"    Failed to get price, skipping")
                            continue
                        forced_qty = self._calc_scalping_qty(current_price, 0.55, config)
                        order_result = self.api.order_buy(ticker, forced_qty, price=current_price)
                        if order_result:
                            self.positions[ticker] = {
                                'entry_price': current_price,
                                'quantity': forced_qty,
                                'entry_time': datetime.now(),
                                'tp': tp_override or model_data['params']['take_profit'],
                                'sl': sl_override or model_data['params']['stop_loss']
                            }
                            print(f"    FORCED ORDER PLACED! ({forced_qty}주)")
                    continue

                result = self.predict_entry(ticker, model_data)
                if result is None:
                    continue

                # TP/SL 오버라이드
                if tp_override:
                    result['tp'] = tp_override
                if sl_override:
                    result['sl'] = sl_override

                # ── 진입 조건 (ML + Rule 앙상블) ────────────────────────────
                score = result['signal_score']
                prob  = result['buy_prob']
                prob_adj = result['buy_prob_adj']

                # A) ML 기본: 신호=1 AND 확률 충족
                ml_ok   = result['signal'] == 1 and prob >= min_prob
                # B) 룰 보조: 룰 2개+ AND 확률 10%p 완화 AND 거짓돌파 아님
                rule_ok = score >= 2 and prob >= max(min_prob - 0.10, 0.40) and not result['breakdown_risk']
                # C) 룰 3개 일치: 확률 0.40 이상이면 진입
                rule_strong = score == 3 and prob >= 0.40 and not result['breakdown_risk']

                enter = ml_ok or rule_ok or rule_strong

                if enter:
                    # 진입 사유 태그
                    tags = []
                    if ml_ok:   tags.append("ML")
                    if rule_ok or rule_strong: tags.append(f"RULE({score})")
                    if result['s1']: tags.append("S1-Box")
                    if result['s2']: tags.append("S2-EMA")
                    if result['s3']: tags.append("S3-Liq")

                    print(f"\n  BUY SIGNAL: {ticker} [{' '.join(tags)}]")
                    print(f"    Price: ${result['price']:.2f}")
                    print(f"    ML prob: {prob*100:.1f}% → adj: {prob_adj*100:.1f}%")
                    print(f"    TP: +{result['tp']}%, SL: -{result['sl']}%")

                    # Step-1 매수: 전체 자산의 6% (스캘핑60%×10%)
                    buy_qty = self._calc_step_qty(result['price'], 1, config)

                    if execute:
                        order_result = self.api.order_buy(ticker, buy_qty, price=result['price'])
                        if order_result:
                            self.positions[ticker] = {
                                'entry_price': result['price'],
                                'quantity':    buy_qty,
                                'entry_time':  datetime.now(),
                                'tp':          result['tp'],
                                'sl':          result['sl'],
                                'step':        1,   # 2차 매수 대기
                            }
                            print(f"    ORDER PLACED! Step-1 ({buy_qty}주, 6%total)")
                            notify_trade_open(
                                ticker, result['price'], buy_qty,
                                result['tp'], result['sl'], tags,
                                market="US", paper_trading=self.paper_trading,
                            )
                        else:
                            print(f"    ORDER FAILED")
                    else:
                        self.positions[ticker] = {
                            'entry_price': result['price'],
                            'quantity':    buy_qty,
                            'entry_time':  datetime.now(),
                            'tp':          result['tp'],
                            'sl':          result['sl'],
                            'step':        1,
                        }
                        print(f"    [SIMULATED] Step-1 ({buy_qty}주)")

                else:
                    strats = f"S1={result['s1']} S2={result['s2']} S3={result['s3']}"
                    sig_txt = "ML" if result['signal'] == 1 else "NO"
                    print(f"  {ticker}: {sig_txt} prob={prob*100:.1f}% rule={score} | {strats}")

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

        # 시작 시 즉시 체크포인트 생성 (Phase 2 artifact 전달 보장)
        self._save_checkpoint()

        # US 마감 강제 청산 기준: NYSE 종료 21:00 UTC의 10분 전 = 20:50 UTC
        US_FORCE_CLOSE_HOUR = 20
        US_FORCE_CLOSE_MIN = 50

        try:
            while datetime.now() < end_time:
                cycle += 1

                # 장 마감 10분 전 강제 청산 (오버나잇 포지션 방지)
                now_utc = datetime.now(timezone.utc)
                if (now_utc.hour > US_FORCE_CLOSE_HOUR or
                        (now_utc.hour == US_FORCE_CLOSE_HOUR and
                         now_utc.minute >= US_FORCE_CLOSE_MIN)):
                    print(f"\n  [MARKET CLOSE] UTC {now_utc.strftime('%H:%M')} - 마감 10분 전 전체 청산")
                    self._force_close_all('MARKET_CLOSE', execute=execute)
                    break

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

                # 매 사이클마다 체크포인트 저장
                self._save_checkpoint()

                remaining = (end_time - datetime.now()).total_seconds() / 60
                print(f"\n  Cycle #{cycle} | Remaining: {remaining:.1f}min | "
                      f"Next in {live_interval}s...")

                if datetime.now() < end_time:
                    time.sleep(live_interval)

        except KeyboardInterrupt:
            print("\n\n  STOPPED by user (Ctrl+C)")

        # 최종 요약 + 일일 리포트 저장
        self._print_summary()
        self._save_checkpoint()
        self._save_daily_summary()
        notify_daily_summary(
            self.trade_log, self.positions,
            market="US", paper_trading=self.paper_trading,
        )

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
            config = load_strategy_config()
            min_prob = config.get('scalping', {}).get('min_probability', 0.55)
            print(f"\n  Executing {len(buy_signals)} buy orders...")
            for sig in buy_signals:
                if sig['buy_prob'] >= min_prob:
                    buy_qty = self._calc_scalping_qty(sig['price'], sig['buy_prob'], config)
                    result = self.api.order_buy(sig['ticker'], buy_qty, price=sig['price'])
                    status = "OK" if result else "FAIL"
                    print(f"  ORDER {sig['ticker']}: {status} ({buy_qty}주 @ ${sig['price']:.2f})")

        # 요약
        print(f"\n{'='*70}")
        print(f"  RESULTS")
        print(f"{'='*70}")
        print(f"  BUY signals: {len(buy_signals)}")
        for s in buy_signals:
            print(f"    {s['ticker']}: ${s['price']:.2f} (prob={s['buy_prob']*100:.1f}%)")
        print(f"  NO signals:  {len(no_signals)}")

        # 체크포인트 저장
        self._save_checkpoint()

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
