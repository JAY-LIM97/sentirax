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
import yfinance as yf
from datetime import datetime, timedelta, timezone

from core.kis_trading_api import KISTradingAPI
from core.scalping_signals import create_scalping_features, compute_rule_signals
from core.slack_notifier import (
    notify_trade_open, notify_trade_close, notify_daily_summary,
)

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
        """계좌 총 KRW 잔고 조회 (API 우선, 실패 시 설정값 fallback + 경고)"""
        if self._balance_cache is not None:
            return self._balance_cache

        summary = self.api.get_account_summary_domestic()
        if summary and summary.get('total_krw', 0) > 0:
            self._balance_cache = summary['total_krw']
            print(f"  [Balance] API 조회 성공: {self._balance_cache:,.0f}원")
            return self._balance_cache

        # API 실패 시 config 설정값 fallback — 경고 출력
        alloc = config.get('allocation', {})
        fallback = float(alloc.get('account_balance_krw', 0))
        if fallback > 0:
            print(f"  ⚠️  [Balance] API 조회 실패! config fallback {fallback:,.0f}원 사용")
            print(f"  ⚠️  실제 잔고와 다를 수 있음. API 키/토큰/계좌번호 확인 필요!")
            self._balance_cache = fallback
            return self._balance_cache

        print(f"  ❌ [Balance] 잔고 조회 완전 실패 — 매수 차단")
        return 0.0

    def _calc_step_kr_qty(self, price: float, step: int, config: dict) -> int:
        """
        2단계 진입 수량 계산 (KRW, 종목당 최대 전체 자산 12%)
          step 1: 스캘핑 자산(60%) × 10% = 전체 자산의 6%
          step 2: 동일 (수익 1% 이상일 때만 허용)
        """
        alloc = config.get('allocation', {})
        scalping_pct = alloc.get('scalping_pct', 0.60)
        step_pct = alloc.get('scalping_step_pct', 0.10)

        total_krw = self._get_total_balance_krw(config)
        if total_krw > 0 and price > 0:
            budget = total_krw * scalping_pct * step_pct
            qty = max(1, int(budget / price))
            eff_pct = scalping_pct * step_pct * 100
            print(f"    [Step{step}] {total_krw:,.0f}원×{scalping_pct:.0%}×{step_pct:.0%}"
                  f"(={eff_pct:.0f}%total)/{price:,.0f}원 = {qty}주")
            return qty
        return config.get('domestic_scalping', {}).get('order_quantity', 1)

    def _calc_kr_qty(self, price: float, _buy_prob: float, config: dict) -> int:
        """레거시 호출용 (step1 수량 반환)"""
        return self._calc_step_kr_qty(price, 1, config)

    def _select_top_volume_kr_tickers(self, models: dict, max_n: int = 5) -> dict:
        """
        실시간 거래량 기준 상위 N개 국내 종목 선정 (다이나믹 스캐너)
        yfinance fast_info 기반으로 빠르게 거래량 비율 계산
        """
        volume_scores: dict[str, float] = {}
        for api_ticker, model_data in models.items():
            yf_ticker = model_data.get('yf_ticker', api_ticker + '.KS')
            try:
                fi = yf.Ticker(yf_ticker).fast_info
                today_vol = getattr(fi, 'last_volume', None) or 0
                avg_vol = getattr(fi, 'three_month_average_volume', None) or today_vol or 1
                volume_scores[api_ticker] = today_vol / avg_vol if avg_vol > 0 else 0.0
            except Exception:
                volume_scores[api_ticker] = 0.0

        sorted_tickers = sorted(volume_scores, key=lambda t: volume_scores[t], reverse=True)
        top = sorted_tickers[:max_n]

        print(f"\n  [볼륨 스캐너] 상위 {max_n}개 타겟 (전체 {len(models)}개 중):")
        for i, t in enumerate(top):
            name = models[t].get('name', t) if t in models else t
            print(f"    {i+1}. {t}({name}): ×{volume_scores.get(t, 0):.2f}")

        return {t: models[t] for t in top if t in models}

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
                'name': pos.get('name', api_ticker),
                'step': pos.get('step', 2),
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
                    'name': pos.get('name', api_ticker),
                    'step': pos.get('step', 2),
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
        """
        매수 진입 신호 예측 (ML + Rule 앙상블)

        진입 조건 (OR):
          A) ML 신호(1) AND ML확률 >= min_prob  [기본]
          B) ML확률 >= min_prob-0.10 AND 룰점수 >= 2 AND 거짓돌파 아님  [룰 보조]
          C) 룰점수 == 3 AND ML확률 >= 0.40  [3전략 모두 일치]
        """
        yf_ticker = model_data.get('yf_ticker', api_ticker + '.KS')
        df = self.get_latest_1min_kr(yf_ticker)
        if df is None or len(df) < 60:
            return None

        model         = model_data['model']
        scaler        = model_data['scaler']
        feature_names = model_data['feature_names']
        params        = model_data['params']

        # ML Feature — 기존 모델이 저장한 피처명만 사용 (하위 호환)
        features  = create_scalping_features(df)
        available = [f for f in feature_names if f in features.columns]
        X         = features[available]
        valid_idx = X.notna().all(axis=1)
        X_clean   = X[valid_idx]

        if len(X_clean) == 0:
            return None

        X_latest    = X_clean.iloc[-1:]
        X_scaled    = scaler.transform(X_latest)
        prediction  = model.predict(X_scaled)[0]
        probability = model.predict_proba(X_scaled)[0]
        buy_prob    = float(probability[1]) if len(probability) > 1 else 0.0

        current_price = float(df['Close'].iloc[-1])

        # ── Rule-based 신호 (3전략) ──────────────────────────────────────────
        rule  = compute_rule_signals(df)
        score = rule['signal_score']

        if score >= 2:
            tp = rule['dynamic_tp']
            sl = rule['dynamic_sl']
        else:
            tp = params['take_profit']
            sl = params['stop_loss']

        buy_prob_adj = min(buy_prob * (1 + 0.10 * score), 0.99)

        return {
            'api_ticker':    api_ticker,
            'yf_ticker':     yf_ticker,
            'name':          model_data.get('name', api_ticker),
            'signal':        int(prediction),
            'buy_prob':      buy_prob,
            'buy_prob_adj':  buy_prob_adj,
            'price':         current_price,
            'price_int':     int(round(current_price)),
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

        closed = []

        for api_ticker, pos in self.positions.items():
            try:
                yf_ticker = pos.get('yf_ticker', api_ticker + '.KS')
                df = yf.Ticker(yf_ticker).history(period='1d', interval='1m')
                if df.empty:
                    continue

                current_price = float(df['Close'].iloc[-1])
                current_price_int = int(round(current_price))
                pnl_pct = (current_price / pos['entry_price'] - 1) * 100
                hold_min = (datetime.now() - pos['entry_time']).total_seconds() / 60

                if pnl_pct >= pos['tp']:
                    print(f"  TP HIT {api_ticker}({pos.get('name','')}): +{pnl_pct:.2f}%")
                    self._close_position(api_ticker, current_price_int, 'TP', execute=execute)
                    closed.append(api_ticker)
                elif pnl_pct <= -pos['sl']:
                    print(f"  SL HIT {api_ticker}({pos.get('name','')}): {pnl_pct:.2f}%")
                    self._close_position(api_ticker, current_price_int, 'SL', execute=execute)
                    closed.append(api_ticker)
                elif hold_min >= 60:
                    print(f"  TIMEOUT {api_ticker}: {pnl_pct:+.2f}% ({hold_min:.0f}분)")
                    self._close_position(api_ticker, current_price_int, 'TIMEOUT', execute=execute)
                    closed.append(api_ticker)
                elif pos.get('step', 2) == 1 and pnl_pct >= step2_min_profit:
                    # Step-2 추가매수 (수익 1% 이상 + 1차 매수 상태)
                    add_qty = self._calc_step_kr_qty(current_price, 2, config)
                    print(f"  STEP-2 ADD {api_ticker}: +{pnl_pct:.2f}% → +{add_qty}주 추가")
                    order_ok = True
                    if execute:
                        order_result = self.api.order_buy_domestic(
                            api_ticker, add_qty, price=current_price_int)
                        order_ok = bool(order_result)
                        if order_ok:
                            print(f"    ✅ Step-2 매수 성공")
                        else:
                            print(f"    ❌ Step-2 매수 실패")
                    if order_ok:
                        total_qty = pos['quantity'] + add_qty
                        avg_price = (pos['entry_price'] * pos['quantity'] +
                                     current_price * add_qty) / total_qty
                        self.positions[api_ticker]['quantity'] = total_qty
                        self.positions[api_ticker]['entry_price'] = avg_price
                        self.positions[api_ticker]['step'] = 2
                        print(f"    avg={avg_price:,.0f}원, total={total_qty}주 (12%까지 사용)")
                else:
                    step_txt = f"[S{pos.get('step',2)}]" if pos.get('step', 2) == 1 else ""
                    print(f"  HOLD {api_ticker}({pos.get('name','')}){step_txt}: "
                          f"{pnl_pct:+.2f}% ({hold_min:.0f}분)")

            except Exception as e:
                print(f"  포지션 체크 오류 {api_ticker}: {e}")

        for t in closed:
            del self.positions[t]

    def _close_position(self, api_ticker: str, exit_price: int, reason: str, execute: bool = True):
        """포지션 종료 (매도) — execute=False 시 시뮬레이션만"""
        pos = self.positions[api_ticker]
        qty = pos['quantity']
        pnl_pct = (exit_price / pos['entry_price'] - 1) * 100
        hold_min = (datetime.now() - pos['entry_time']).total_seconds() / 60

        print(f"  [CLOSE] {api_ticker}({pos.get('name','')}): {pnl_pct:+.2f}% ({reason}) | "
              f"진입 {pos['entry_price']:,}원 → 청산 {exit_price:,}원 | {qty}주")

        if execute:
            # ── 실제 보유수량 API 검증 ──────────────────────────────────
            actual_qty = self.api.get_holding_qty_domestic(api_ticker)
            if actual_qty == 0:
                print(f"    ⚠️  실제 보유수량 0주 확인 — 매도 주문 취소 (잔고 없음)")
            else:
                if actual_qty < 0:
                    print(f"    ⚠️  보유수량 API 조회 실패 — 내부 기록({qty}주)으로 매도 진행")
                    actual_qty = qty
                elif actual_qty != qty:
                    print(f"    ⚠️  수량 불일치: 내부={qty}주 vs API={actual_qty}주 → API 값 사용")
                    qty = actual_qty

                order_result = self.api.order_sell_domestic(api_ticker, qty, price=exit_price)
                if order_result:
                    print(f"    ✅ 매도 주문 성공 — {qty}주 @ {exit_price:,}원")
                else:
                    print(f"    ❌ 매도 주문 실패 — 수동 확인 필요! ({api_ticker} {qty}주)")
        else:
            print(f"    [SIMULATED] 매도 {qty}주 @ {exit_price:,}원")

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
        notify_trade_close(
            api_ticker, pos['entry_price'], exit_price,
            pnl_pct, reason, hold_min,
            market="KR", paper_trading=self.paper_trading,
        )

    def _force_close_all(self, reason: str = 'MARKET_CLOSE', execute: bool = True):
        """장 마감 전 모든 포지션 시장가 강제 청산 (오버나잇 리스크 제거)"""
        if not self.positions:
            print(f"  [{reason}] 청산할 포지션 없음")
            return
        print(f"\n{'='*70}")
        print(f"  [{reason}] 전체 {len(self.positions)}개 포지션 강제 청산 중...")
        print(f"{'='*70}")
        for api_ticker in list(self.positions.keys()):
            try:
                pos = self.positions[api_ticker]
                yf_ticker = pos.get('yf_ticker', api_ticker + '.KS')
                df = yf.Ticker(yf_ticker).history(period='1d', interval='1m')
                price = int(round(float(df['Close'].iloc[-1]))) if not df.empty \
                    else pos['entry_price']
                self._close_position(api_ticker, price, reason, execute=execute)
            except Exception as e:
                print(f"  {api_ticker} 강제 청산 오류: {e}")
        self.positions.clear()

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
            self.check_positions(execute=execute)

        if len(self.positions) < max_positions:
            print(f"\n  [매수 신호 탐색]")
            if only_tickers:
                scan_models = {k: v for k, v in models.items() if k in only_tickers}
            else:
                # 볼륨 기반 상위 N개 다이나믹 선정
                max_targets = config.get('allocation', {}).get('scalping_max_volume_targets', max_positions)
                scan_models = self._select_top_volume_kr_tickers(models, max_n=max_targets)

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

                # ── 진입 조건 (ML + Rule 앙상블) ────────────────────────────
                score    = result['signal_score']
                prob     = result['buy_prob']
                prob_adj = result['buy_prob_adj']

                ml_ok       = result['signal'] == 1 and prob >= min_prob
                rule_ok     = score >= 2 and prob >= max(min_prob - 0.10, 0.40) and not result['breakdown_risk']
                rule_strong = score == 3 and prob >= 0.40 and not result['breakdown_risk']

                enter = ml_ok or rule_ok or rule_strong

                if enter:
                    tags = []
                    if ml_ok:   tags.append("ML")
                    if rule_ok or rule_strong: tags.append(f"RULE({score})")
                    if result['s1']: tags.append("S1-Box")
                    if result['s2']: tags.append("S2-EMA")
                    if result['s3']: tags.append("S3-Liq")

                    name = result['name']
                    print(f"\n  BUY SIGNAL: {api_ticker}({name}) [{' '.join(tags)}]")
                    print(f"    가격: {result['price_int']:,}원")
                    print(f"    ML확률: {prob*100:.1f}% → 보정: {prob_adj*100:.1f}%")
                    print(f"    TP: +{result['tp']}%, SL: -{result['sl']}%")

                    # Step-1 매수: 전체 자산의 6% (스캘핑60%×10%)
                    buy_qty = self._calc_step_kr_qty(result['price'], 1, config)

                    if execute:
                        order_result = self.api.order_buy_domestic(
                            api_ticker, buy_qty, price=result['price_int'])
                        if order_result:
                            self.positions[api_ticker] = {
                                'entry_price': result['price_int'],
                                'quantity':    buy_qty,
                                'entry_time':  datetime.now(),
                                'tp':          result['tp'],
                                'sl':          result['sl'],
                                'yf_ticker':   result['yf_ticker'],
                                'name':        result['name'],
                                'step':        1,
                            }
                            print(f"    ORDER PLACED! Step-1 ({buy_qty}주, 6%total)")
                            notify_trade_open(
                                api_ticker, result['price_int'], buy_qty,
                                result['tp'], result['sl'], tags,
                                market="KR", paper_trading=self.paper_trading,
                            )
                        else:
                            print(f"    ORDER FAILED")
                    else:
                        self.positions[api_ticker] = {
                            'entry_price': result['price_int'],
                            'quantity':    buy_qty,
                            'entry_time':  datetime.now(),
                            'tp':          result['tp'],
                            'sl':          result['sl'],
                            'yf_ticker':   result['yf_ticker'],
                            'name':        result['name'],
                            'step':        1,
                        }
                        print(f"    [SIMULATED] ({buy_qty}주)")
                else:
                    strats  = f"S1={result['s1']} S2={result['s2']} S3={result['s3']}"
                    sig_txt = "ML" if result['signal'] == 1 else "NO"
                    print(f"  {api_ticker}: {sig_txt} prob={prob*100:.1f}% rule={score} | {strats}")

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

        # KR 마감 강제 청산 기준: 15:30 KST = 06:30 UTC → 10분 전 = 06:20 UTC
        KR_FORCE_CLOSE_HOUR = 6
        KR_FORCE_CLOSE_MIN = 20

        try:
            while datetime.now() < end_time:
                cycle += 1

                # 장 마감 10분 전 강제 청산 (오버나잇 리스크 제거)
                now_utc = datetime.now(timezone.utc)
                if (now_utc.hour > KR_FORCE_CLOSE_HOUR or
                        (now_utc.hour == KR_FORCE_CLOSE_HOUR and
                         now_utc.minute >= KR_FORCE_CLOSE_MIN)):
                    print(f"\n  [MARKET CLOSE] UTC {now_utc.strftime('%H:%M')} - 마감 10분 전 전체 청산")
                    self._force_close_all('MARKET_CLOSE', execute=execute)
                    break

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
        notify_daily_summary(
            self.trade_log, self.positions,
            market="KR", paper_trading=self.paper_trading,
        )

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
