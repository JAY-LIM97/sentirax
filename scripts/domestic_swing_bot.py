"""
국내주식 스윙 봇 (기술적 분석 기반)

- RSI + 이동평균 조합으로 매수/매도 신호 생성
- 매수: RSI < 임계값 AND 5MA > 20MA (과매도 + 상승추세)
- 매도: RSI > 임계값 OR 5MA < 20MA (과매수 또는 하락추세)
- KRW 기반 포지션 사이징
"""

import sys
import os
import io
import json
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
from scripts.get_domestic_surging_stocks import KR_UNIVERSE

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def load_strategy_config() -> dict:
    config_path = os.path.join(PROJECT_ROOT, 'config', 'strategy.json')
    if os.path.exists(config_path):
        with open(config_path) as f:
            return json.load(f)
    return {}


class DomesticSwingBot:
    """국내주식 스윙 봇 (RSI + MA 기반)"""

    def __init__(self, paper_trading: bool = True, account_no: str = None):
        self.paper_trading = paper_trading
        print("=" * 70)
        print("Sentirax 국내주식 스윙 봇 v1.0")
        print("=" * 70)

        self.api = KISTradingAPI(paper_trading=paper_trading)
        if not self.api.authenticate():
            raise Exception("API 인증 실패!")

        if account_no:
            self.api.set_account(account_no, "01")

        self.results_dir = os.path.join(PROJECT_ROOT, 'results')
        os.makedirs(self.results_dir, exist_ok=True)
        print("  봇 초기화 완료!")

    def get_daily_data(self, yf_ticker: str, days: int = 60) -> pd.DataFrame:
        """일봉 데이터 수집"""
        try:
            stock = yf.Ticker(yf_ticker)
            df = stock.history(period=f'{days}d', interval='1d')
            if df.empty or len(df) < 20:
                return None
            return df
        except Exception as e:
            print(f"  데이터 오류 {yf_ticker}: {e}")
            return None

    def compute_signals(self, df: pd.DataFrame, cfg: dict) -> dict:
        """RSI + MA 기반 매매 신호 계산"""
        rsi_buy = cfg.get('rsi_buy_threshold', 35)
        rsi_sell = cfg.get('rsi_sell_threshold', 70)

        # RSI (14일)
        delta = df['Close'].diff()
        gain = delta.where(delta > 0, 0).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))

        # 이동평균
        ma5 = df['Close'].rolling(5).mean()
        ma20 = df['Close'].rolling(20).mean()

        # 최신 값
        current_price = float(df['Close'].iloc[-1])
        current_rsi = float(rsi.iloc[-1]) if not pd.isna(rsi.iloc[-1]) else 50.0
        uptrend = bool(ma5.iloc[-1] > ma20.iloc[-1]) if not (
            pd.isna(ma5.iloc[-1]) or pd.isna(ma20.iloc[-1])) else False

        # 거래량 비율
        vol_ratio = float(df['Volume'].iloc[-1] / df['Volume'].rolling(20).mean().iloc[-1]) \
            if df['Volume'].rolling(20).mean().iloc[-1] > 0 else 1.0

        # 매수 조건: RSI 과매도 + 상승추세
        buy_signal = (current_rsi < rsi_buy) and uptrend

        # 매도 조건: RSI 과매수 OR 하락추세
        sell_signal = (current_rsi > rsi_sell) or not uptrend

        return {
            'price': current_price,
            'price_int': int(round(current_price)),
            'rsi': round(current_rsi, 1),
            'uptrend': uptrend,
            'vol_ratio': round(vol_ratio, 2),
            'buy_signal': buy_signal,
            'sell_signal': sell_signal,
        }

    def _calc_swing_qty(self, price: float, config: dict) -> int:
        """스윙 매수 수량 계산 (KRW 기반)"""
        alloc = config.get('allocation', {})
        swing_pct = alloc.get('swing_pct', 0.30)
        per_trade_pct = alloc.get('swing_per_trade_pct', 0.30)

        summary = self.api.get_account_summary_domestic()
        if summary and summary.get('total_krw', 0) > 0:
            total_krw = summary['total_krw']
        else:
            total_krw = float(alloc.get('account_balance_krw', 0))

        if total_krw > 0 and price > 0:
            budget = total_krw * swing_pct * per_trade_pct
            qty = max(1, int(budget / price))
            print(f"    {total_krw:,.0f}원×{swing_pct:.0%}×{per_trade_pct:.0%}/{price:,.0f}원 = {qty}주")
            return qty

        return config.get('domestic_swing', {}).get('order_quantity', 1)

    def run_once(self, execute: bool = False):
        """1회 스윙 매매 실행"""
        config = load_strategy_config()
        swing_cfg = config.get('domestic_swing', {})
        risk_cfg = config.get('risk', {})

        if risk_cfg.get('stop_all_trading', False):
            print("\n  !! EMERGENCY STOP")
            return

        if not swing_cfg.get('enabled', True):
            print("\n  !! 국내 스윙 비활성화됨")
            return

        max_positions = swing_cfg.get('max_positions', 10)
        min_vol_ratio = swing_cfg.get('min_volume_ratio', 1.5)
        disabled = swing_cfg.get('disabled_tickers', [])
        forced_buy = swing_cfg.get('forced_buy_tickers', [])
        forced_sell = swing_cfg.get('forced_sell_tickers', [])

        # 커스텀 종목 목록 or 전체 유니버스
        custom_tickers = swing_cfg.get('universe_tickers', [])
        if custom_tickers:
            universe = [(t, t + '.KS', t) for t in custom_tickers]
        else:
            universe = KR_UNIVERSE

        print(f"\n\n{'='*70}")
        print(f"  국내주식 스윙 봇 - {datetime.now().strftime('%Y-%m-%d %H:%M')} KST")
        print(f"  유니버스: {len(universe)}개 / MaxPos: {max_positions}")
        print(f"{'='*70}")

        buy_signals = []
        sell_signals = []
        no_signals = []

        for api_ticker, yf_ticker, name in universe:
            if api_ticker in disabled:
                continue

            df = self.get_daily_data(yf_ticker)
            if df is None:
                continue

            sig = self.compute_signals(df, swing_cfg)
            sig['api_ticker'] = api_ticker
            sig['yf_ticker'] = yf_ticker
            sig['name'] = name

            if sig['buy_signal'] and sig['vol_ratio'] >= min_vol_ratio:
                buy_signals.append(sig)
                print(f"  [BUY] {name}({api_ticker}): {sig['price_int']:,}원  "
                      f"RSI={sig['rsi']}  vol×{sig['vol_ratio']}")
            elif sig['sell_signal']:
                sell_signals.append(sig)
                print(f"  [SELL] {name}({api_ticker}): {sig['price_int']:,}원  "
                      f"RSI={sig['rsi']}")
            else:
                no_signals.append(sig)
                print(f"  [HOLD] {name}({api_ticker}): {sig['price_int']:,}원  "
                      f"RSI={sig['rsi']}  uptrend={'Y' if sig['uptrend'] else 'N'}")

        # 강제 매도
        for api_ticker in forced_sell:
            info = next((s for s in (buy_signals + sell_signals + no_signals)
                         if s['api_ticker'] == api_ticker), None)
            if execute:
                price = info['price_int'] if info else 0
                if price > 0:
                    qty = config.get('domestic_swing', {}).get('order_quantity', 1)
                    print(f"\n  FORCED SELL: {api_ticker} @ {price:,}원")
                    self.api.order_sell_domestic(api_ticker, qty, price=price)

        # 강제 매수
        for api_ticker in forced_buy:
            info = next((s for s in (buy_signals + sell_signals + no_signals)
                         if s['api_ticker'] == api_ticker), None)
            if execute and info and info['price_int'] > 0:
                qty = self._calc_swing_qty(info['price'], config)
                print(f"\n  FORCED BUY: {api_ticker} @ {info['price_int']:,}원 / {qty}주")
                self.api.order_buy_domestic(api_ticker, qty, price=info['price_int'])

        # 매수 신호 주문
        if execute and buy_signals:
            print(f"\n  매수 신호 {len(buy_signals)}개 주문 실행...")
            for sig in buy_signals[:max_positions]:
                qty = self._calc_swing_qty(sig['price'], config)
                print(f"\n  ORDER BUY: {sig['name']}({sig['api_ticker']}) "
                      f"@ {sig['price_int']:,}원 x {qty}주")
                result = self.api.order_buy_domestic(sig['api_ticker'], qty,
                                                     price=sig['price_int'])
                status = "OK" if result else "FAIL"
                print(f"  -> {status}")

        # 매도 신호 주문
        if execute and sell_signals:
            print(f"\n  매도 신호 {len(sell_signals)}개 주문 실행...")
            for sig in sell_signals:
                qty = config.get('domestic_swing', {}).get('order_quantity', 1)
                print(f"\n  ORDER SELL: {sig['name']}({sig['api_ticker']}) "
                      f"@ {sig['price_int']:,}원")
                result = self.api.order_sell_domestic(sig['api_ticker'], qty,
                                                      price=sig['price_int'])
                status = "OK" if result else "FAIL"
                print(f"  -> {status}")

        # 일일 리포트 저장
        today_str = datetime.now().strftime('%Y%m%d')
        report = {
            'date': datetime.now().strftime('%Y-%m-%d'),
            'buy_signals': [{'ticker': s['api_ticker'], 'name': s['name'],
                             'price': s['price_int'], 'rsi': s['rsi']} for s in buy_signals],
            'sell_signals': [{'ticker': s['api_ticker'], 'name': s['name'],
                              'price': s['price_int'], 'rsi': s['rsi']} for s in sell_signals],
            'total_scanned': len(universe)
        }
        report_path = os.path.join(self.results_dir, f'domestic_swing_report_{today_str}.json')
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)

        print(f"\n{'='*70}")
        print(f"  결과: 매수신호 {len(buy_signals)}개 / 매도신호 {len(sell_signals)}개 / HOLD {len(no_signals)}개")
        print(f"  리포트 저장: {report_path}")
        print(f"{'='*70}")
