"""
Sentirax 자동매매 봇

🎯 목적:
- ML 모델 예측 → 실제 매매 자동화
- 모의투자로 안전하게 테스트
- TOP 10 종목 자동 관리

📊 동작 방식:
1. 저장된 ML 모델 로드
2. 최신 시장 데이터 수집
3. 각 종목별 매수/매도 신호 예측
4. API로 실제 주문 실행
5. 결과 로깅
"""

import sys
import os
import io
import json
import pickle
import platform

# Windows 한글/이모지 출력 설정
if platform.system() == 'Windows':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta

from core.feature_engineer import FeatureEngineer
from core.kis_trading_api import KISTradingAPI

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def load_strategy_config() -> dict:
    """strategy.json 실시간 로드"""
    config_path = os.path.join(PROJECT_ROOT, 'config', 'strategy.json')
    if os.path.exists(config_path):
        with open(config_path) as f:
            return json.load(f)
    return {}

# TOP 20 종목 중 모델 저장된 14개 (500일 백테스팅 통과)
TOP20_TICKERS = ['NVDA', 'AAPL', 'GOOGL', 'MSFT', 'AMZN', 'META', 'TSLA', 'AVGO',
                 'JPM', 'ABBV', 'HD', 'BAC', 'PG', 'CVX']


class AutoTradingBot:
    """자동매매 봇"""

    def __init__(self, paper_trading: bool = True, account_no: str = None):
        """
        Args:
            paper_trading: True면 모의투자
            account_no: 계좌번호 8자리
        """
        self.paper_trading = paper_trading
        self.account_no = account_no

        # KIS API 초기화
        print("=" * 70)
        print("🤖 Sentirax 자동매매 봇 초기화")
        print("=" * 70)

        self.api = KISTradingAPI(paper_trading=paper_trading)

        # 인증
        if not self.api.authenticate():
            raise Exception("API 인증 실패!")

        # 계좌 설정
        if account_no:
            self.api.set_account(account_no, "01")

        # 모델 저장 경로
        self.models_dir = os.path.join(os.path.dirname(__file__), '..', 'models')

        # Feature Engineer
        self.engineer = FeatureEngineer(label_threshold=1.0)

        print("\n✅ 봇 초기화 완료!")

    def load_model(self, ticker: str) -> dict:
        """
        저장된 모델 로드

        Args:
            ticker: 종목 티커

        Returns:
            모델 데이터
        """
        model_path = os.path.join(self.models_dir, f'{ticker.lower()}_top20_500d.pkl')

        if not os.path.exists(model_path):
            print(f"⚠️  {ticker} 모델 파일 없음")
            return None

        with open(model_path, 'rb') as f:
            model_data = pickle.load(f)

        return model_data

    def collect_latest_data(self, ticker: str, days: int = 90) -> pd.DataFrame:
        """
        최신 시장 데이터 수집

        Args:
            ticker: 종목 티커
            days: 수집 기간 (일)

        Returns:
            DataFrame
        """
        print(f"\n📊 {ticker} 최신 데이터 수집 중...")

        try:
            end_date = datetime.now()
            start_date = end_date - timedelta(days=days + 100)

            # 주가 데이터
            stock = yf.Ticker(ticker)
            df_stock = stock.history(start=start_date, end=end_date)

            if df_stock.empty:
                print(f"❌ {ticker} 데이터 없음")
                return None

            df = pd.DataFrame({
                'Close': df_stock['Close'],
                'Volume': df_stock['Volume']
            })

            # 거시경제 지표
            vix = yf.Ticker("^VIX")
            df_vix = vix.history(start=start_date, end=end_date)
            df['vix'] = df_vix['Close'].reindex(df.index, method='ffill')

            treasury = yf.Ticker("^TNX")
            df_treasury = treasury.history(start=start_date, end=end_date)
            df['treasury_10y'] = df_treasury['Close'].reindex(df.index, method='ffill')

            oil = yf.Ticker("CL=F")
            df_oil = oil.history(start=start_date, end=end_date)
            df['oil'] = df_oil['Close'].reindex(df.index, method='ffill')

            nasdaq = yf.Ticker("^IXIC")
            df_nasdaq = nasdaq.history(start=start_date, end=end_date)
            df['nasdaq'] = df_nasdaq['Close'].reindex(df.index, method='ffill')

            sp500 = yf.Ticker("^GSPC")
            df_sp500 = sp500.history(start=start_date, end=end_date)
            df['sp500'] = df_sp500['Close'].reindex(df.index, method='ffill')

            # 기술 지표
            delta = df['Close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            rs = gain / loss
            df['rsi'] = 100 - (100 / (1 + rs))

            df['ma_5'] = df['Close'].rolling(window=5).mean()
            df['ma_20'] = df['Close'].rolling(window=20).mean()
            df['ma_50'] = df['Close'].rolling(window=50).mean()

            df['volume_ratio'] = df['Volume'] / df['Volume'].rolling(window=20).mean()
            df['volume_ma_5'] = df['Volume'].rolling(window=5).mean()
            df['volume_ma_20'] = df['Volume'].rolling(window=20).mean()
            df['volume_trend'] = (df['volume_ma_5'] / df['volume_ma_20']).fillna(1)

            df['next_day_return'] = df['Close'].pct_change().shift(-1) * 100

            # 변화율 초기화
            df['vix_change'] = None
            df['treasury_10y_change'] = None
            df['oil_change'] = None
            df['nasdaq_change'] = None
            df['sp500_change'] = None

            # 결측치 처리
            df = df.dropna(subset=['ma_50'])

            print(f"✅ 데이터 수집 완료: {len(df)}개 행")

            return df

        except Exception as e:
            print(f"❌ 데이터 수집 오류: {e}")
            return None

    def predict_signal(self, ticker: str) -> dict:
        """
        특정 종목의 매매 신호 예측

        Args:
            ticker: 종목 티커

        Returns:
            예측 결과 딕셔너리
        """
        print(f"\n{'='*70}")
        print(f"🔮 {ticker} 매매 신호 예측")
        print(f"{'='*70}")

        # 1. 모델 로드
        model_data = self.load_model(ticker)
        if not model_data:
            return None

        model = model_data['model']
        scaler = model_data['scaler']
        feature_names = model_data['feature_names']

        print(f"✅ 모델 로드 완료")
        print(f"  - 학습 날짜: {model_data['train_date']}")
        print(f"  - Test 정확도: {model_data['performance']['test_accuracy']*100:.2f}%")

        # 2. 최신 데이터 수집
        df = self.collect_latest_data(ticker)
        if df is None:
            return None

        # 3. Feature Engineering
        print(f"\n🔧 Feature Engineering...")
        df_features = self.engineer.create_features(df)

        # 최신 날짜 데이터만 사용
        X = df_features[feature_names]
        valid_idx = X.notna().all(axis=1)
        X_clean = X[valid_idx]
        df_clean = df_features[valid_idx]

        if len(X_clean) == 0:
            print(f"❌ 유효한 데이터 없음")
            return None

        # 가장 최신 데이터
        latest_date = df_clean.index[-1]
        X_latest = X_clean.iloc[-1:].copy()
        current_price = df_clean['Close'].iloc[-1]

        print(f"✅ Feature 생성 완료")
        print(f"  - 최신 날짜: {latest_date}")
        print(f"  - 현재가: ${current_price:.2f}")

        # 4. 예측
        print(f"\n🎯 예측 수행...")
        X_scaled = scaler.transform(X_latest)
        prediction = model.predict(X_scaled)[0]
        probability = model.predict_proba(X_scaled)[0]

        signal_text = "매수 (BUY)" if prediction == 1 else "매도 (SELL)"
        signal_emoji = "🟢" if prediction == 1 else "🔴"

        print(f"\n{signal_emoji} 신호: {signal_text}")
        print(f"  - 매도 확률: {probability[0]*100:.1f}%")
        print(f"  - 매수 확률: {probability[1]*100:.1f}%")

        return {
            'ticker': ticker,
            'date': latest_date,
            'price': current_price,
            'signal': int(prediction),
            'signal_text': signal_text,
            'buy_probability': probability[1],
            'sell_probability': probability[0]
        }

    def _calc_swing_qty(self, price: float, buy_prob: float) -> int:
        """스윙 매수 수량 계산 (잔고 × 배분 비율 기반)"""
        config = load_strategy_config()
        alloc = config.get('allocation', {})
        swing_pct = alloc.get('swing_pct', 0.30)
        per_trade_pct = alloc.get('swing_per_trade_pct', 0.30)
        strong_threshold = alloc.get('strong_signal_threshold', 0.70)
        strong_pct = alloc.get('strong_signal_swing_pct', 0.50)

        # 강한 신호 → per_trade 비율 자동 상향
        if buy_prob >= strong_threshold:
            per_trade_pct = min(strong_pct, 1.0)

        # API 잔고 조회 시도
        summary = self.api.get_account_summary()
        if summary and summary.get('total_usd', 0) > 0:
            total_usd = summary['total_usd']
        else:
            total_usd = float(alloc.get('account_balance_usd', 0))

        if total_usd > 0 and price > 0:
            budget = total_usd * swing_pct * per_trade_pct
            qty = max(1, int(budget / price))
            label = "강한" if buy_prob >= strong_threshold else "기본"
            print(f"    [{label}신호] ${total_usd:,.0f}×{swing_pct:.0%}×{per_trade_pct:.0%}/${price:.2f} = {qty}주")
            return qty

        return config.get('swing', {}).get('order_quantity', 1)

    def execute_trade(self, ticker: str, signal: int, price: float, quantity: int = 1):
        """
        실제 매매 실행

        Args:
            ticker: 종목 티커
            signal: 0(매도) or 1(매수)
            price: 현재가
            quantity: 수량

        Returns:
            주문 결과
        """
        if signal == 1:
            result = self.api.order_buy(ticker, quantity, price=price)
        else:
            result = self.api.order_sell(ticker, quantity, price=price)

        return result

    def run_once(self, tickers: list = None, execute: bool = False):
        """
        한 번 실행 (전략 설정 반영)
        """
        config = load_strategy_config()
        swing_cfg = config.get('swing', {})
        risk_cfg = config.get('risk', {})

        # 긴급 정지
        if risk_cfg.get('stop_all_trading', False):
            print("\n  !! EMERGENCY STOP - stop_all_trading=true")
            return

        if not swing_cfg.get('enabled', True):
            print("\n  !! Swing trading disabled in strategy.json")
            return

        if tickers is None:
            tickers = TOP20_TICKERS

        # 전략 설정 반영
        min_prob = swing_cfg.get('min_probability', 0.55)
        order_qty = swing_cfg.get('order_quantity', 1)
        disabled = [t.upper() for t in swing_cfg.get('disabled_tickers', [])]
        forced_buy = [t.upper() for t in swing_cfg.get('forced_buy_tickers', [])]
        forced_sell = [t.upper() for t in swing_cfg.get('forced_sell_tickers', [])]

        # 비활성 종목 제외
        tickers = [t for t in tickers if t not in disabled]

        print("\n\n" + "=" * 70)
        print("Swing Trading Bot (strategy.json live)")
        print("=" * 70)
        print(f"  Tickers: {len(tickers)} | MinProb: {min_prob} | Qty: {order_qty}")
        if disabled:
            print(f"  Disabled: {disabled}")
        if forced_buy:
            print(f"  Forced BUY: {forced_buy}")
        if forced_sell:
            print(f"  Forced SELL: {forced_sell}")
        print()

        results = []

        # 강제 매수 처리
        for ticker in forced_buy:
            if execute:
                print(f"\n  FORCED BUY: {ticker}")
                try:
                    current_price = yf.Ticker(ticker).history(period='1d', interval='1m')['Close'].iloc[-1]
                except Exception:
                    current_price = 0
                if current_price > 0:
                    forced_qty = self._calc_swing_qty(current_price, 0.55)
                    self.api.order_buy(ticker, forced_qty, price=current_price)
                else:
                    print(f"    Failed to get price for {ticker}, skipping")

        # 강제 매도 처리
        for ticker in forced_sell:
            if execute:
                print(f"\n  FORCED SELL: {ticker}")
                try:
                    current_price = yf.Ticker(ticker).history(period='1d', interval='1m')['Close'].iloc[-1]
                except Exception:
                    current_price = 0
                if current_price > 0:
                    self.api.order_sell(ticker, order_qty, price=current_price)
                else:
                    print(f"    Failed to get price for {ticker}, skipping")

        for ticker in tickers:
            try:
                prediction = self.predict_signal(ticker)

                if prediction:
                    results.append(prediction)

                    if execute and prediction['buy_probability'] >= min_prob:
                        print(f"\n  Executing order...")
                        dynamic_qty = self._calc_swing_qty(
                            prediction['price'], prediction['buy_probability']
                        )
                        order_result = self.execute_trade(
                            ticker,
                            prediction['signal'],
                            prediction['price'],
                            quantity=dynamic_qty
                        )

                        if order_result:
                            print(f"  Order success!")
                        else:
                            print(f"  Order failed")

            except Exception as e:
                print(f"\n❌ {ticker} 오류: {e}")
                import traceback
                traceback.print_exc()

        # 요약
        print("\n\n" + "=" * 70)
        print("📊 실행 결과 요약")
        print("=" * 70)

        if results:
            buy_signals = [r for r in results if r['signal'] == 1]
            sell_signals = [r for r in results if r['signal'] == 0]

            print(f"\n🟢 매수 신호 ({len(buy_signals)}개):")
            for r in buy_signals:
                print(f"  - {r['ticker']:6s}: ${r['price']:8.2f} (확률: {r['buy_probability']*100:.1f}%)")

            print(f"\n🔴 매도 신호 ({len(sell_signals)}개):")
            for r in sell_signals:
                print(f"  - {r['ticker']:6s}: ${r['price']:8.2f} (확률: {r['sell_probability']*100:.1f}%)")

        print("\n" + "=" * 70)
        print("✨ 자동매매 봇 실행 완료!")
        print("=" * 70)


def main():
    """메인 함수"""

    print("\n" + "=" * 70)
    print("Sentirax v2.0 - TOP 20 500-day Model Trading Bot")
    print("=" * 70)
    print("\n  Mode: Paper Trading (Mock)")
    print(f"  Tickers: {len(TOP20_TICKERS)} stocks (500-day backtested)")
    print(f"  Stocks: {', '.join(TOP20_TICKERS)}")
    print("  Order: Market price, 1 share each")
    print("  Note: Orders execute only during US market hours")
    print()

    # 사용자 확인
    confirm = input("Execute? (yes/no): ").strip().lower()

    if confirm != 'yes':
        print("\nCancelled.")
        return

    print("\nStarting...\n")

    # 봇 초기화 (모의투자)
    bot = AutoTradingBot(
        paper_trading=True,
        account_no="50163140"
    )

    # 실제 매매 실행
    bot.run_once(tickers=TOP20_TICKERS, execute=True)

    # 결과 로그 저장
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(os.path.dirname(__file__), '..', 'results', f'trading_log_{timestamp}.txt')
    os.makedirs(os.path.dirname(log_file), exist_ok=True)

    print(f"\n📝 로그 저장: {log_file}")


if __name__ == "__main__":
    main()
