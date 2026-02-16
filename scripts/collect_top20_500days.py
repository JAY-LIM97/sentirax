"""
TOP 20 종목 500일 데이터 수집

🎯 목적:
- 장기 데이터로 신뢰성 확보
- 500일 = 약 2년치 데이터
- 통계적 유의성 극대화
"""

import sys
import os
import io
import platform

if platform.system() == 'Windows':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta

# TOP 20 종목
TOP20_TICKERS = ['NVDA', 'AAPL', 'GOOGL', 'MSFT', 'AMZN', 'META', 'TSLA', 'AVGO', 'WMT', 'LLY',
                 'JPM', 'XOM', 'JNJ', 'ORCL', 'COST', 'ABBV', 'HD', 'BAC', 'PG', 'CVX']


def collect_stock_data_500days(ticker: str) -> pd.DataFrame:
    """500일 데이터 수집"""

    print(f"\n{'='*70}")
    print(f"📊 {ticker} 데이터 수집 (500일)")
    print(f"{'='*70}")

    try:
        # 날짜 계산 (500일 + 여유)
        end_date = datetime.now()
        start_date = end_date - timedelta(days=750)  # 주말/공휴일 고려

        # 주가 데이터
        print(f"1️⃣  주가 데이터...")
        stock = yf.Ticker(ticker)
        df_stock = stock.history(start=start_date, end=end_date)

        if df_stock.empty:
            print(f"❌ 데이터 없음!")
            return None

        df = pd.DataFrame({
            'Close': df_stock['Close'],
            'Volume': df_stock['Volume']
        })

        print(f"  ✅ {len(df)}개 행")

        # 거시경제 지표
        print(f"2️⃣  거시경제 지표...")

        # VIX
        vix = yf.Ticker("^VIX")
        df_vix = vix.history(start=start_date, end=end_date)
        df['vix'] = df_vix['Close'].reindex(df.index, method='ffill')

        # 10년 국채
        treasury = yf.Ticker("^TNX")
        df_treasury = treasury.history(start=start_date, end=end_date)
        df['treasury_10y'] = df_treasury['Close'].reindex(df.index, method='ffill')

        # 유가
        oil = yf.Ticker("CL=F")
        df_oil = oil.history(start=start_date, end=end_date)
        df['oil'] = df_oil['Close'].reindex(df.index, method='ffill')

        # 나스닥
        nasdaq = yf.Ticker("^IXIC")
        df_nasdaq = nasdaq.history(start=start_date, end=end_date)
        df['nasdaq'] = df_nasdaq['Close'].reindex(df.index, method='ffill')

        # S&P 500
        sp500 = yf.Ticker("^GSPC")
        df_sp500 = sp500.history(start=start_date, end=end_date)
        df['sp500'] = df_sp500['Close'].reindex(df.index, method='ffill')

        print(f"  ✅ 완료")

        # 기술 지표
        print(f"3️⃣  기술 지표...")

        # RSI
        delta = df['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df['rsi'] = 100 - (100 / (1 + rs))

        # 이동평균
        df['ma_5'] = df['Close'].rolling(window=5).mean()
        df['ma_20'] = df['Close'].rolling(window=20).mean()
        df['ma_50'] = df['Close'].rolling(window=50).mean()

        # 거래량 지표
        df['volume_ratio'] = df['Volume'] / df['Volume'].rolling(window=20).mean()
        df['volume_ma_5'] = df['Volume'].rolling(window=5).mean()
        df['volume_ma_20'] = df['Volume'].rolling(window=20).mean()
        df['volume_trend'] = (df['volume_ma_5'] / df['volume_ma_20']).fillna(1)

        # 다음날 수익률
        df['next_day_return'] = df['Close'].pct_change().shift(-1) * 100

        # 변화율 (빈 컬럼)
        df['vix_change'] = None
        df['treasury_10y_change'] = None
        df['oil_change'] = None
        df['nasdaq_change'] = None
        df['sp500_change'] = None

        print(f"  ✅ 완료")

        # 결측치 처리
        print(f"4️⃣  결측치 처리...")
        before_len = len(df)
        df = df.dropna(subset=['ma_50'])
        after_len = len(df)

        print(f"  - 처리 전: {before_len}개")
        print(f"  - 처리 후: {after_len}개")

        # 저장
        if after_len > 0:
            data_dir = os.path.join(os.path.dirname(__file__), '..', 'data')
            os.makedirs(data_dir, exist_ok=True)

            filename = f"{ticker.lower()}_top20_500days.csv"
            filepath = os.path.join(data_dir, filename)
            df.to_csv(filepath)

            print(f"\n💾 저장: {filename}")

        # 통계
        if after_len > 0:
            print(f"\n📊 요약:")
            print(f"  - 기간: {df.index[0].date()} ~ {df.index[-1].date()}")
            print(f"  - 일수: {after_len}일")
            print(f"  - 시작가: ${df['Close'].iloc[0]:.2f}")
            print(f"  - 종료가: ${df['Close'].iloc[-1]:.2f}")
            print(f"  - 수익률: {(df['Close'].iloc[-1] / df['Close'].iloc[0] - 1) * 100:+.2f}%")

        print(f"\n✅ {ticker} 완료!")

        return df

    except Exception as e:
        print(f"\n❌ 오류: {e}")
        import traceback
        traceback.print_exc()
        return None


def main():
    print("=" * 70)
    print("🚀 TOP 20 종목 500일 데이터 수집")
    print("=" * 70)

    print(f"\n📋 대상 종목 ({len(TOP20_TICKERS)}개):")
    for i, ticker in enumerate(TOP20_TICKERS, 1):
        print(f"  {i:2d}. {ticker}")

    results = []

    for ticker in TOP20_TICKERS:
        df = collect_stock_data_500days(ticker)
        if df is not None:
            results.append({
                'ticker': ticker,
                'rows': len(df),
                'start_date': df.index[0],
                'end_date': df.index[-1]
            })

    # 최종 요약
    print("\n\n" + "=" * 70)
    print("📊 최종 요약")
    print("=" * 70)

    if results:
        print(f"\n✅ 성공 ({len(results)}개):")
        for r in results:
            print(f"  - {r['ticker']:6s}: {r['rows']:3d}일")

        avg_days = sum(r['rows'] for r in results) / len(results)
        print(f"\n📈 평균 일수: {avg_days:.0f}일")

    print("\n" + "=" * 70)
    print("✨ 데이터 수집 완료!")
    print("=" * 70)


if __name__ == "__main__":
    main()
