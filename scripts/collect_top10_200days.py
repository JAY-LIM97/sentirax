"""
나스닥 거래량 TOP 10 종목 200일 데이터 수집

🎯 목적:
- 객관적으로 선정된 TOP 10 종목
- 200일 장기 데이터로 신뢰성 확보
- 통계적 유의성 검증

📊 대상 종목:
NVDA, INTC, TSLA, AMZN, AAPL, NFLX, MSFT, AMD, MU, GOOGL
"""

import sys
import os
import io
import platform

# Windows 한글/이모지 출력 설정
if platform.system() == 'Windows':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta

# TOP 10 종목 (거래량 순)
TOP10_TICKERS = ['NVDA', 'INTC', 'TSLA', 'AMZN', 'AAPL', 'NFLX', 'MSFT', 'AMD', 'MU', 'GOOGL']


def collect_stock_data_200days(ticker: str, save: bool = True):
    """
    단일 종목 200일 데이터 수집
    """

    print(f"\n{'='*70}")
    print(f"📊 {ticker} 데이터 수집 (200일)")
    print(f"{'='*70}")

    try:
        # 날짜 계산 (200일 + 여유)
        end_date = datetime.now()
        start_date = end_date - timedelta(days=250)

        # 주가 데이터
        print(f"1️⃣ {ticker} 주가 데이터...")
        stock = yf.Ticker(ticker)
        df_stock = stock.history(start=start_date, end=end_date)

        if df_stock.empty:
            print(f"❌ {ticker} 데이터 없음!")
            return None

        df = pd.DataFrame({
            'Close': df_stock['Close'],
            'Volume': df_stock['Volume']
        })

        print(f"  ✅ {len(df)}개 행")

        # 거시경제 지표
        print(f"2️⃣ 거시경제 지표...")

        # VIX
        vix = yf.Ticker("^VIX")
        df_vix = vix.history(start=start_date, end=end_date)
        df['vix'] = df_vix['Close'].reindex(df.index, method='ffill')

        # 10년 국채
        treasury = yf.Ticker("^TNX")
        df_treasury = treasury.history(start=start_date, end=end_date)
        df['treasury_10y'] = df_treasury['Close'].reindex(df.index, method='ffill')

        # 달러 인덱스
        dxy = yf.Ticker("DX-Y.NYB")
        df_dxy = dxy.history(start=start_date, end=end_date)
        df['dxy'] = df_dxy['Close'].reindex(df.index, method='ffill')

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
        print(f"3️⃣ 기술 지표...")

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

        # 변화율 (빈 컬럼 - feature_engineer에서 계산)
        df['vix_change'] = None
        df['treasury_10y_change'] = None
        df['dxy_change'] = None
        df['oil_change'] = None
        df['nasdaq_change'] = None
        df['sp500_change'] = None

        print(f"  ✅ 완료")

        # 결측치 처리
        print(f"4️⃣ 결측치 처리...")
        before_len = len(df)
        df = df.dropna(subset=['ma_50'])
        after_len = len(df)

        print(f"  - 처리 전: {before_len}개")
        print(f"  - 처리 후: {after_len}개")

        # 저장
        if save and after_len > 0:
            data_dir = os.path.join(os.path.dirname(__file__), '..', 'data')
            os.makedirs(data_dir, exist_ok=True)

            filename = f"{ticker.lower()}_top10_200days.csv"
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

        return {
            'ticker': ticker,
            'success': True,
            'rows': after_len,
            'start_date': df.index[0] if after_len > 0 else None,
            'end_date': df.index[-1] if after_len > 0 else None,
            'return': (df['Close'].iloc[-1] / df['Close'].iloc[0] - 1) * 100 if after_len > 0 else None
        }

    except Exception as e:
        print(f"\n❌ 오류: {e}")
        import traceback
        traceback.print_exc()
        return {
            'ticker': ticker,
            'success': False,
            'error': str(e)
        }


def main():
    print("=" * 70)
    print("🚀 나스닥 TOP 10 종목 200일 데이터 수집")
    print("=" * 70)

    print(f"\n📋 대상 종목 ({len(TOP10_TICKERS)}개):")
    for i, ticker in enumerate(TOP10_TICKERS, 1):
        print(f"  {i:2d}. {ticker}")

    results = []

    for ticker in TOP10_TICKERS:
        result = collect_stock_data_200days(ticker, save=True)
        if result:
            results.append(result)

    # 최종 요약
    print("\n\n" + "=" * 70)
    print("📊 최종 요약")
    print("=" * 70)

    success_results = [r for r in results if r['success']]
    failed_results = [r for r in results if not r['success']]

    if success_results:
        print(f"\n✅ 성공 ({len(success_results)}개):")
        for r in success_results:
            print(f"  - {r['ticker']:6s}: {r['rows']:3d}일, "
                  f"{r['start_date'].date()} ~ {r['end_date'].date()}, "
                  f"{r['return']:+6.2f}%")

    if failed_results:
        print(f"\n❌ 실패 ({len(failed_results)}개):")
        for r in failed_results:
            print(f"  - {r['ticker']}: {r.get('error', 'Unknown')}")

    print(f"\n📊 통계:")
    print(f"  - 총 종목: {len(TOP10_TICKERS)}개")
    print(f"  - 성공: {len(success_results)}개")
    print(f"  - 실패: {len(failed_results)}개")
    print(f"  - 성공률: {len(success_results)/len(TOP10_TICKERS)*100:.1f}%")

    if success_results:
        avg_days = sum(r['rows'] for r in success_results) / len(success_results)
        avg_return = sum(r['return'] for r in success_results) / len(success_results)

        print(f"\n📈 평균:")
        print(f"  - 평균 일수: {avg_days:.0f}일")
        print(f"  - 평균 수익률: {avg_return:+.2f}%")

    print("\n" + "=" * 70)
    print("✨ 데이터 수집 완료!")
    print("=" * 70)


if __name__ == "__main__":
    main()
