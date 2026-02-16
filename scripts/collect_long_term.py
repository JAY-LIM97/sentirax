"""
장기 데이터 수집 스크립트 (180일+)

🎯 목적:
- 더 많은 데이터로 모델 성능 향상
- 여러 종목 동시 수집
- 자동화된 데이터 파이프라인

📊 수집 대상:
- TSLA (테슬라)
- NVDA (엔비디아)
- AAPL (애플)
- MSFT (마이크로소프트)
"""

import sys
import os
import io

# Windows 한글/이모지 출력 설정
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

# 프로젝트 루트를 파이썬 경로에 추가
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
from collectors.optimized_collector import OptimizedCollector


def collect_stock_data(ticker: str, days: int = 180, save: bool = True):
    """
    주식 데이터 수집 (거시경제 지표 포함)

    Args:
        ticker: 주식 티커 (예: 'TSLA')
        days: 수집 기간 (일)
        save: 파일 저장 여부

    Returns:
        수집된 DataFrame
    """

    print(f"\n{'='*70}")
    print(f"📊 {ticker} 데이터 수집 시작 (과거 {days}일)")
    print(f"{'='*70}")

    try:
        # 날짜 계산
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days + 30)  # 여유 있게

        # OptimizedCollector 사용
        collector = OptimizedCollector()

        print(f"\n📅 기간: {start_date.date()} ~ {end_date.date()}")

        # 주가 데이터 수집
        print(f"\n1️⃣ {ticker} 주가 데이터 수집...")
        stock = yf.Ticker(ticker)
        df_stock = stock.history(start=start_date, end=end_date)

        if df_stock.empty:
            print(f"❌ {ticker} 데이터를 가져올 수 없습니다!")
            return None

        # 필요한 컬럼만 선택
        df = pd.DataFrame({
            'Close': df_stock['Close'],
            'Volume': df_stock['Volume']
        })

        print(f"✅ {len(df)}개 행 수집 완료")

        # 거시경제 지표 수집
        print(f"\n2️⃣ 거시경제 지표 수집...")

        # VIX (변동성 지수)
        print(f"  - VIX...")
        vix = yf.Ticker("^VIX")
        df_vix = vix.history(start=start_date, end=end_date)
        df['vix'] = df_vix['Close'].reindex(df.index, method='ffill')

        # 10년 국채
        print(f"  - 10년 국채...")
        treasury = yf.Ticker("^TNX")
        df_treasury = treasury.history(start=start_date, end=end_date)
        df['treasury_10y'] = df_treasury['Close'].reindex(df.index, method='ffill')

        # 달러 인덱스
        print(f"  - 달러 인덱스...")
        dxy = yf.Ticker("DX-Y.NYB")
        df_dxy = dxy.history(start=start_date, end=end_date)
        df['dxy'] = df_dxy['Close'].reindex(df.index, method='ffill')

        # 유가 (WTI)
        print(f"  - WTI 유가...")
        oil = yf.Ticker("CL=F")
        df_oil = oil.history(start=start_date, end=end_date)
        df['oil'] = df_oil['Close'].reindex(df.index, method='ffill')

        # 나스닥
        print(f"  - 나스닥...")
        nasdaq = yf.Ticker("^IXIC")
        df_nasdaq = nasdaq.history(start=start_date, end=end_date)
        df['nasdaq'] = df_nasdaq['Close'].reindex(df.index, method='ffill')

        # S&P 500
        print(f"  - S&P 500...")
        sp500 = yf.Ticker("^GSPC")
        df_sp500 = sp500.history(start=start_date, end=end_date)
        df['sp500'] = df_sp500['Close'].reindex(df.index, method='ffill')

        print(f"✅ 거시경제 지표 수집 완료")

        # 기술 지표 계산
        print(f"\n3️⃣ 기술 지표 계산...")

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

        # 다음날 수익률 (라벨)
        df['next_day_return'] = df['Close'].pct_change().shift(-1) * 100

        # 변화율 (빈 컬럼 생성 - feature_engineer에서 계산)
        df['vix_change'] = None
        df['treasury_10y_change'] = None
        df['dxy_change'] = None
        df['oil_change'] = None
        df['nasdaq_change'] = None
        df['sp500_change'] = None

        print(f"✅ 기술 지표 계산 완료")

        # 결측치 처리
        print(f"\n4️⃣ 결측치 처리...")
        before_len = len(df)

        # 앞쪽 NaN 제거 (이동평균 계산 위해)
        df = df.dropna(subset=['ma_50'])

        after_len = len(df)
        print(f"  - 처리 전: {before_len}개")
        print(f"  - 처리 후: {after_len}개")
        print(f"  - 제거됨: {before_len - after_len}개")

        # 저장
        if save:
            data_dir = os.path.join(os.path.dirname(__file__), '..', 'data')
            os.makedirs(data_dir, exist_ok=True)

            filename = f"{ticker.lower()}_longterm_{days}days.csv"
            filepath = os.path.join(data_dir, filename)

            df.to_csv(filepath)
            print(f"\n💾 저장 완료: {filepath}")

        # 통계
        print(f"\n📊 데이터 요약:")
        print(f"  - 기간: {df.index[0]} ~ {df.index[-1]}")
        print(f"  - 총 일수: {len(df)}일")
        print(f"  - 컬럼 수: {len(df.columns)}개")
        print(f"  - 시작 가격: ${df['Close'].iloc[0]:.2f}")
        print(f"  - 종료 가격: ${df['Close'].iloc[-1]:.2f}")
        print(f"  - 수익률: {(df['Close'].iloc[-1] / df['Close'].iloc[0] - 1) * 100:+.2f}%")

        print(f"\n✅ {ticker} 수집 완료!")

        return df

    except Exception as e:
        print(f"\n❌ 오류 발생: {e}")
        import traceback
        traceback.print_exc()
        return None


def main():
    print("=" * 70)
    print("📊 장기 데이터 수집 시작")
    print("=" * 70)

    # 수집할 종목과 기간
    tickers = ['TSLA', 'NVDA', 'AAPL', 'MSFT']
    days = 180

    results = {}

    for ticker in tickers:
        df = collect_stock_data(ticker, days=days, save=True)

        if df is not None:
            results[ticker] = {
                'success': True,
                'rows': len(df),
                'start_date': df.index[0],
                'end_date': df.index[-1],
                'return': (df['Close'].iloc[-1] / df['Close'].iloc[0] - 1) * 100
            }
        else:
            results[ticker] = {
                'success': False
            }

    # 최종 요약
    print("\n\n" + "=" * 70)
    print("📊 최종 요약")
    print("=" * 70)

    print(f"\n✅ 성공한 종목:")
    for ticker, info in results.items():
        if info['success']:
            print(f"  - {ticker:5s}: {info['rows']:3d}일, "
                  f"{info['start_date'].date()} ~ {info['end_date'].date()}, "
                  f"수익률 {info['return']:+.2f}%")

    failed = [ticker for ticker, info in results.items() if not info['success']]
    if failed:
        print(f"\n❌ 실패한 종목: {', '.join(failed)}")

    print("\n" + "=" * 70)
    print("✨ 데이터 수집 완료!")
    print("=" * 70)


if __name__ == "__main__":
    main()
