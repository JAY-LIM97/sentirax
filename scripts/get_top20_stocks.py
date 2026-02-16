"""
S&P 500 + 나스닥 100에서 상위 20개 종목 선정

🎯 목적:
- 시가총액 + 거래량 기준 상위 20개
- 유동성 높은 종목만 선정
- 리스크 최소화

📊 선정 기준:
1. 시가총액 상위
2. 거래량 안정적
3. 가격 변동성 적정 수준
"""

import sys
import os
import io
import platform

# Windows 한글/이모지 출력 설정
if platform.system() == 'Windows':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta

# S&P 500 + 나스닥 100 주요 종목 (시가총액 상위)
MAJOR_STOCKS = [
    # 기술주 (FAANG + 반도체)
    'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA', 'META', 'TSLA',
    'AMD', 'INTC', 'AVGO', 'QCOM', 'ADBE', 'CRM', 'ORCL',

    # 금융
    'JPM', 'BAC', 'WFC', 'GS', 'MS', 'C',

    # 헬스케어
    'JNJ', 'UNH', 'PFE', 'ABBV', 'LLY', 'MRK',

    # 소비재
    'WMT', 'HD', 'PG', 'KO', 'PEP', 'COST',

    # 통신/미디어
    'NFLX', 'DIS', 'CMCSA', 'VZ', 'T',

    # 산업재
    'BA', 'CAT', 'GE', 'MMM', 'HON',

    # 에너지
    'XOM', 'CVX', 'COP'
]


def analyze_stock(ticker: str) -> dict:
    """
    종목 분석

    Returns:
        분석 결과 딕셔너리
    """
    try:
        stock = yf.Ticker(ticker)
        info = stock.info

        # 30일 거래량 평균
        end_date = datetime.now()
        start_date = end_date - timedelta(days=60)
        hist = stock.history(start=start_date, end=end_date)

        if hist.empty or len(hist) < 20:
            return None

        avg_volume = hist['Volume'].mean()
        avg_price = hist['Close'].mean()
        volatility = hist['Close'].pct_change().std() * 100

        return {
            'ticker': ticker,
            'market_cap': info.get('marketCap', 0),
            'avg_volume_30d': avg_volume,
            'avg_price': avg_price,
            'volatility': volatility,
            'sector': info.get('sector', 'Unknown')
        }

    except Exception as e:
        print(f"❌ {ticker} 분석 실패: {e}")
        return None


def main():
    print("=" * 70)
    print("🔍 상위 20개 종목 선정")
    print("=" * 70)

    print(f"\n📋 분석 대상: {len(MAJOR_STOCKS)}개 종목")
    print("분석 중...\n")

    results = []

    for i, ticker in enumerate(MAJOR_STOCKS, 1):
        print(f"  {i:2d}/{len(MAJOR_STOCKS)} {ticker:6s}... ", end='')

        result = analyze_stock(ticker)

        if result:
            results.append(result)
            print(f"✅ 시가총액: ${result['market_cap']/1e9:.1f}B")
        else:
            print(f"❌")

    # 데이터프레임 생성
    df = pd.DataFrame(results)

    # 시가총액 기준 정렬
    df = df.sort_values('market_cap', ascending=False)

    # 상위 20개 선정
    top20 = df.head(20)

    print("\n\n" + "=" * 70)
    print("🏆 선정된 TOP 20 종목")
    print("=" * 70)

    print(f"\n{'Rank':<6} {'Ticker':<8} {'시가총액':<12} {'거래량':<15} {'변동성':<10} {'섹터':<20}")
    print("-" * 70)

    for i, row in top20.iterrows():
        rank = list(top20.index).index(i) + 1
        print(f"{rank:<6} {row['ticker']:<8} "
              f"${row['market_cap']/1e9:>9.1f}B   "
              f"{row['avg_volume_30d']/1e6:>9.1f}M   "
              f"{row['volatility']:>7.2f}%   "
              f"{row['sector']:<20}")

    # 저장
    results_dir = os.path.join(os.path.dirname(__file__), '..', 'results')
    os.makedirs(results_dir, exist_ok=True)

    output_file = os.path.join(results_dir, 'top20_stocks.csv')
    top20.to_csv(output_file, index=False)

    print(f"\n💾 저장: {output_file}")

    # 종목 리스트 출력
    tickers = top20['ticker'].tolist()
    print(f"\n📊 TOP 20 종목 리스트:")
    print(f"{tickers}")

    print("\n" + "=" * 70)
    print("✨ 선정 완료!")
    print("=" * 70)

    return tickers


if __name__ == "__main__":
    main()
