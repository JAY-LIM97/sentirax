"""
나스닥 거래량 TOP 10 종목 확인

🎯 목적:
- 객관적 기준으로 종목 선정
- 거래량이 많은 = 유동성 높음 = 신뢰할 수 있는 시그널
"""

import sys
import io
import platform

# Windows 한글/이모지 출력 설정
if platform.system() == 'Windows':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta

# 나스닥 주요 종목 리스트 (시가총액 상위 50개)
nasdaq_tickers = [
    'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA', 'META', 'TSLA', 'AVGO', 'COST', 'NFLX',
    'AMD', 'ADBE', 'CSCO', 'PEP', 'CMCSA', 'INTC', 'INTU', 'QCOM', 'AMGN', 'AMAT',
    'TXN', 'ISRG', 'HON', 'SBUX', 'BKNG', 'ADI', 'GILD', 'VRTX', 'ADP', 'REGN',
    'MDLZ', 'PANW', 'MU', 'LRCX', 'KLAC', 'SNPS', 'CDNS', 'MRVL', 'PYPL', 'CRWD',
    'ASML', 'MAR', 'MELI', 'ORLY', 'CSX', 'ABNB', 'FTNT', 'ADSK', 'NXPI', 'WDAY'
]

print("=" * 70)
print("📊 나스닥 거래량 TOP 10 분석")
print("=" * 70)

print("\n⏳ 최근 30일 평균 거래량 분석 중...")

volume_data = []

for ticker in nasdaq_tickers:
    try:
        stock = yf.Ticker(ticker)

        # 최근 30일 데이터
        end_date = datetime.now()
        start_date = end_date - timedelta(days=30)

        hist = stock.history(start=start_date, end=end_date)

        if not hist.empty:
            avg_volume = hist['Volume'].mean()
            current_price = hist['Close'].iloc[-1]

            volume_data.append({
                'Ticker': ticker,
                'Avg_Volume_30d': avg_volume,
                'Current_Price': current_price,
                'Dollar_Volume': avg_volume * current_price  # 금액 거래량
            })

            print(f"  ✓ {ticker}: {avg_volume/1e6:.1f}M shares")

    except Exception as e:
        print(f"  ✗ {ticker}: 실패 ({e})")

# DataFrame 생성
df = pd.DataFrame(volume_data)

# 거래량 기준 정렬
df = df.sort_values('Avg_Volume_30d', ascending=False)

print("\n" + "=" * 70)
print("🏆 나스닥 거래량 TOP 10")
print("=" * 70)

top10 = df.head(10)

print("\n")
print(top10.to_string(index=False))

print("\n\n📊 선정 기준:")
print(f"  - 분석 종목 수: {len(nasdaq_tickers)}개")
print(f"  - 분석 기간: 최근 30일")
print(f"  - 정렬 기준: 평균 거래량 (주식 수)")

print("\n🎯 선정된 TOP 10:")
for i, row in top10.iterrows():
    print(f"  {row['Ticker']:6s}: {row['Avg_Volume_30d']/1e6:8.1f}M shares/day, ${row['Current_Price']:7.2f}")

print("\n" + "=" * 70)
print("✨ 완료!")
print("=" * 70)

# 저장
top10_tickers = top10['Ticker'].tolist()
print(f"\n📝 TOP 10 티커 리스트:")
print(top10_tickers)

# CSV 저장
import os
results_dir = os.path.join(os.path.dirname(__file__), '..', 'results')
os.makedirs(results_dir, exist_ok=True)

output_path = os.path.join(results_dir, 'nasdaq_volume_top10.csv')
top10.to_csv(output_path, index=False)
print(f"\n💾 저장: {output_path}")
