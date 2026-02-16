"""
급등주 TOP 20 선별 스크립트

- 당일 거래량 급증 + 가격 상승률 기준
- S&P 500 + Nasdaq 100 주요 종목 중 급등 종목 선별
- yfinance 1분봉 데이터 활용
"""

import sys
import os
import io
import platform

if platform.system() == 'Windows':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# 스캔 대상 (유동성 높은 100개 종목)
SCAN_UNIVERSE = [
    # Mega Cap Tech
    'NVDA', 'AAPL', 'GOOGL', 'MSFT', 'AMZN', 'META', 'TSLA', 'AVGO', 'NFLX', 'AMD',
    # S&P 500 주요
    'JPM', 'XOM', 'JNJ', 'WMT', 'PG', 'HD', 'BAC', 'CVX', 'ABBV', 'LLY',
    'COST', 'ORCL', 'MRK', 'PEP', 'KO', 'TMO', 'CSCO', 'ABT', 'CRM', 'MCD',
    'ACN', 'DHR', 'TXN', 'NEE', 'PM', 'UNP', 'RTX', 'LOW', 'INTC', 'QCOM',
    # 고변동성 + 밈주식
    'PLTR', 'SOFI', 'RIVN', 'LCID', 'NIO', 'MARA', 'COIN', 'HOOD', 'SNAP', 'PINS',
    'ROKU', 'XYZ', 'SHOP', 'SNOW', 'DKNG', 'CRWD', 'ZS', 'PANW', 'SMCI', 'ARM',
    # 바이오/헬스케어
    'MRNA', 'BNTX', 'REGN', 'VRTX', 'ISRG', 'GILD', 'AMGN', 'BMY', 'BIIB', 'ILMN',
    # 에너지/금융
    'SLB', 'OXY', 'MPC', 'PSX', 'VLO', 'GS', 'MS', 'C', 'WFC', 'AXP',
    # 추가 나스닥
    'MRVL', 'MU', 'LRCX', 'KLAC', 'AMAT', 'ON', 'ENPH', 'SEDG', 'FSLR', 'CEG',
    # 리테일/소비재
    'NKE', 'SBUX', 'TGT', 'DG', 'LULU', 'ABNB', 'BKNG', 'MAR', 'CMG', 'YUM'
]


def scan_surging_stocks(top_n: int = 20) -> pd.DataFrame:
    """당일 급등주 스캔"""

    print("=" * 70)
    print("Scanning for surging stocks...")
    print("=" * 70)
    print(f"  Universe: {len(SCAN_UNIVERSE)} stocks")

    results = []

    for i, ticker in enumerate(SCAN_UNIVERSE):
        try:
            stock = yf.Ticker(ticker)

            # 최근 5일 1분봉 데이터
            df = stock.history(period='5d', interval='1m')

            if df.empty or len(df) < 100:
                continue

            # 오늘 데이터
            today = df.index[-1].date()
            df_today = df[df.index.date == today]

            if len(df_today) < 10:
                # 장 전이면 전일 데이터 사용
                yesterday = df.index[-1].date() - timedelta(days=1)
                df_today = df[df.index.date >= yesterday]

            if len(df_today) < 10:
                continue

            # 급등 지표 계산
            open_price = df_today['Close'].iloc[0]
            current_price = df_today['Close'].iloc[-1]
            high_price = df_today['High'].max()
            low_price = df_today['Low'].min()

            day_return = (current_price / open_price - 1) * 100
            day_range = (high_price / low_price - 1) * 100  # 변동폭

            # 거래량 급증
            avg_volume_5d = df['Volume'].mean()
            today_volume = df_today['Volume'].sum()
            volume_spike = today_volume / avg_volume_5d if avg_volume_5d > 0 else 0

            # 최근 1시간 모멘텀
            if len(df_today) >= 60:
                recent_return = (df_today['Close'].iloc[-1] / df_today['Close'].iloc[-60] - 1) * 100
            else:
                recent_return = day_return

            # 급등 점수 (종합)
            surge_score = (
                abs(day_return) * 2 +         # 일중 수익률 (절대값 - 급락도 포함)
                day_range * 1.5 +              # 변동폭
                min(volume_spike, 10) * 3 +    # 거래량 스파이크 (10배 cap)
                abs(recent_return) * 1.5       # 최근 모멘텀
            )

            results.append({
                'ticker': ticker,
                'current_price': current_price,
                'day_return': day_return,
                'day_range': day_range,
                'volume_spike': volume_spike,
                'recent_1h_return': recent_return,
                'surge_score': surge_score,
                'data_points': len(df_today)
            })

            if (i + 1) % 20 == 0:
                print(f"  Scanned {i+1}/{len(SCAN_UNIVERSE)}...")

        except Exception as e:
            continue

    if not results:
        print("No surging stocks found!")
        return pd.DataFrame()

    df_results = pd.DataFrame(results)
    df_results = df_results.sort_values('surge_score', ascending=False)

    # TOP N 선별
    top_stocks = df_results.head(top_n)

    print(f"\n{'='*70}")
    print(f"TOP {top_n} Surging Stocks")
    print(f"{'='*70}")
    print(f"\n{'Rank':<5} {'Ticker':<8} {'Price':>10} {'Return':>8} {'Range':>8} {'VolSpk':>8} {'Score':>8}")
    print("-" * 60)

    for idx, row in top_stocks.iterrows():
        rank = top_stocks.index.get_loc(idx) + 1
        print(f"{rank:<5} {row['ticker']:<8} ${row['current_price']:>8.2f} "
              f"{row['day_return']:>+7.2f}% {row['day_range']:>7.2f}% "
              f"{row['volume_spike']:>7.1f}x {row['surge_score']:>7.1f}")

    # 결과 저장
    results_dir = os.path.join(os.path.dirname(__file__), '..', 'results')
    os.makedirs(results_dir, exist_ok=True)

    save_path = os.path.join(results_dir, 'surging_stocks_today.csv')
    top_stocks.to_csv(save_path, index=False)
    print(f"\nSaved: {save_path}")

    ticker_list = top_stocks['ticker'].tolist()
    print(f"\nSelected tickers: {ticker_list}")

    return top_stocks


if __name__ == "__main__":
    scan_surging_stocks(top_n=20)
