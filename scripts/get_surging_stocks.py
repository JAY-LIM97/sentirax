"""
급등주 TOP 20 선별 스크립트

모드:
1. 기본 모드: 5일 데이터 기반 종합 급등 점수
2. 오프닝 서지 모드: 장 시작 직후 30분 데이터 기반 실시간 급등주 선별

- S&P 500 + Nasdaq 100 + 고변동성 종목 중 급등 종목 선별
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
    'NKE', 'SBUX', 'TGT', 'DG', 'LULU', 'ABNB', 'BKNG', 'MAR', 'CMG', 'YUM',
    # 추가 고변동성 / 중소형
    'RBLX', 'U', 'UPST', 'AFRM', 'PATH', 'IONQ', 'RGTI', 'QUBT',
    'SOUN', 'HIMS', 'CAVA', 'DUOL', 'APP', 'TTD', 'DASH', 'NET',
    'OKTA', 'DDOG', 'S', 'CFLT', 'BILL', 'GTLB', 'ESTC', 'MDB',
    'CELH', 'SMMT', 'RXRX', 'ASTS', 'LUNR', 'RKLB', 'ACHR', 'JOBY',
    'GRAB', 'SE', 'MELI', 'NU', 'BABA', 'PDD', 'JD', 'LI',
    'XPEV', 'BIDU', 'FUTU', 'TIGR', 'MSTR', 'CLSK', 'RIOT', 'HUT'
]


def scan_surging_stocks(top_n: int = 20, opening_surge: bool = False) -> pd.DataFrame:
    """
    급등주 스캔

    Args:
        top_n: 선별할 종목 수
        opening_surge: True면 장 시작 직후 오프닝 서지 모드
                       (오늘 데이터만으로 시가 대비 급등 + 거래량 폭증 종목 선별)
    """

    mode_name = "OPENING SURGE" if opening_surge else "Standard"
    print("=" * 70)
    print(f"Scanning for surging stocks... [{mode_name} mode]")
    print("=" * 70)
    print(f"  Universe: {len(SCAN_UNIVERSE)} stocks")

    results = []

    for i, ticker in enumerate(SCAN_UNIVERSE):
        try:
            stock = yf.Ticker(ticker)

            if opening_surge:
                # 오프닝 서지: 오늘 1분봉 + 5일 평균 거래량 비교용
                df_today_raw = stock.history(period='1d', interval='1m')
                df_5d = stock.history(period='5d', interval='1m')

                if df_today_raw is None or df_today_raw.empty or len(df_today_raw) < 5:
                    continue

                df_today = df_today_raw
                current_price = df_today['Close'].iloc[-1]
                open_price = df_today['Open'].iloc[0]

                # 시가 대비 현재가 변화율
                open_return = (current_price / open_price - 1) * 100

                # 장중 변동폭
                high_price = df_today['High'].max()
                low_price = df_today['Low'].min()
                day_range = (high_price / low_price - 1) * 100

                # 거래량: 오늘 vs 5일 평균 (시간 비례 보정)
                today_volume = df_today['Volume'].sum()
                if df_5d is not None and not df_5d.empty:
                    avg_volume_5d = df_5d['Volume'].mean()
                    # 5일 평균은 1분봉 기준이므로, 오늘 캔들 수 만큼 비교
                    expected_volume = avg_volume_5d * len(df_today)
                    volume_vs_avg = today_volume / expected_volume if expected_volume > 0 else 0
                else:
                    volume_vs_avg = 1.0

                # 최근 10분 모멘텀 (장 초반 급등 포착)
                lookback = min(10, len(df_today) - 1)
                if lookback > 0:
                    momentum = (df_today['Close'].iloc[-1] / df_today['Close'].iloc[-1 - lookback] - 1) * 100
                else:
                    momentum = open_return

                # 오프닝 서지 점수 (시가 대비 변화 + 거래량 폭증 중심)
                surge_score = (
                    abs(open_return) * 3 +           # 시가 대비 변화율 (핵심)
                    min(volume_vs_avg, 10) * 3 +     # 거래량 폭증 (10배 cap)
                    abs(momentum) * 2 +              # 최근 10분 모멘텀
                    day_range * 1.5                  # 장중 변동폭
                )

                results.append({
                    'ticker': ticker,
                    'current_price': current_price,
                    'day_return': open_return,
                    'day_range': day_range,
                    'volume_spike': volume_vs_avg,
                    'recent_1h_return': momentum,
                    'surge_score': surge_score,
                    'data_points': len(df_today)
                })

            else:
                # 기존 방식: 5일 데이터 기반 종합 스캔
                df = stock.history(period='5d', interval='1m')

                if df.empty or len(df) < 100:
                    continue

                today = df.index[-1].date()
                df_today = df[df.index.date == today]

                if len(df_today) < 10:
                    yesterday = df.index[-1].date() - timedelta(days=1)
                    df_today = df[df.index.date >= yesterday]

                if len(df_today) < 10:
                    continue

                open_price = df_today['Close'].iloc[0]
                current_price = df_today['Close'].iloc[-1]
                high_price = df_today['High'].max()
                low_price = df_today['Low'].min()

                day_return = (current_price / open_price - 1) * 100
                day_range = (high_price / low_price - 1) * 100

                avg_volume_5d = df['Volume'].mean()
                today_volume = df_today['Volume'].sum()
                volume_spike = today_volume / avg_volume_5d if avg_volume_5d > 0 else 0

                if len(df_today) >= 60:
                    recent_return = (df_today['Close'].iloc[-1] / df_today['Close'].iloc[-60] - 1) * 100
                else:
                    recent_return = day_return

                surge_score = (
                    abs(day_return) * 2 +
                    day_range * 1.5 +
                    min(volume_spike, 10) * 3 +
                    abs(recent_return) * 1.5
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

            if (i + 1) % 25 == 0:
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
    print(f"TOP {top_n} {'Opening' if opening_surge else ''} Surging Stocks")
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

    if opening_surge:
        save_path = os.path.join(results_dir, 'opening_surge_today.csv')
    else:
        save_path = os.path.join(results_dir, 'surging_stocks_today.csv')
    top_stocks.to_csv(save_path, index=False)
    print(f"\nSaved: {save_path}")

    ticker_list = top_stocks['ticker'].tolist()
    print(f"\nSelected tickers: {ticker_list}")

    return top_stocks


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--opening', action='store_true', help='Opening surge mode')
    parser.add_argument('--top', type=int, default=20, help='Top N stocks')
    args = parser.parse_args()
    scan_surging_stocks(top_n=args.top, opening_surge=args.opening)
