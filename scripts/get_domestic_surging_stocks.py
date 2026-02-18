"""
국내주식 급등주 스캔 (KOSPI/KOSDAQ)

- 주요 KOSPI/KOSDAQ 종목 대상 상승률·거래량 기준 스캔
- 오프닝 서지: 장 시작 30분 후 급등주 선별
- 결과를 results/domestic_surge_today.csv 저장
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
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# KOSPI/KOSDAQ 주요 종목 유니버스 (api_ticker, yf_ticker, name) — 40개
KR_UNIVERSE = [
    # 반도체 / IT (5)
    ('005930', '005930.KS', '삼성전자'),
    ('000660', '000660.KS', 'SK하이닉스'),
    ('035420', '035420.KS', 'NAVER'),
    ('035720', '035720.KS', '카카오'),
    ('066570', '066570.KS', 'LG전자'),
    # 자동차 (3)
    ('005380', '005380.KS', '현대차'),
    ('000270', '000270.KS', '기아'),
    ('012330', '012330.KS', '현대모비스'),
    # 배터리 / 에너지 (5)
    ('051910', '051910.KS', 'LG화학'),
    ('006400', '006400.KS', '삼성SDI'),
    ('373220', '373220.KS', 'LG에너지솔루션'),
    ('096770', '096770.KS', 'SK이노베이션'),
    ('009830', '009830.KS', '한화솔루션'),
    # 바이오 / 헬스케어 (3)
    ('207940', '207940.KS', '삼성바이오로직스'),
    ('068270', '068270.KS', '셀트리온'),
    ('091990', '091990.KS', '셀트리온헬스케어'),
    # 금융 (5)
    ('105560', '105560.KS', 'KB금융'),
    ('055550', '055550.KS', '신한지주'),
    ('086790', '086790.KS', '하나금융지주'),
    ('032830', '032830.KS', '삼성생명'),
    ('000810', '000810.KS', '삼성화재'),
    # 철강 / 소재 (3)
    ('005490', '005490.KS', 'POSCO홀딩스'),
    ('003670', '003670.KS', '포스코퓨처엠'),
    ('011170', '011170.KS', '롯데케미칼'),
    # 방산 / 조선 (3)
    ('042660', '042660.KS', '한화오션'),
    ('329180', '329180.KS', 'HD현대중공업'),
    ('267250', '267250.KS', 'HD현대'),
    # 통신 (2)
    ('017670', '017670.KS', 'SK텔레콤'),
    ('030200', '030200.KS', 'KT'),
    # 유통 / 소비재 (3)
    ('028260', '028260.KS', '삼성물산'),
    ('003550', '003550.KS', 'LG'),
    ('034730', '034730.KS', 'SK'),
    # 해운 / 운송 (2)
    ('011200', '011200.KS', 'HMM'),
    ('018880', '018880.KS', '한온시스템'),
    # 에너지 (2)
    ('010950', '010950.KS', 'S-Oil'),
    ('015760', '015760.KS', '한국전력'),
    # KOSDAQ 성장주 (4)
    ('086520', '086520.KQ', '에코프로'),
    ('247540', '247540.KQ', '에코프로비엠'),
    ('259960', '259960.KS', '크래프톤'),
    ('323410', '323410.KS', '카카오뱅크'),
]


def get_stock_momentum(api_ticker: str, yf_ticker: str, name: str,
                       opening_surge: bool = False) -> dict:
    """단일 종목 모멘텀 스코어 계산"""
    try:
        stock = yf.Ticker(yf_ticker)

        if opening_surge:
            # 오프닝 서지: 오늘 1분봉으로 상승률·거래량 계산
            df = stock.history(period='1d', interval='1m')
            if df.empty or len(df) < 5:
                return None

            # 장 시작 대비 현재 수익률
            open_price = df['Open'].iloc[0]
            current_price = df['Close'].iloc[-1]
            change_pct = (current_price / open_price - 1) * 100

            # 거래량 비율 (최근 5분 vs 전체 평균)
            avg_vol = df['Volume'].mean()
            recent_vol = df['Volume'].iloc[-5:].mean()
            vol_ratio = recent_vol / avg_vol if avg_vol > 0 else 1.0

        else:
            # 일반 스캔: 일봉 5일 데이터
            df = stock.history(period='5d', interval='1d')
            if df.empty or len(df) < 2:
                return None

            open_price = df['Close'].iloc[-2]
            current_price = df['Close'].iloc[-1]
            change_pct = (current_price / open_price - 1) * 100

            vol_20d = df['Volume'].mean()
            recent_vol = df['Volume'].iloc[-1]
            vol_ratio = recent_vol / vol_20d if vol_20d > 0 else 1.0

        # 모멘텀 스코어: 상승률 * 거래량비율
        score = change_pct * vol_ratio

        return {
            'api_ticker': api_ticker,
            'yf_ticker': yf_ticker,
            'name': name,
            'price': float(current_price),
            'change_pct': round(change_pct, 2),
            'vol_ratio': round(vol_ratio, 2),
            'score': round(score, 3)
        }

    except Exception as e:
        print(f"  {name}({yf_ticker}) 오류: {e}")
        return None


def scan_domestic_surging_stocks(top_n: int = 20, opening_surge: bool = False) -> pd.DataFrame:
    """
    KOSPI/KOSDAQ 급등주 스캔

    Args:
        top_n: 상위 N개 선택
        opening_surge: True면 오프닝 서지 모드 (장 시작 후 30분 데이터)

    Returns:
        급등주 DataFrame (api_ticker, yf_ticker, name, price, change_pct, vol_ratio, score)
    """
    mode = "오프닝 서지" if opening_surge else "일반"
    print(f"\n{'='*70}")
    print(f"  국내주식 급등주 스캔 [{mode}] - {datetime.now().strftime('%H:%M:%S')}")
    print(f"  유니버스: {len(KR_UNIVERSE)}개 종목")
    print(f"{'='*70}")

    results = []

    for api_ticker, yf_ticker, name in KR_UNIVERSE:
        print(f"  스캔 중: {name}({yf_ticker})...", end=' ')
        result = get_stock_momentum(api_ticker, yf_ticker, name, opening_surge)
        if result:
            results.append(result)
            print(f"{result['change_pct']:+.2f}% (vol×{result['vol_ratio']:.1f})")
        else:
            print("skip")

    if not results:
        print("\n  급등주 없음")
        return pd.DataFrame()

    df = pd.DataFrame(results)

    # 상승 종목만 필터 (마이너스 제외)
    df_up = df[df['change_pct'] > 0].copy()

    if df_up.empty:
        print("\n  상승 종목 없음. 전체 종목 반환.")
        df_up = df.copy()

    # 스코어 기준 정렬
    df_up = df_up.sort_values('score', ascending=False).head(top_n)
    df_up = df_up.reset_index(drop=True)

    # 결과 저장
    results_dir = os.path.join(PROJECT_ROOT, 'results')
    os.makedirs(results_dir, exist_ok=True)
    out_path = os.path.join(results_dir, 'domestic_surge_today.csv')
    df_up.to_csv(out_path, index=False, encoding='utf-8-sig')

    print(f"\n{'='*70}")
    print(f"  TOP {len(df_up)} 국내 급등주")
    print(f"{'='*70}")
    for _, row in df_up.iterrows():
        print(f"  {row['name']:12s} ({row['yf_ticker']}) "
              f"  {row['change_pct']:+.2f}%  vol×{row['vol_ratio']:.1f}  "
              f"score={row['score']:.2f}")

    print(f"\n  저장: {out_path}")
    return df_up


if __name__ == '__main__':
    df = scan_domestic_surging_stocks(top_n=20, opening_surge=False)
    print(df)
