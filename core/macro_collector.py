import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict
import requests

class MacroDataCollector:
    """거시경제 및 시장 지표 수집"""
    
    def __init__(self):
        pass
    
    def collect_macro_data(self, start_date: str, end_date: str) -> pd.DataFrame:
        """
        거시경제 지표 수집
        
        Returns:
            DataFrame with columns:
            - date
            - vix (변동성 지수)
            - treasury_10y (10년 국채 수익률)
            - dxy (달러 인덱스)
            - oil_price (WTI 유가)
            - nasdaq (나스닥 지수)
            - sp500 (S&P 500)
        """
        
        print("📊 거시경제 지표 수집 중...\n")
        
        # yfinance 심볼
        symbols = {
            'vix': '^VIX',           # 변동성 지수
            'treasury_10y': '^TNX',  # 10년 국채
            'dxy': 'DX-Y.NYB',       # 달러 인덱스
            'oil': 'CL=F',           # WTI 유가
            'nasdaq': '^IXIC',       # 나스닥
            'sp500': '^GSPC'         # S&P 500
        }
        
        data_frames = {}
        
        for name, symbol in symbols.items():
            try:
                print(f"   📥 {name.upper()} 다운로드 중...")
                ticker = yf.Ticker(symbol)
                df = ticker.history(start=start_date, end=end_date)
                data_frames[name] = df['Close']
            except Exception as e:
                print(f"   ❌ {name} 실패: {e}")
                data_frames[name] = pd.Series()
        
        # DataFrame 합치기
        result = pd.DataFrame(data_frames)
        result.index = result.index.strftime('%Y-%m-%d')
        
        # 일일 변화율 추가
        for col in result.columns:
            result[f'{col}_change'] = result[col].pct_change() * 100
        
        print(f"✅ {len(result)}일 거시 데이터 수집 완료\n")
        
        return result
    
    def get_technical_indicators(self, symbol: str, start_date: str, 
                                 end_date: str) -> pd.DataFrame:
        """기술적 지표 계산"""
        
        print(f"📈 {symbol} 기술적 지표 계산 중...\n")
        
        ticker = yf.Ticker(symbol)
        df = ticker.history(start=start_date, end=end_date)
        
        # RSI 계산 (간단 버전)
        delta = df['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df['rsi'] = 100 - (100 / (1 + rs))
        
        # 이동평균선
        df['ma_5'] = df['Close'].rolling(window=5).mean()
        df['ma_20'] = df['Close'].rolling(window=20).mean()
        df['ma_50'] = df['Close'].rolling(window=50).mean()
        
        # 볼린저 밴드
        df['bb_middle'] = df['Close'].rolling(window=20).mean()
        bb_std = df['Close'].rolling(window=20).std()
        df['bb_upper'] = df['bb_middle'] + (bb_std * 2)
        df['bb_lower'] = df['bb_middle'] - (bb_std * 2)
        
        # 거래량 이동평균
        df['volume_ma'] = df['Volume'].rolling(window=20).mean()
        df['volume_ratio'] = df['Volume'] / df['volume_ma']
        
        # 날짜 인덱스 문자열로
        df.index = df.index.strftime('%Y-%m-%d')
        
        print(f"✅ 기술적 지표 계산 완료\n")
        
        return df[['rsi', 'ma_5', 'ma_20', 'ma_50', 'volume_ratio']]