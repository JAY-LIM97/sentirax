"""
Feature Engineer for Sentirax ML System

📚 Feature Engineering이란?
- 원본 데이터를 머신러닝 모델이 학습하기 좋은 형태로 변환하는 과정
- "GIGO (Garbage In, Garbage Out)": 좋은 특징이 없으면 좋은 모델도 없음
- 도메인 지식(금융 지식)과 데이터 과학의 결합

🎯 우리 프로젝트의 특징 철학:
1. 검증된 기술적 지표 활용 (RSI, MA 등)
2. 거시경제 지표의 변화율 포착
3. 시계열의 시차(Lag) 특성 반영
4. 과적합 방지를 위한 적절한 특징 수 유지 (20개 내외)
"""

import pandas as pd
import numpy as np
from typing import Tuple


class FeatureEngineer:
    """
    머신러닝을 위한 특징 생성 및 라벨링 클래스

    🎓 지도 학습 (Supervised Learning)이란?
    - 정답(Label)이 있는 데이터로 모델을 학습
    - 예: "이 특징들일 때 주가가 올랐다/내렸다"를 학습
    - 우리 목표: 20개 특징 → 매수/매도 결정 (0 또는 1)
    """

    def __init__(self, label_threshold: float = 1.0):
        """
        Args:
            label_threshold: 라벨링 기준 (%)
                           - +1% 이상: 매수(1)
                           - -1% 이하: 매도(0)
                           - 그 외: 무시(NaN)

        💡 왜 1%인가?
        - 너무 작으면: 노이즈(잡음)까지 학습 → 과적합
        - 너무 크면: 학습 데이터 너무 적음
        - 1%: 명확한 신호만 학습, 수수료 고려
        """
        self.label_threshold = label_threshold

    def create_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        20개 이상의 머신러닝 특징 생성

        Args:
            df: 원본 데이터 (tsla_optimized_90days.csv)

        Returns:
            특징이 추가된 DataFrame

        📊 특징 카테고리:
        A. 가격 기반 (7개)
        B. 거래량 기반 (3개)
        C. 변동성 (2개)
        D. 모멘텀 (이미 존재: RSI)
        E. 거시경제 변화율 (이미 존재: vix_change 등)
        F. 시차 특징 (2개)
        """

        df = df.copy()

        # ========================================
        # A. 가격 기반 특징 (Price-Based Features)
        # ========================================

        # 1. return_1d: 1일 수익률
        # 💡 의미: 오늘의 가격 변화 (%)
        # 📐 공식: (오늘 종가 - 어제 종가) / 어제 종가 × 100
        # 🎯 왜 필요: 모멘텀의 가장 기본 단위
        df['return_1d'] = df['Close'].pct_change() * 100

        # 2. return_5d: 5일 수익률
        # 💡 의미: 1주일(거래일 기준) 동안의 추세
        # 🎯 왜 필요: 단기 추세 파악
        df['return_5d'] = df['Close'].pct_change(5) * 100

        # 3. return_20d: 20일 수익률
        # 💡 의미: 1개월(거래일 기준) 동안의 추세
        # 🎯 왜 필요: 중기 추세 파악
        df['return_20d'] = df['Close'].pct_change(20) * 100

        # 4-6. 이동평균은 이미 존재 (ma_5, ma_20, ma_50)
        # 💡 의미: 일정 기간 평균 가격 → 추세 파악
        # 🎯 왜 필요: 노이즈 제거, 주요 지지/저항선

        # 7. ma_cross_5_20: 5일 MA가 20일 MA보다 위에 있는가?
        # 💡 의미: 골든크로스(상승 신호) / 데드크로스(하락 신호)
        # 📐 공식: 1 if MA5 > MA20 else 0
        # 🎯 왜 필요: 전통적인 기술적 분석의 핵심 신호
        df['ma_cross_5_20'] = (df['ma_5'] > df['ma_20']).astype(int)

        # 8. ma_cross_20_50: 20일 MA가 50일 MA보다 위에 있는가?
        # 💡 의미: 중기 추세의 방향
        df['ma_cross_20_50'] = (df['ma_20'] > df['ma_50']).astype(int)

        # ========================================
        # B. 거래량 기반 특징 (Volume-Based Features)
        # ========================================

        # 9. volume_ma_5: 5일 평균 거래량
        # 💡 의미: 최근 거래 활발도
        # 🎯 왜 필요: 거래량이 많으면 추세가 강하다 (Volume 전략이 +0.245 상관관계)
        df['volume_ma_5'] = df['Volume'].rolling(window=5).mean()

        # 10. volume_ma_20: 20일 평균 거래량
        df['volume_ma_20'] = df['Volume'].rolling(window=20).mean()

        # 11. volume_change: 거래량 변화율
        # 💡 의미: 어제 대비 거래량 증가/감소
        # 🎯 왜 필요: 급격한 거래량 변화 = 중요한 이벤트 발생
        df['volume_change'] = df['Volume'].pct_change() * 100

        # volume_ratio는 이미 존재 (현재 거래량 / 평균 거래량)

        # ========================================
        # C. 변동성 특징 (Volatility Features)
        # ========================================

        # 12. volatility_5d: 5일 변동성 (표준편차)
        # 💡 의미: 최근 5일 가격이 얼마나 출렁거렸나
        # 📐 공식: 5일 수익률의 표준편차
        # 🎯 왜 필요: 변동성 높을 때 = 위험하지만 기회도 큼
        # 📚 표준편차: 데이터가 평균에서 얼마나 떨어져 있는지 측정
        df['volatility_5d'] = df['Close'].pct_change().rolling(window=5).std() * 100

        # 13. volatility_20d: 20일 변동성
        # 💡 의미: 중기 변동성
        df['volatility_20d'] = df['Close'].pct_change().rolling(window=20).std() * 100

        # ========================================
        # D. 모멘텀 특징 (Momentum Features)
        # ========================================

        # RSI는 이미 존재
        # 💡 RSI (Relative Strength Index): 상대강도지수
        # 📐 공식: 100 - (100 / (1 + RS)), RS = 평균상승폭/평균하락폭
        # 🎯 의미:
        #    - RSI > 70: 과매수 (너무 올라서 조정 가능성)
        #    - RSI < 30: 과매도 (너무 내려서 반등 가능성)
        #    - RSI = 50: 중립

        # ========================================
        # E. 거시경제 변화율
        # ========================================
        # 💡 왜 변화율인가?
        # - 절대값보다 변화가 중요 (VIX 20→25는 큰 이벤트)
        # - 모델이 상대적 변화를 학습하기 쉬움

        # 원본 데이터의 변화율 컬럼이 비어있으면 직접 계산
        if df['vix_change'].isna().all():
            df['vix_change'] = df['vix'].pct_change() * 100

        if df['treasury_10y_change'].isna().all():
            df['treasury_10y_change'] = df['treasury_10y'].pct_change() * 100

        if df['oil_change'].isna().all():
            df['oil_change'] = df['oil'].pct_change() * 100

        if df['nasdaq_change'].isna().all():
            df['nasdaq_change'] = df['nasdaq'].pct_change() * 100

        if df['sp500_change'].isna().all():
            df['sp500_change'] = df['sp500'].pct_change() * 100

        if 'dxy_change' in df.columns and df['dxy_change'].isna().all():
            df['dxy_change'] = df['dxy'].pct_change() * 100

        # ========================================
        # F. 시차 특징 (Lag Features)
        # ========================================

        # 14. return_1d_lag1: 어제의 1일 수익률
        # 💡 의미: 시계열 데이터의 시간 의존성 포착
        # 🎯 왜 필요: "어제 올랐으면 오늘도 오를까?" 패턴 학습
        # 📚 시차 특징: 과거 값을 현재 특징으로 사용
        df['return_1d_lag1'] = df['return_1d'].shift(1)

        # 15. return_1d_lag2: 그제의 1일 수익률
        # 💡 의미: 2일 전 패턴까지 고려
        df['return_1d_lag2'] = df['return_1d'].shift(2)

        # 16. vix_lag1: 어제의 VIX
        # 💡 의미: 시장 공포 지수의 시차 효과
        df['vix_lag1'] = df['vix'].shift(1)

        # ========================================
        # G. 기존 지표 활용
        # ========================================
        # vix, treasury_10y, oil, nasdaq_change, sp500_change 등

        return df

    def create_labels(self, df: pd.DataFrame) -> pd.Series:
        """
        정답 라벨 생성 (지도 학습용)

        Args:
            df: 특징이 포함된 DataFrame

        Returns:
            라벨 시리즈
            - 1: 매수 신호 (다음날 +1% 이상)
            - 0: 매도 신호 (다음날 -1% 이하)
            - NaN: 무시 (중립)

        🎓 라벨링 전략:
        1. 이진 분류 (Binary Classification)
           - 매수 vs 매도 (홀드는 없음)
           - 단순하지만 효과적

        2. 임계값 사용 (Threshold)
           - 명확한 신호만 학습
           - 애매한 구간(±1% 이내) 제외
           - 과적합 방지

        3. 다음날 수익률 사용
           - 우리는 "내일" 예측
           - 실시간 매매에서는 오늘 특징 → 내일 예측
        """

        # next_day_return이 이미 있으면 사용, 없으면 생성
        if 'next_day_return' not in df.columns:
            df['next_day_return'] = df['Close'].pct_change().shift(-1) * 100

        # 라벨 초기화 (전부 NaN)
        labels = pd.Series(np.nan, index=df.index, name='label')

        # 매수 신호: 다음날 +threshold% 이상
        # 💡 의미: "이 특징들일 때 주가가 확실히 올랐다"
        buy_signal = df['next_day_return'] >= self.label_threshold
        labels[buy_signal] = 1

        # 매도 신호: 다음날 -threshold% 이하
        # 💡 의미: "이 특징들일 때 주가가 확실히 내렸다"
        sell_signal = df['next_day_return'] <= -self.label_threshold
        labels[sell_signal] = 0

        # 중립 신호: -threshold% < 수익률 < +threshold%
        # 💡 처리: 학습에서 제외 (NaN 유지)
        # 🎯 왜 제외: 노이즈, 명확한 패턴 없음

        return labels

    def prepare_ml_data(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
        """
        머신러닝 학습 준비 (원스톱 메서드)

        Args:
            df: 원본 데이터

        Returns:
            (특징 DataFrame, 라벨 Series)

        📝 사용 예시:
        ```python
        engineer = FeatureEngineer()
        X, y = engineer.prepare_ml_data(df)

        # X: 특징 (예: 20개 컬럼, 100개 행)
        # y: 라벨 (예: [1, 0, NaN, 1, 0, ...])
        ```
        """

        # 1. 특징 생성
        df_features = self.create_features(df)

        # 2. 라벨 생성
        labels = self.create_labels(df_features)

        # 3. 결측치가 없는 행만 선택
        # 💡 왜 필요: ML 모델은 NaN을 처리 못함
        # 🎯 방법:
        #   - 특징에 NaN 있으면 제외 (초기 데이터 부족)
        #   - 라벨이 NaN이면 제외 (중립 신호)

        # 특징 컬럼 선택 (학습에 사용할 것만)
        feature_columns = [
            # 가격 기반
            'return_1d', 'return_5d', 'return_20d',
            'ma_5', 'ma_20', 'ma_50',
            'ma_cross_5_20', 'ma_cross_20_50',

            # 거래량 기반
            'volume_ma_5', 'volume_ma_20', 'volume_change',
            'volume_ratio',

            # 변동성
            'volatility_5d', 'volatility_20d',

            # 모멘텀
            'rsi',

            # 거시경제
            'vix', 'vix_change', 'vix_lag1',
            'treasury_10y', 'treasury_10y_change',
            'oil', 'oil_change',
            'nasdaq_change', 'sp500_change',

            # 시차
            'return_1d_lag1', 'return_1d_lag2'
        ]

        # 실제로 존재하는 컬럼만 선택
        available_features = [col for col in feature_columns if col in df_features.columns]

        X = df_features[available_features]
        y = labels

        # NaN 제거
        valid_idx = X.notna().all(axis=1) & y.notna()
        X_clean = X[valid_idx]
        y_clean = y[valid_idx]

        print(f"📊 Feature Engineering 완료:")
        print(f"  - 총 특징 수: {len(available_features)}개")
        print(f"  - 학습 가능한 데이터: {len(y_clean)}개 (원본: {len(df)}개)")
        print(f"  - 매수 신호: {(y_clean == 1).sum()}개 ({(y_clean == 1).sum() / len(y_clean) * 100:.1f}%)")
        print(f"  - 매도 신호: {(y_clean == 0).sum()}개 ({(y_clean == 0).sum() / len(y_clean) * 100:.1f}%)")

        return X_clean, y_clean

    def get_feature_info(self) -> pd.DataFrame:
        """
        특징 정보 요약 (문서화용)

        Returns:
            특징 설명 DataFrame
        """

        features = [
            {'category': '가격', 'name': 'return_1d', 'description': '1일 수익률', 'formula': '(오늘-어제)/어제×100'},
            {'category': '가격', 'name': 'return_5d', 'description': '5일 수익률', 'formula': '(오늘-5일전)/5일전×100'},
            {'category': '가격', 'name': 'return_20d', 'description': '20일 수익률', 'formula': '(오늘-20일전)/20일전×100'},
            {'category': '가격', 'name': 'ma_5', 'description': '5일 이동평균', 'formula': 'mean(Close[-5:])'},
            {'category': '가격', 'name': 'ma_20', 'description': '20일 이동평균', 'formula': 'mean(Close[-20:])'},
            {'category': '가격', 'name': 'ma_50', 'description': '50일 이동평균', 'formula': 'mean(Close[-50:])'},
            {'category': '가격', 'name': 'ma_cross_5_20', 'description': '골든/데드크로스 (5-20)', 'formula': '1 if MA5>MA20 else 0'},
            {'category': '가격', 'name': 'ma_cross_20_50', 'description': '골든/데드크로스 (20-50)', 'formula': '1 if MA20>MA50 else 0'},

            {'category': '거래량', 'name': 'volume_ma_5', 'description': '5일 평균 거래량', 'formula': 'mean(Volume[-5:])'},
            {'category': '거래량', 'name': 'volume_ma_20', 'description': '20일 평균 거래량', 'formula': 'mean(Volume[-20:])'},
            {'category': '거래량', 'name': 'volume_change', 'description': '거래량 변화율', 'formula': '(오늘-어제)/어제×100'},
            {'category': '거래량', 'name': 'volume_ratio', 'description': '거래량 비율', 'formula': 'Volume/MA_Volume'},

            {'category': '변동성', 'name': 'volatility_5d', 'description': '5일 변동성', 'formula': 'std(returns[-5:])'},
            {'category': '변동성', 'name': 'volatility_20d', 'description': '20일 변동성', 'formula': 'std(returns[-20:])'},

            {'category': '모멘텀', 'name': 'rsi', 'description': '상대강도지수', 'formula': '100-100/(1+RS)'},

            {'category': '거시경제', 'name': 'vix', 'description': '시장 공포 지수', 'formula': 'VIX 절대값'},
            {'category': '거시경제', 'name': 'vix_change', 'description': 'VIX 변화율', 'formula': '(오늘-어제)/어제×100'},
            {'category': '거시경제', 'name': 'treasury_10y', 'description': '10년 국채 수익률', 'formula': '절대값'},
            {'category': '거시경제', 'name': 'oil', 'description': '유가', 'formula': 'WTI 가격'},
            {'category': '거시경제', 'name': 'nasdaq_change', 'description': '나스닥 변화율', 'formula': '(오늘-어제)/어제×100'},
            {'category': '거시경제', 'name': 'sp500_change', 'description': 'S&P500 변화율', 'formula': '(오늘-어제)/어제×100'},

            {'category': '시차', 'name': 'return_1d_lag1', 'description': '어제의 1일 수익률', 'formula': 'return_1d.shift(1)'},
            {'category': '시차', 'name': 'return_1d_lag2', 'description': '그제의 1일 수익률', 'formula': 'return_1d.shift(2)'},
            {'category': '시차', 'name': 'vix_lag1', 'description': '어제의 VIX', 'formula': 'vix.shift(1)'},
        ]

        return pd.DataFrame(features)


# ========================================
# 📚 추가 학습 자료
# ========================================

"""
🎓 Feature Engineering 핵심 개념

1. **도메인 지식의 중요성**
   - RSI, MA 같은 지표는 수십 년간 검증됨
   - 무작정 특징 늘리는 것보다 의미 있는 특징이 중요
   - "왜 이 특징이 주가에 영향을 주는가?" 항상 질문

2. **과적합 (Overfitting) 방지**
   - 특징이 너무 많으면: 학습 데이터에만 맞춤, 실전 실패
   - 특징이 너무 적으면: 패턴 못 찾음
   - 적절한 균형: 20-30개 내외

3. **시계열 특성**
   - 주가는 시간에 따라 변함
   - 과거가 미래에 영향: Lag 특징 필요
   - 순서 중요: 데이터 섞으면 안 됨 (Train/Test 분할 시 주의)

4. **정규화 (Normalization) - 다음 단계**
   - 특징마다 스케일이 다름 (RSI: 0-100, return: -10~+10)
   - 모델은 큰 숫자에 영향 많이 받음
   - 해결: StandardScaler로 평균 0, 표준편차 1로 변환

5. **Feature Importance - 모델 학습 후**
   - 어떤 특징이 중요한지 파악
   - 불필요한 특징 제거
   - 반복적 개선
"""
