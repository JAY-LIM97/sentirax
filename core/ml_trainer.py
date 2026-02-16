"""
Machine Learning Trainer for Sentirax

📚 이 모듈이 하는 일:
1. 데이터 정규화 (Normalization/Scaling)
2. Train/Test 분할 (시계열 특성 유지)
3. 모델 학습 및 평가
4. 모델 저장 및 로드
5. 예측 및 백테스팅

🎓 핵심 개념:
- 정규화: 특징마다 스케일이 다른 문제 해결
- Train/Test 분할: 과적합 방지, 실전 성능 평가
- Random Forest: 여러 Decision Tree의 앙상블
"""

import pandas as pd
import numpy as np
from typing import Tuple, Dict, Any
import pickle
import os
from datetime import datetime

from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)


class MLTrainer:
    """
    머신러닝 학습 및 평가 클래스

    🎯 설계 철학:
    1. 시계열 데이터의 특성 존중 (무작위 섞지 않음)
    2. 과적합 방지 (Train/Test 엄격 분리)
    3. 재현성 (random_state 고정)
    4. 확장성 (여러 모델 쉽게 추가 가능)
    """

    def __init__(self, test_size: float = 0.2, random_state: int = 42):
        """
        Args:
            test_size: 테스트 데이터 비율 (기본 20%)
            random_state: 재현성을 위한 랜덤 시드

        💡 왜 20%인가?
        - 너무 작으면: 테스트 신뢰도 낮음
        - 너무 크면: 학습 데이터 부족
        - 20%: 일반적으로 검증된 비율
        """
        self.test_size = test_size
        self.random_state = random_state

        # 나중에 저장/로드할 객체들
        self.scaler = None
        self.model = None
        self.feature_names = None
        self.training_history = {}

    def normalize_data(
        self,
        X_train: pd.DataFrame,
        X_test: pd.DataFrame
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        데이터 정규화 (표준화)

        Args:
            X_train: 학습 데이터
            X_test: 테스트 데이터

        Returns:
            (정규화된 학습 데이터, 정규화된 테스트 데이터)

        🎓 정규화(Normalization)란?
        - 목적: 모든 특징을 같은 스케일로 맞추기
        - 방법: StandardScaler 사용
          → 평균 0, 표준편차 1로 변환
        - 공식: (x - mean) / std

        💡 왜 필요한가?
        예시:
        - RSI: 0~100 범위
        - return_1d: -10~+10 범위
        - vix: 10~40 범위

        정규화 전:
        - 모델이 큰 숫자(RSI)에만 집중
        - 작은 숫자(return) 무시됨

        정규화 후:
        - 모든 특징이 -3~+3 범위
        - 공평하게 고려됨

        ⚠️ 중요한 규칙:
        1. Train 데이터로만 fit (평균, 표준편차 계산)
        2. Test 데이터는 transform만 (Train 통계 사용)
        3. 왜? Test는 "미래 데이터"라 평균/표준편차 모름

        📊 실전 예시:
        ```
        Train 평균: 50, 표준편차: 10
        Test는 이 값 사용!
        Test의 평균은 절대 사용 금지 (정보 유출)
        ```
        """

        print("\n🔧 [Step 1] 데이터 정규화 (StandardScaler)...")

        # StandardScaler 초기화
        self.scaler = StandardScaler()

        # ⚠️ Train 데이터로만 fit!
        # 평균과 표준편차를 Train에서만 계산
        self.scaler.fit(X_train)

        # 정규화 적용
        X_train_scaled = self.scaler.transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)

        print(f"✅ 정규화 완료")
        print(f"  - Train 평균: {self.scaler.mean_[:3].round(2)} ... (처음 3개 특징)")
        print(f"  - Train 표준편차: {self.scaler.scale_[:3].round(2)} ... (처음 3개 특징)")
        print(f"  - 정규화 후 Train 평균: {X_train_scaled.mean(axis=0)[:3].round(4)} (거의 0)")
        print(f"  - 정규화 후 Train 표준편차: {X_train_scaled.std(axis=0)[:3].round(4)} (거의 1)")

        return X_train_scaled, X_test_scaled

    def split_data(
        self,
        X: pd.DataFrame,
        y: pd.Series
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        """
        시계열 Train/Test 분할

        Args:
            X: 특징 데이터
            y: 라벨 데이터

        Returns:
            (X_train, X_test, y_train, y_test)

        🎓 시계열 분할의 중요성:

        ❌ 일반 분할 (무작위 섞기):
        ```
        전체: [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        Train: [1, 3, 5, 7, 9]  ← 무작위
        Test: [2, 4, 6, 8, 10]  ← 무작위

        문제: "미래"가 "과거"를 학습함 (시간 역행!)
        ```

        ✅ 시계열 분할 (순서 유지):
        ```
        전체: [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        Train: [1, 2, 3, 4, 5, 6, 7, 8]  ← 과거
        Test: [9, 10]                    ← 미래

        장점: 실전과 동일 (과거로 미래 예측)
        ```

        💡 왜 이렇게 해야 하나?
        - 실전에서는 "내일"을 모르는 상태에서 예측
        - Train에 미래 정보가 섞이면 → 과적합
        - Test는 반드시 Train 이후 시점이어야 함

        📊 우리 프로젝트:
        - 총 77일 데이터
        - Train: 처음 80% (약 62일)
        - Test: 마지막 20% (약 15일)
        """

        print("\n📊 [Step 2] Train/Test 분할 (시계열 순서 유지)...")

        # 시계열 분할: 순서 유지!
        split_idx = int(len(X) * (1 - self.test_size))

        # 절대 섞지 않음! (shuffle=False)
        X_train = X.iloc[:split_idx]
        X_test = X.iloc[split_idx:]
        y_train = y.iloc[:split_idx]
        y_test = y.iloc[split_idx:]

        print(f"✅ 분할 완료")
        print(f"  - 전체 데이터: {len(X)}개")
        print(f"  - Train: {len(X_train)}개 ({len(X_train)/len(X)*100:.1f}%)")
        print(f"  - Test: {len(X_test)}개 ({len(X_test)/len(X)*100:.1f}%)")
        print(f"\n  💡 Train 라벨 분포:")
        print(f"     - 매수(1): {(y_train == 1).sum()}개 ({(y_train == 1).sum()/len(y_train)*100:.1f}%)")
        print(f"     - 매도(0): {(y_train == 0).sum()}개 ({(y_train == 0).sum()/len(y_train)*100:.1f}%)")
        print(f"\n  💡 Test 라벨 분포:")
        print(f"     - 매수(1): {(y_test == 1).sum()}개")
        print(f"     - 매도(0): {(y_test == 0).sum()}개")

        return X_train, X_test, y_train, y_test

    def train_random_forest(
        self,
        X_train: np.ndarray,
        y_train: pd.Series,
        n_estimators: int = 100,
        max_depth: int = 10,
        min_samples_split: int = 5,
        min_samples_leaf: int = 2
    ) -> RandomForestClassifier:
        """
        Random Forest 모델 학습

        Args:
            X_train: 정규화된 학습 데이터
            y_train: 학습 라벨
            n_estimators: 트리 개수 (기본 100)
            max_depth: 트리 최대 깊이 (기본 10)
            min_samples_split: 분할 최소 샘플 수 (기본 5)
            min_samples_leaf: 리프 최소 샘플 수 (기본 2)

        Returns:
            학습된 Random Forest 모델

        🎓 Random Forest란?

        1. **Decision Tree (결정 트리)**:
           ```
           RSI < 30?
           ├─ Yes → 매수 (과매도)
           └─ No → VIX > 25?
                   ├─ Yes → 매도 (공포)
                   └─ No → 매수
           ```
           - 장점: 이해하기 쉬움
           - 단점: 과적합 위험

        2. **Random Forest (랜덤 포레스트)**:
           ```
           Tree 1: RSI 중심으로 판단 → 매수
           Tree 2: VIX 중심으로 판단 → 매도
           Tree 3: Volume 중심으로 판단 → 매수
           ...
           Tree 100: 종합 판단 → 매수

           최종 결과: 투표 (매수 60표 vs 매도 40표) → 매수!
           ```
           - 장점: 과적합 방지, 안정적
           - 단점: 느림 (트리 100개)

        💡 하이퍼파라미터 설명:

        - **n_estimators (트리 개수)**: 100
          - 많을수록: 성능 ↑, 느림 ↑
          - 100~200 적당

        - **max_depth (최대 깊이)**: 10
          - 작을수록: 과적합 방지
          - 너무 작으면: 성능 ↓
          - 10~15 적당

        - **min_samples_split**: 5
          - 노드 분할 최소 샘플
          - 과적합 방지

        - **min_samples_leaf**: 2
          - 리프 노드 최소 샘플
          - 과적합 방지

        🎯 우리가 Random Forest를 선택한 이유:
        1. 과적합에 강함 (데이터 77개로 적음)
        2. Feature Importance 제공 (어떤 특징이 중요한지)
        3. 하이퍼파라미터 튜닝 쉬움
        4. 검증된 알고리즘 (Kaggle 많이 사용)
        """

        print("\n🌲 [Step 3] Random Forest 학습...")

        # Random Forest 초기화
        self.model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            random_state=self.random_state,
            n_jobs=-1,  # 모든 CPU 코어 사용
            class_weight='balanced'  # 클래스 불균형 자동 조정
        )

        print(f"  하이퍼파라미터:")
        print(f"    - 트리 개수 (n_estimators): {n_estimators}")
        print(f"    - 최대 깊이 (max_depth): {max_depth}")
        print(f"    - 분할 최소 샘플 (min_samples_split): {min_samples_split}")
        print(f"    - 리프 최소 샘플 (min_samples_leaf): {min_samples_leaf}")
        print(f"    - Class Weight: balanced (불균형 자동 조정)")

        # 학습 시작
        print(f"\n  학습 시작...")
        start_time = datetime.now()

        self.model.fit(X_train, y_train)

        elapsed = (datetime.now() - start_time).total_seconds()
        print(f"✅ 학습 완료 (소요 시간: {elapsed:.2f}초)")

        return self.model

    def evaluate_model(
        self,
        X_test: np.ndarray,
        y_test: pd.Series,
        X_train: np.ndarray = None,
        y_train: pd.Series = None
    ) -> Dict[str, Any]:
        """
        모델 성능 평가

        Args:
            X_test: 테스트 데이터
            y_test: 테스트 라벨
            X_train: 학습 데이터 (선택)
            y_train: 학습 라벨 (선택)

        Returns:
            성능 지표 딕셔너리

        🎓 평가 지표 설명:

        1. **Accuracy (정확도)**:
           - 전체 중 맞춘 비율
           - 공식: (맞춘 개수) / (전체 개수)
           - 예: 10개 중 7개 맞춤 → 70%

        2. **Precision (정밀도)**:
           - 매수 예측 중 실제 상승 비율
           - "매수 신호가 얼마나 정확한가?"
           - 높을수록: 허위 신호 적음

        3. **Recall (재현율)**:
           - 실제 상승 중 잡아낸 비율
           - "상승 기회를 얼마나 놓치지 않았나?"
           - 높을수록: 기회 놓침 적음

        4. **F1-Score**:
           - Precision과 Recall의 조화평균
           - 균형 잡힌 성능 지표

        5. **Confusion Matrix (혼동 행렬)**:
           ```
                      예측 매도(0)  예측 매수(1)
           실제 매도(0)     TN          FP
           실제 매수(1)     FN          TP

           TN (True Negative): 매도 맞춤
           TP (True Positive): 매수 맞춤
           FN (False Negative): 매수 놓침
           FP (False Positive): 허위 매수
           ```

        💡 주식 매매에서 중요한 지표:
        - Precision: 매수 신호의 신뢰도
        - Recall: 기회를 놓치지 않는 능력
        - F1-Score: 종합 성능
        """

        print("\n📈 [Step 4] 모델 성능 평가...")

        results = {}

        # 예측
        y_pred_test = self.model.predict(X_test)

        # Test 성능
        results['test_accuracy'] = accuracy_score(y_test, y_pred_test)
        results['test_precision'] = precision_score(y_test, y_pred_test, zero_division=0)
        results['test_recall'] = recall_score(y_test, y_pred_test, zero_division=0)
        results['test_f1'] = f1_score(y_test, y_pred_test, zero_division=0)
        results['confusion_matrix'] = confusion_matrix(y_test, y_pred_test)

        print(f"✅ Test 성능:")
        print(f"  - Accuracy (정확도): {results['test_accuracy']*100:.2f}%")
        print(f"  - Precision (정밀도): {results['test_precision']*100:.2f}%")
        print(f"  - Recall (재현율): {results['test_recall']*100:.2f}%")
        print(f"  - F1-Score: {results['test_f1']:.3f}")

        print(f"\n  📊 Confusion Matrix:")
        cm = results['confusion_matrix']
        print(f"                예측 매도(0)  예측 매수(1)")
        print(f"  실제 매도(0)      {cm[0,0]:3d}         {cm[0,1]:3d}")
        print(f"  실제 매수(1)      {cm[1,0]:3d}         {cm[1,1]:3d}")

        # Train 성능 (과적합 확인)
        if X_train is not None and y_train is not None:
            y_pred_train = self.model.predict(X_train)
            results['train_accuracy'] = accuracy_score(y_train, y_pred_train)

            print(f"\n  💡 과적합 체크:")
            print(f"     - Train Accuracy: {results['train_accuracy']*100:.2f}%")
            print(f"     - Test Accuracy: {results['test_accuracy']*100:.2f}%")
            print(f"     - 차이: {(results['train_accuracy'] - results['test_accuracy'])*100:.2f}%p")

            if results['train_accuracy'] - results['test_accuracy'] > 0.15:
                print(f"     ⚠️  과적합 가능성 (차이 15%p 이상)")
            else:
                print(f"     ✅ 과적합 없음")

        return results

    def get_feature_importance(self, feature_names: list) -> pd.DataFrame:
        """
        Feature Importance 추출

        Args:
            feature_names: 특징 이름 리스트

        Returns:
            Feature Importance DataFrame

        🎓 Feature Importance란?
        - Random Forest가 각 특징을 얼마나 중요하게 사용했는지
        - 0~1 사이 값 (합계 1.0)
        - 높을수록 중요

        💡 활용:
        1. 중요한 특징 파악
        2. 불필요한 특징 제거
        3. 도메인 지식 검증
           예: "거래량이 중요하다" → Importance 높으면 검증됨
        """

        if self.model is None:
            raise ValueError("모델이 학습되지 않았습니다!")

        importances = self.model.feature_importances_
        feature_imp = pd.DataFrame({
            'feature': feature_names,
            'importance': importances
        }).sort_values('importance', ascending=False)

        return feature_imp

    def save_model(self, save_dir: str = 'models', model_name: str = 'random_forest'):
        """
        모델 저장

        Args:
            save_dir: 저장 디렉토리
            model_name: 모델 이름
        """

        os.makedirs(save_dir, exist_ok=True)

        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"{model_name}_{timestamp}.pkl"
        filepath = os.path.join(save_dir, filename)

        # 저장할 객체
        save_obj = {
            'model': self.model,
            'scaler': self.scaler,
            'feature_names': self.feature_names,
            'training_history': self.training_history
        }

        with open(filepath, 'wb') as f:
            pickle.dump(save_obj, f)

        print(f"✅ 모델 저장 완료: {filepath}")

    def load_model(self, filepath: str):
        """
        모델 로드

        Args:
            filepath: 모델 파일 경로
        """

        with open(filepath, 'rb') as f:
            save_obj = pickle.load(f)

        self.model = save_obj['model']
        self.scaler = save_obj['scaler']
        self.feature_names = save_obj['feature_names']
        self.training_history = save_obj['training_history']

        print(f"✅ 모델 로드 완료: {filepath}")


# ========================================
# 📚 추가 학습 자료
# ========================================

"""
🎓 머신러닝 핵심 개념 정리

1. **과적합 (Overfitting)**
   - 학습 데이터에만 맞춤
   - 새 데이터에서 성능 떨어짐
   - 징후: Train >> Test 정확도
   - 해결: 정규화, 데이터 증강, 앙상블

2. **과소적합 (Underfitting)**
   - 학습 자체가 부족
   - Train, Test 둘 다 낮음
   - 해결: 모델 복잡도 증가, 특징 추가

3. **편향-분산 트레이드오프**
   - 편향(Bias): 단순화 오류
   - 분산(Variance): 과민 반응
   - Random Forest: 분산 낮춤 (앙상블)

4. **정규화 vs 표준화**
   - MinMaxScaler: 0~1 범위
   - StandardScaler: 평균 0, 표준편차 1
   - 우리 선택: StandardScaler (이상치에 강함)

5. **Class Imbalance (클래스 불균형)**
   - 매수 60%, 매도 40% 같은 상황
   - 문제: 다수 클래스만 맞추려 함
   - 해결: class_weight='balanced'

6. **교차 검증 (Cross-Validation)**
   - 일반: K-Fold (데이터 K개로 분할)
   - 시계열: Time Series Split
   - 우리: 단순 Train/Test (데이터 적음)
"""
