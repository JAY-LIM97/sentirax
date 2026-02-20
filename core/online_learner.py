"""
온라인 학습 매니저 — SGDClassifier partial_fit

실거래 결과(TP/SL/TIMEOUT)를 실시간으로 학습에 반영.
GradientBoosting 대신 SGDClassifier 사용 (partial_fit 지원).

- 최초: train_universal_scalping_model()로 과거 데이터 대량 학습
- 이후: 매 거래 종료 시 update()로 partial_fit 업데이트
- 예측: predict()로 매수 신호 + 확률 반환
"""

import os
import pickle
import numpy as np
from sklearn.linear_model import SGDClassifier
from sklearn.preprocessing import StandardScaler


class OnlineLearner:
    """
    SGDClassifier 기반 온라인 학습 모델.

    Attributes:
        model_path: pkl 파일 경로
        threshold:  매수 결정 확률 임계값 (기본 0.55)
        model:      SGDClassifier (log_loss → predict_proba 지원)
        scaler:     StandardScaler (초기 학습 시 fit, 이후 frozen)
        feature_names: 학습에 사용된 피처 이름 목록
        update_count: partial_fit 업데이트 횟수 (실거래 피드백)
        trade_count:  누적 거래 수
    """

    def __init__(self, model_path: str, threshold: float = 0.55):
        self.model_path = model_path
        self.threshold = threshold
        self.model: SGDClassifier | None = None
        self.scaler: StandardScaler | None = None
        self.feature_names: list[str] | None = None
        self.update_count: int = 0
        self.trade_count: int = 0
        self._load()

    # ──────────────────────────────────────────────────────────────────────────
    # Public API
    # ──────────────────────────────────────────────────────────────────────────

    def is_ready(self) -> bool:
        """모델·스케일러·피처명이 모두 있으면 True"""
        return (self.model is not None
                and self.scaler is not None
                and self.feature_names is not None)

    def predict(self, X_raw: np.ndarray) -> tuple[int, float]:
        """
        매수 신호 예측.

        Args:
            X_raw: shape (1, n_features), unscaled 원본값

        Returns:
            (signal, buy_prob)  signal=1 → 매수, 0 → 관망
        """
        if not self.is_ready():
            return 0, 0.0

        try:
            X_scaled = self.scaler.transform(X_raw)
            proba = self.model.predict_proba(X_scaled)[0]
            buy_prob = float(proba[1]) if len(proba) > 1 else 0.0
            return (1 if buy_prob >= self.threshold else 0), buy_prob
        except Exception as e:
            print(f"  [OnlineLearner] predict error: {e}")
            return 0, 0.0

    def update(self, X_entry_raw: np.ndarray, pnl_pct: float, reason: str) -> bool:
        """
        실거래 결과로 partial_fit 업데이트.

        레이블 결정:
          TP           → 1  (익절 = 올바른 매수)
          SL           → 0  (손절 = 잘못된 매수)
          TIMEOUT/기타 → pnl > 0이면 1, 아니면 0

        Args:
            X_entry_raw: 진입 시점 피처 배열 (unscaled, shape=(1, n))
            pnl_pct:     실현 손익 (%)
            reason:      청산 사유 ('TP', 'SL', 'TIMEOUT', 'MARKET_CLOSE', ...)

        Returns:
            성공 여부
        """
        if not self.is_ready():
            return False

        if reason == 'TP':
            label = 1
        elif reason == 'SL':
            label = 0
        else:
            label = 1 if pnl_pct > 0 else 0

        try:
            X_scaled = self.scaler.transform(X_entry_raw)
            self.model.partial_fit(X_scaled, [label], classes=[0, 1])
            self.update_count += 1
            self.trade_count += 1
            self._save()
            print(f"  [OnlineLearner] #{self.update_count} update: "
                  f"label={label} ({reason}, pnl={pnl_pct:+.2f}%)")
            return True
        except Exception as e:
            print(f"  [OnlineLearner] update error: {e}")
            return False

    def initialize_with_bulk(
        self,
        X_all: np.ndarray,
        y_all: np.ndarray,
        feature_names: list[str],
    ) -> bool:
        """
        대량 과거 데이터로 초기 학습 (train_universal_scalping_model에서 호출).

        scaler.fit_transform → SGD.fit (초기 가중치 설정)
        이후 partial_fit으로 실거래 피드백 누적.

        Args:
            X_all:         shape (N, n_features), unscaled 원본값
            y_all:         shape (N,), int (0 or 1)
            feature_names: 피처 이름 목록

        Returns:
            성공 여부
        """
        if len(X_all) < 50:
            print(f"  [OnlineLearner] bulk init: insufficient data ({len(X_all)} rows)")
            return False

        try:
            self.feature_names = list(feature_names)

            self.scaler = StandardScaler()
            X_scaled = self.scaler.fit_transform(X_all)

            self.model = SGDClassifier(
                loss='log_loss',        # predict_proba 지원
                max_iter=1000,
                tol=1e-3,
                random_state=42,
                learning_rate='optimal',
                class_weight='balanced',
                alpha=1e-4,             # L2 정규화
            )
            self.model.fit(X_scaled, y_all)

            self.update_count = 0
            self.trade_count = 0
            self._save()

            buy_pct = y_all.mean() * 100
            print(f"  [OnlineLearner] Bulk init complete: "
                  f"{len(X_all)} rows, features={len(feature_names)}, "
                  f"buy_ratio={buy_pct:.1f}%")
            return True
        except Exception as e:
            print(f"  [OnlineLearner] bulk init error: {e}")
            return False

    def get_blend_weight(self) -> float:
        """
        온라인 모델의 GBM 대비 신뢰 가중치.
        업데이트 횟수에 따라 0.3 → 0.7 선형 증가.
        """
        n = self.update_count
        if n < 10:
            return 0.0   # 초기엔 GBM 전용
        return min(0.7, 0.30 + n / 100.0)

    # ──────────────────────────────────────────────────────────────────────────
    # Internal
    # ──────────────────────────────────────────────────────────────────────────

    def _save(self):
        os.makedirs(os.path.dirname(self.model_path), exist_ok=True)
        data = {
            'model':         self.model,
            'scaler':        self.scaler,
            'feature_names': self.feature_names,
            'update_count':  self.update_count,
            'trade_count':   self.trade_count,
            'threshold':     self.threshold,
        }
        with open(self.model_path, 'wb') as f:
            pickle.dump(data, f)

    def _load(self):
        if not os.path.exists(self.model_path):
            return
        try:
            with open(self.model_path, 'rb') as f:
                data = pickle.load(f)
            self.model         = data.get('model')
            self.scaler        = data.get('scaler')
            self.feature_names = data.get('feature_names')
            self.update_count  = data.get('update_count', 0)
            self.trade_count   = data.get('trade_count', 0)
            self.threshold     = data.get('threshold', self.threshold)
            print(f"  [OnlineLearner] Loaded: {os.path.basename(self.model_path)} "
                  f"(updates={self.update_count}, trades={self.trade_count})")
        except Exception as e:
            print(f"  [OnlineLearner] load error: {e}")
