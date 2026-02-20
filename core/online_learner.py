"""
온라인 학습 매니저 — SGDClassifier partial_fit + Dynamic Variance-Inverted Blending

실거래 결과(TP/SL/TIMEOUT)를 실시간으로 학습에 반영.
GradientBoosting 대신 SGDClassifier 사용 (partial_fit 지원).

블렌딩 방식 (Variance Inversion + Time Decay):
  - 최근 N건 거래에서 OL · GBM 각각의 Log Loss를 시간지수가중으로 계산
  - σ²_i = 시간가중 평균 Log Loss (최신 거래 가중치 높음)
  - W_i  = (σ²_i + ε)^-1  /  Σ(σ²_j + ε)^-1
  - 데이터 < 5건: GBM 100% (SGD 미성숙)
"""

import os
import math
import pickle
from collections import deque

import numpy as np
from sklearn.linear_model import SGDClassifier
from sklearn.preprocessing import StandardScaler

_N_MIN_BLEND = 5      # 블렌딩 시작 최소 거래 수
_WINDOW_SIZE  = 20    # 최근 N건 rolling window
_DECAY        = 0.9   # 지수 감쇠 (최신 거래 가중치 높음)
_EPS          = 1e-6  # 분모 안전값


class OnlineLearner:
    """
    SGDClassifier 기반 온라인 학습 모델.

    Attributes:
        model_path:     pkl 파일 경로
        threshold:      매수 결정 확률 임계값 (기본 0.55)
        model:          SGDClassifier (log_loss → predict_proba 지원)
        scaler:         StandardScaler (초기 학습 시 fit, 이후 frozen)
        feature_names:  학습에 사용된 피처 이름 목록
        update_count:   partial_fit 업데이트 횟수 (실거래 피드백)
        trade_count:    누적 거래 수
        recent_trades:  최근 _WINDOW_SIZE건 {ol_prob, gbm_prob, label} (분산 역가중용)
    """

    def __init__(self, model_path: str, threshold: float = 0.55):
        self.model_path = model_path
        self.threshold = threshold
        self.model: SGDClassifier | None = None
        self.scaler: StandardScaler | None = None
        self.feature_names: list[str] | None = None
        self.update_count: int = 0
        self.trade_count: int = 0
        self.recent_trades: deque = deque(maxlen=_WINDOW_SIZE)
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

    def update(
        self,
        X_entry_raw: np.ndarray,
        pnl_pct: float,
        reason: str,
        ol_prob: float | None = None,
        gbm_prob: float | None = None,
    ) -> bool:
        """
        실거래 결과로 partial_fit 업데이트 + 블렌딩 성과 기록.

        레이블 결정:
          TP           → 1  (익절 = 올바른 매수)
          SL           → 0  (손절 = 잘못된 매수)
          TIMEOUT/기타 → pnl > 0이면 1, 아니면 0

        Args:
            X_entry_raw: 진입 시점 피처 배열 (unscaled, shape=(1, n))
            pnl_pct:     실현 손익 (%)
            reason:      청산 사유 ('TP', 'SL', 'TIMEOUT', 'MARKET_CLOSE', ...)
            ol_prob:     진입 당시 OL 예측 확률 (분산 역가중 계산용)
            gbm_prob:    진입 당시 GBM 예측 확률 (분산 역가중 계산용)

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

            # 블렌딩 성과 기록 (두 모델의 예측 확률이 모두 있을 때만)
            if ol_prob is not None and gbm_prob is not None:
                self.recent_trades.append({
                    'ol_prob':  float(ol_prob),
                    'gbm_prob': float(gbm_prob),
                    'label':    label,
                })

            self._save()
            print(f"  [OnlineLearner] #{self.update_count} update: "
                  f"label={label} ({reason}, pnl={pnl_pct:+.2f}%) "
                  f"[trades_in_window={len(self.recent_trades)}]")
            return True
        except Exception as e:
            print(f"  [OnlineLearner] update error: {e}")
            return False

    def get_dynamic_weights(self) -> tuple[float, float]:
        """
        Variance-Inverted + Time-Decayed 동적 블렌딩 가중치 계산.

        Formula:
            σ²_i  = Σ(decay^(N-1-t) × LogLoss_i(t)) / Σdecay^(N-1-t)
            W_i   = (σ²_i + ε)^-1  /  Σ(σ²_j + ε)^-1

        최근 거래 성과가 나쁜 모델은 자동으로 가중치 감소.
        N < _N_MIN_BLEND 이면 GBM 전용 반환.

        Returns:
            (w_ol, w_gbm) — 합이 1.0인 가중치 쌍
        """
        trades = list(self.recent_trades)

        if len(trades) < _N_MIN_BLEND:
            return 0.0, 1.0  # 데이터 부족 → GBM 전용

        N = len(trades)
        ol_loss_sum  = 0.0
        gbm_loss_sum = 0.0
        weight_sum   = 0.0

        for i, t in enumerate(trades):
            w     = _DECAY ** (N - 1 - i)   # 최신 거래일수록 높은 가중치
            label = t['label']

            # Log Loss (binary cross-entropy), 수치 안전을 위해 클리핑
            ol_p  = max(min(t['ol_prob'],  1 - _EPS), _EPS)
            gbm_p = max(min(t['gbm_prob'], 1 - _EPS), _EPS)

            ol_ll  = -(label * math.log(ol_p)  + (1 - label) * math.log(1 - ol_p))
            gbm_ll = -(label * math.log(gbm_p) + (1 - label) * math.log(1 - gbm_p))

            ol_loss_sum  += w * ol_ll
            gbm_loss_sum += w * gbm_ll
            weight_sum   += w

        # 시간가중 평균 Log Loss = σ²
        ol_sigma2  = ol_loss_sum  / weight_sum
        gbm_sigma2 = gbm_loss_sum / weight_sum

        # 분산 역가중
        w_ol_raw  = 1.0 / (ol_sigma2  + _EPS)
        w_gbm_raw = 1.0 / (gbm_sigma2 + _EPS)
        total     = w_ol_raw + w_gbm_raw

        w_ol  = w_ol_raw  / total
        w_gbm = w_gbm_raw / total

        return w_ol, w_gbm

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
            self.trade_count  = 0
            self.recent_trades.clear()
            self._save()

            buy_pct = y_all.mean() * 100
            print(f"  [OnlineLearner] Bulk init complete: "
                  f"{len(X_all)} rows, features={len(feature_names)}, "
                  f"buy_ratio={buy_pct:.1f}%")
            return True
        except Exception as e:
            print(f"  [OnlineLearner] bulk init error: {e}")
            return False

    # ──────────────────────────────────────────────────────────────────────────
    # Internal
    # ──────────────────────────────────────────────────────────────────────────

    def _save(self):
        os.makedirs(os.path.dirname(self.model_path), exist_ok=True)
        data = {
            'model':          self.model,
            'scaler':         self.scaler,
            'feature_names':  self.feature_names,
            'update_count':   self.update_count,
            'trade_count':    self.trade_count,
            'threshold':      self.threshold,
            'recent_trades':  list(self.recent_trades),
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
            saved_trades       = data.get('recent_trades', [])
            self.recent_trades = deque(saved_trades, maxlen=_WINDOW_SIZE)
            print(f"  [OnlineLearner] Loaded: {os.path.basename(self.model_path)} "
                  f"(updates={self.update_count}, "
                  f"window={len(self.recent_trades)}/{_WINDOW_SIZE})")
        except Exception as e:
            print(f"  [OnlineLearner] load error: {e}")
