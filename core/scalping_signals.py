"""
Sentirax 스캘핑 신호 엔진 (1분봉)

[전략 1] 5분봉 박스 돌파
  - 직전 5분봉의 고점·저점을 '박스'로 정의
  - 1분봉이 박스 상단을 종가 돌파 시 매수
  - SL: 매수 캔들 저점 / TP: SL 범위의 1.5배

[전략 2] EMA200 + RSI50 + 상승장악형 캔들
  - 가격이 EMA200 위 (상승추세)
  - RSI > 50 (상승 모멘텀)
  - 상승장악형 캔들 발생
  - SL: 최근 저점 / TP: SL 범위의 2배

[전략 3] 유동성 채널 패턴 돌파
  - 상승추세 확인 (5봉 고점·저점 연속 상승)
  - 시장장학 캔들 발생 후 8봉 이상 수렴 채널 형성
  - 채널 상단 돌파 매수 (망치형 캔들 = 거짓 돌파 → 제외)
  - SL: 채널 하단 / TP: SL 범위의 2배

앙상블 방식:
  - GradientBoosting ML 모델 (기존, 기본 필터)
  - Rule-based Signal Score 0~3 (전략별 합산)
  - 최종 진입: ML + Rule 복합 조건 판단
  - 동적 TP/SL: ATR 기반, 전략 수에 따라 배율 상향
"""

import numpy as np
import pandas as pd


# ──────────────────────────────────────────────────────────────────────────────
# Feature Engineering  (학습·실행 파일 공유)
# ──────────────────────────────────────────────────────────────────────────────

def create_scalping_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    1분봉 스캘핑 Feature Engineering

    기존 피처 (하위 호환) + 전략별 신호 피처 추가
    기존 모델은 저장된 feature_names만 사용 → 새 피처 무시 (하위 호환)
    재학습 시 새 피처까지 포함하여 성능 향상
    """
    feat = pd.DataFrame(index=df.index)

    # ── 기존 피처 (하위 호환) ────────────────────────────────────────────────
    feat['close']  = df['Close']
    feat['volume'] = df['Volume']

    for p in [1, 3, 5, 10, 15, 30]:
        feat[f'return_{p}m'] = df['Close'].pct_change(p) * 100

    for p in [5, 10, 20, 60]:
        feat[f'ma_{p}m']          = df['Close'].rolling(p).mean()
        feat[f'price_to_ma_{p}m'] = (df['Close'] / feat[f'ma_{p}m'] - 1) * 100

    ma20  = df['Close'].rolling(20).mean()
    std20 = df['Close'].rolling(20).std()
    feat['bb_upper'] = (df['Close'] - (ma20 + 2 * std20)) / (df['Close'] + 1e-9) * 100
    feat['bb_lower'] = (df['Close'] - (ma20 - 2 * std20)) / (df['Close'] + 1e-9) * 100
    feat['bb_width'] = (4 * std20) / (ma20 + 1e-9) * 100

    delta = df['Close'].diff()
    gain  = delta.where(delta > 0, 0).rolling(14).mean()
    loss  = (-delta.where(delta < 0, 0)).rolling(14).mean()
    feat['rsi_14m'] = 100 - (100 / (1 + gain / (loss + 1e-9)))

    feat['volume_ratio_5m']  = df['Volume'] / (df['Volume'].rolling(5).mean()  + 1e-9)
    feat['volume_ratio_20m'] = df['Volume'] / (df['Volume'].rolling(20).mean() + 1e-9)
    feat['volume_spike']     = (df['Volume'] > df['Volume'].rolling(20).mean() * 2).astype(int)

    feat['candle_body']       = (df['Close'] - df['Open']) / (df['Open'] + 1e-9) * 100
    feat['candle_wick_upper'] = (
        df['High'] - df[['Open', 'Close']].max(axis=1)
    ) / (df['Open'] + 1e-9) * 100
    feat['candle_wick_lower'] = (
        df[['Open', 'Close']].min(axis=1) - df['Low']
    ) / (df['Open'] + 1e-9) * 100

    day_high = df['High'].rolling(60).max()
    day_low  = df['Low'].rolling(60).min()
    feat['price_position'] = (df['Close'] - day_low) / (day_high - day_low + 0.001) * 100

    ret_1m = df['Close'].pct_change() * 100
    feat['momentum_accel'] = ret_1m.diff()

    feat['volatility_5m']  = df['Close'].pct_change().rolling(5).std()  * 100
    feat['volatility_20m'] = df['Close'].pct_change().rolling(20).std() * 100

    typical_price = (df['High'] + df['Low'] + df['Close']) / 3
    cum_vol  = df['Volume'].cumsum()
    cum_vwap = (typical_price * df['Volume']).cumsum()
    feat['vwap_deviation'] = (df['Close'] / (cum_vwap / (cum_vol + 1e-9)) - 1) * 100

    # ── ATR (전략 공통 — 동적 TP/SL) ─────────────────────────────────────────
    tr = pd.concat([
        df['High'] - df['Low'],
        (df['High'] - df['Close'].shift(1)).abs(),
        (df['Low']  - df['Close'].shift(1)).abs(),
    ], axis=1).max(axis=1)
    atr14 = tr.rolling(14).mean()
    feat['atr_14m']   = atr14
    feat['atr_ratio'] = atr14 / (df['Close'] + 1e-9) * 100   # 변동성 정규화(%)

    # ── 전략 1: 5분봉 박스 돌파 ─────────────────────────────────────────────
    # 직전 5봉 고점·저점 = 박스 상단·하단
    box_high = df['High'].rolling(5).max().shift(1)
    box_low  = df['Low'].rolling(5).min().shift(1)
    feat['box_breakout_5m']  = (df['Close'] > box_high).astype(int)
    feat['box_breakdown_5m'] = (df['Close'] < box_low).astype(int)
    feat['box_breakout_pct'] = (df['Close'] - box_high) / (box_high + 1e-9) * 100  # 돌파 강도

    # ── 전략 2: EMA200 + RSI50 + 상승장악형 ─────────────────────────────────
    ema200 = df['Close'].ewm(span=200, adjust=False).mean()
    feat['ema200_dev']         = (df['Close'] / (ema200 + 1e-9) - 1) * 100
    feat['price_above_ema200'] = (df['Close'] > ema200).astype(int)
    feat['rsi_above_50']       = (feat['rsi_14m'] > 50).astype(int)

    # 상승장악형: 양봉 + 현재 시가 < 전봉 종가 + 현재 종가 > 전봉 시가
    curr_bull = df['Close'] > df['Open']
    engulf_o  = df['Open']  < df['Close'].shift(1)
    engulf_c  = df['Close'] > df['Open'].shift(1)
    feat['bullish_engulfing']        = (curr_bull & engulf_o & engulf_c).astype(int)
    # 강력 장악형: 전봉이 음봉 + 완전 장악
    feat['strong_bullish_engulfing'] = (
        curr_bull &
        (df['Close'].shift(1) < df['Open'].shift(1)) &
        engulf_o & engulf_c
    ).astype(int)

    # 전략 2 통합 플래그
    feat['s2_signal'] = (
        feat['price_above_ema200'] &
        feat['rsi_above_50'] &
        feat['bullish_engulfing']
    ).astype(int)

    # ── 전략 3: 유동성 채널 패턴 ─────────────────────────────────────────────
    # 시장장학 캔들: 캔들 범위 > 20봉 평균의 2배
    candle_range = df['High'] - df['Low']
    avg_range    = candle_range.rolling(20).mean()
    feat['market_maker_candle'] = (candle_range > avg_range * 2.0).astype(int)

    # 수렴 캔들: 현재 범위 < 최근 10봉 최대 범위의 50%
    mm_range_ref            = candle_range.rolling(10).max().shift(1)
    is_conv                 = (candle_range < mm_range_ref * 0.5).astype(int)
    feat['channel_convergence'] = is_conv
    feat['convergence_count']   = is_conv.rolling(8).sum()  # 8봉 이상 → 채널 확정

    # 채널 박스 (최근 10봉)
    ch_high = df['High'].rolling(10).max().shift(1)
    ch_low  = df['Low'].rolling(10).min().shift(1)
    feat['channel_breakout_up']       = (df['Close'] > ch_high).astype(int)
    feat['channel_breakout_strength'] = (df['Close'] - ch_high) / (ch_high + 1e-9) * 100

    # 망치형 캔들: 아래꼬리 > 몸통×2 + 위꼬리 < 몸통×0.5 → 거짓 돌파 신호
    body_abs   = (df['Close'] - df['Open']).abs()
    lower_wick = df[['Open', 'Close']].min(axis=1) - df['Low']
    upper_wick = df['High'] - df[['Open', 'Close']].max(axis=1)
    feat['hammer_candle'] = (
        (lower_wick > body_abs * 2) &
        (upper_wick < body_abs * 0.5) &
        (body_abs > 0)
    ).astype(int)

    # 추세 확인: 최근 5봉 중 고점·저점 모두 4봉 이상 상승
    highs_up = (df['High'] > df['High'].shift(1)).rolling(5).sum() >= 4
    lows_up  = (df['Low']  > df['Low'].shift(1)).rolling(5).sum()  >= 4
    feat['uptrend_5bar'] = (highs_up & lows_up).astype(int)

    # 전략 3 통합 플래그
    feat['s3_signal'] = (
        feat['channel_breakout_up'] &
        feat['uptrend_5bar'] &
        (feat['hammer_candle'] == 0)  # 거짓 돌파 제외
    ).astype(int)

    return feat


# ──────────────────────────────────────────────────────────────────────────────
# Rule-based Signal Engine
# ──────────────────────────────────────────────────────────────────────────────

def compute_rule_signals(df: pd.DataFrame) -> dict:
    """
    최신 1분봉 기준 3전략 룰 신호 계산

    Returns dict:
        s1 (bool)            전략1 — 5분봉 박스 돌파
        s2 (bool)            전략2 — EMA200+RSI50+장악형
        s3 (bool)            전략3 — 유동성 채널 돌파
        signal_score (int)   0~3 (부합 전략 수)
        dynamic_tp (float)   ATR 기반 동적 TP%
        dynamic_sl (float)   ATR 기반 동적 SL%
        breakdown_risk (bool) 거짓 돌파 위험 여부
    """
    default = {
        's1': False, 's2': False, 's3': False,
        'signal_score': 0,
        'dynamic_tp': 1.5, 'dynamic_sl': 1.0,
        'breakdown_risk': False,
    }

    if df is None or len(df) < 210:   # EMA200 계산 최소 봉 수
        return default

    try:
        feat    = create_scalping_features(df)
        latest  = feat.iloc[-1]
        current = float(df['Close'].iloc[-1])

        s1 = bool(latest.get('box_breakout_5m', 0))
        s2 = bool(latest.get('s2_signal', 0))
        s3 = bool(latest.get('s3_signal', 0))
        score = int(s1) + int(s2) + int(s3)

        # 거짓 돌파 위험: 망치형이면서 채널 상향 돌파인 경우
        breakdown_risk = (
            bool(latest.get('hammer_candle', 0)) and
            bool(latest.get('channel_breakout_up', 0))
        )

        # ── 동적 TP/SL (ATR + 캔들 저점 기반) ──────────────────────────────
        atr_val    = float(latest.get('atr_14m', 0) or 0)
        buy_low    = float(df['Low'].iloc[-1])

        sl_candle  = max((current - buy_low) / (current + 1e-9) * 100, 0.3)
        sl_atr     = max(atr_val / (current + 1e-9) * 100, 0.3) if atr_val > 0 else 1.0

        # 캔들 저점 기준 60%, ATR 기준 40% 혼합
        sl_pct = sl_candle * 0.6 + sl_atr * 0.4

        # TP 배율: 신호 수에 따라 상향
        if score == 3:
            tp_ratio = 2.5   # 3전략 모두 일치 → 더 길게
        elif s2 or s3:
            tp_ratio = 2.0   # 전략 2·3 우선
        else:
            tp_ratio = 1.5   # 전략 1 전용

        tp_pct = sl_pct * tp_ratio

        # 합리적 범위 제한
        sl_pct = max(0.5, min(sl_pct, 3.0))
        tp_pct = max(0.75, min(tp_pct, 7.5))

        return {
            's1': s1,
            's2': s2,
            's3': s3,
            'signal_score': score,
            'dynamic_tp': round(tp_pct, 2),
            'dynamic_sl': round(sl_pct, 2),
            'breakdown_risk': breakdown_risk,
        }

    except Exception:
        return default
