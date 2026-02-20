# Sentirax

AI-powered stock trading system — US & Korean markets, ML models with online learning, automated execution, cloud deployment.

---

## Overview

Sentirax는 머신러닝 기반 주식 자동매매 시스템입니다.

| 구분 | 해외 (미국) | 국내 (KOSPI/KOSDAQ) |
|------|------------|---------------------|
| Swing | TOP 40 대형주, 500일 LogisticRegression | 40종목 유니버스, RSI + 이동평균 |
| Scalping | 당일 급등주 TOP 20, 1분봉 앙상블 (GBM + SGD Online) | 당일 급등주 TOP 20, 1분봉 앙상블 |
| 거래 시간 | KST 23:30~06:30 (미국 장) | KST 09:00~15:30 (한국 장) |
| 워크플로우 | 3개 (swing, scalping, retrain) | 3개 (kr-swing, kr-scalping, kr-retrain) |

---

## Architecture

```
sentirax/
├── core/
│   ├── kis_trading_api.py              # KIS API (해외 + 국내 주문/잔고/보유수량)
│   ├── scalping_signals.py             # 스캘핑 Feature Engineering + 3전략 룰 엔진
│   ├── online_learner.py               # SGDClassifier 온라인 학습 (partial_fit)
│   ├── feature_engineer.py             # Swing ML Feature Engineering
│   ├── model_manager.py                # 모델 재학습 + 성과 추적 + 자동 교체
│   └── slack_notifier.py               # Slack 알림 (진입/청산/일일요약)
│
├── scripts/
│   ├── auto_trading_bot.py             # 해외 Swing 봇 (TOP 40)
│   ├── scalping_bot.py                 # 해외 Scalping 봇 (GBM + Online SGD)
│   ├── get_surging_stocks.py           # 해외 급등주 스캔 (100개 → TOP 20)
│   ├── train_scalping_model.py         # 해외 Scalping 모델 학습 (GBM + Universal SGD)
│   ├── domestic_swing_bot.py           # 국내 Swing 봇 (RSI+MA)
│   ├── domestic_scalping_bot.py        # 국내 Scalping 봇 (GBM + Online SGD)
│   ├── get_domestic_surging_stocks.py  # 국내 급등주 스캔 (40개 유니버스)
│   └── train_domestic_scalping_model.py # 국내 Scalping 모델 학습
│
├── cloud/
│   ├── run_cloud.py             # GitHub Actions 헤드리스 실행기
│   └── requirements-cloud.txt
│
├── config/
│   └── strategy.json            # 실시간 전략 설정 (GitHub 웹에서 수정 가능)
│
├── models/                      # 학습된 ML 모델 (.pkl)
│   ├── *_scalping.pkl           # 종목별 GBM 모델
│   ├── *_kr_scalping.pkl        # 국내 종목별 GBM 모델
│   ├── scalping_online.pkl      # 유니버설 SGD 온라인 모델 (해외)
│   └── kr_scalping_online.pkl   # 유니버설 SGD 온라인 모델 (국내)
│
├── results/                     # 거래 결과 + 리포트 + 체크포인트
│
└── .github/workflows/           # GitHub Actions 자동화 (6개)
    ├── swing-trading.yml        # 해외 스윙 (KST 23:33)
    ├── scalping-bot.yml         # 해외 스캘핑 Phase1+2 (KST 23:35~)
    ├── daily-retrain.yml        # 해외 일일 재학습 (KST 22:07)
    ├── kr-swing-trading.yml     # 국내 스윙 (KST 09:05)
    ├── kr-scalping-bot.yml      # 국내 스캘핑 Phase1+2 (KST 09:35~)
    └── kr-daily-retrain.yml     # 국내 일일 재학습 (KST 08:07)
```

---

## Trading Models

### 해외 Swing Trading (일간)
- **대상**: 40종목 (Mega Cap Tech, 반도체, 금융, 헬스케어, 소비재 등)
- **데이터**: 500일 일봉 + 거시경제 지표 (VIX, 국채, 유가, 나스닥, S&P500)
- **모델**: LogisticRegression + StandardScaler
- **레이블**: 다음날 +1% 이상 = 매수(1), -1% 이하 = 매도(0)

### 해외 Scalping (분간)
- **대상**: 당일 오프닝 서지 TOP 20 (100개 종목 스캔, 장 시작 30분 거래량 기준)
- **데이터**: 5일 1분봉 캔들
- **모델**: GBM(종목별) + SGDClassifier(유니버설, 전 종목 합산 학습)
- **블렌딩**: Variance-Inverted Dynamic Blending — 최근 N거래 Log Loss 기반 시간가중 역분산 가중치 자동 조정
- **온라인 학습**: 매 거래 청산 시 `partial_fit`으로 SGD 모델 실시간 업데이트
- **TP/SL**: 변동성(ATR) 기반 동적 조정
- **포지션**: Step-1(전체 자산 6%) → Step-2(수익 1%+ 시 추가 6%)

### 국내 Swing Trading (일간)
- **대상**: KOSPI/KOSDAQ 40종목 유니버스 전체 스캔
- **전략**: RSI(14) + 5MA/20MA (ML 없이 기술적 분석, 즉시 적용)
- **매수 조건**: RSI < 35 AND 5MA > 20MA (과매도 + 상승추세)
- **매도 조건**: RSI > 70 OR 5MA < 20MA

### 국내 Scalping (분간)
- **대상**: 당일 오프닝 서지 TOP 20 (40개 유니버스 스캔)
- **모델**: GBM(종목별) + SGDClassifier(유니버설)
- **블렌딩**: 해외와 동일한 Variance-Inversion 방식

---

## Scalping Signal Engine (3전략 앙상블)

### S1 — 5분봉 박스 돌파
직전 5봉 고점을 종가 돌파 + 거래량 확인 → 단기 모멘텀 추종

### S2 — EMA200 + RSI50 + 상승장악형
추세(EMA200 위) + 모멘텀(RSI > 50) + 캔들 패턴(Bullish Engulfing) → 중기 추세 추종

### S3 — Liquidity Sweep (저점 스윕 + 흡수)
전략적 매수세 포착 (Stop Hunting 역이용):
- **Stop Run**: 직전 20봉 저점 하향 스윕 감지
- **Volume Confirmation**: 스윕 시 거래량 > 2× 20봉 평균 (`volume_spike_2x`)
- **Absorption**: 흡수 캔들 (아래 꼬리 > 몸통, 매수세가 매도세 흡수)
- **Reclaim (2-Bar Rule)**: 1봉 또는 2봉 이내 저점 위로 종가 복귀
- **CVD / RSI 확인**: Delta Proxy(OHLCV 매수압력 근사) 양수 OR RSI 강세 다이버전스

### 진입 조건 (OR)
| 조건 | 내용 |
|------|------|
| A (ML 기본) | ML 신호=1 AND 확률 ≥ min_prob |
| B (룰 보조) | 룰 2개+ AND 확률 ≥ min_prob-10% AND 거짓돌파 없음 |
| C (룰 강신호) | 룰 3개 일치 AND 확률 ≥ 40% |

---

## Online Learning (Real-time Adaptive Model)

```
[초기 학습]  train_scalping_model.py
              → 전 종목 합산 1분봉 데이터로 SGD 초기 학습 (scikit-learn SGDClassifier)
              → models/scalping_online.pkl 저장

[실시간 추론] predict_entry()
              → OL 예측: SGD.predict_proba() [universal]
              → GBM 예측: GBM.predict_proba() [per-ticker]
              → Dynamic Blending: Variance-Inverted 가중치 (최근 20건 Log Loss 기반)

[실시간 학습] _close_position() (TP/SL/TIMEOUT 청산 시)
              → label = 1(TP), 0(SL), pnl>0(TIMEOUT)
              → SGD.partial_fit(X_entry_scaled, [label])
              → recent_trades 큐 업데이트 (최근 20건 유지)
              → pkl 자동 저장

[가중치 조정] get_dynamic_weights()
              σ²_i = Σ(decay^t × LogLoss_i(t)) / Σdecay^t  (decay=0.9, 최신 우선)
              W_i  = (σ²_i + ε)^-1 / Σ(σ²_j + ε)^-1
              → 성능 나쁜 모델 자동 감점, 좋은 모델 자동 가중치 상승
              → 거래 < 5건: GBM 100% (SGD 미성숙)
```

---

## Daily Execution Flow (GitHub Actions)

```
[해외]
KST 22:07  [daily-retrain.yml]
           ├─ Performance tracking + model auto-switch
           ├─ Retrain 40 swing models (최신 데이터)
           └─ Git commit updated models

KST 23:33  [swing-trading.yml]
           └─ Swing trading (40종목 매수/매도)

KST 23:35  [scalping-bot.yml]
           ├─ Phase 1: 30분 대기 → 오프닝 서지 스캔 → GBM+SGD 학습 → 스캘핑 (~2.5hr)
           └─ Phase 2: 체크포인트 복원 → 스캘핑 계속 (~3.5hr) → SGD 모델 자동 업데이트

[국내]
KST 08:07  [kr-daily-retrain.yml]
           ├─ 국내 급등주 스캔 (40개 유니버스)
           └─ GBM + Universal SGD 재학습

KST 09:05  [kr-swing-trading.yml]
           └─ 국내 스윙 (RSI+MA 신호 → 매수/매도)

KST 09:35  [kr-scalping-bot.yml]
           ├─ Phase 1: 30분 대기 → 오프닝 서지 → GBM+SGD 학습 → 스캘핑 (~2.5hr)
           └─ Phase 2: 체크포인트 복원 → 스캘핑 계속 (~3hr)

* 미국/한국 휴장일 자동 스킵
* cron 3시간 선행 + standby job (장 시작 전이면 대기, 이미 장 중이면 즉시 시작)
* cron 시간 홀수 분 오프셋 → GitHub Actions 혼잡 시간대(:00, :30) 회피
```

---

## Live Strategy Control

`config/strategy.json`을 GitHub 웹에서 수정하면 다음 사이클부터 즉시 반영됩니다.

```json
{
  "swing":            { "enabled": true, "max_positions": 14, "min_probability": 0.55 },
  "scalping":         { "enabled": true, "max_positions": 5,  "min_probability": 0.55 },
  "domestic_swing":   { "enabled": true, "max_positions": 10, "rsi_buy_threshold": 35 },
  "domestic_scalping":{ "enabled": true, "max_positions": 5,  "min_probability": 0.55 },
  "allocation": {
    "swing_pct": 0.25,  "scalping_pct": 0.60,
    "scalping_step_pct": 0.10,
    "scalping_step2_min_profit_pct": 1.0,
    "scalping_max_volume_targets": 5,
    "account_balance_usd": 200000,  "account_balance_krw": 30000000
  },
  "risk": { "max_daily_loss_pct": -10.0, "stop_all_trading": false }
}
```

**주요 제어:**
- `stop_all_trading: true` → 긴급 정지 (모든 신규 진입 중단)
- `disabled_tickers` → 특정 종목 제외
- `forced_buy_tickers` → 강제 매수
- `tp_override / sl_override` → TP/SL 수동 오버라이드
- `universe_tickers` (domestic_swing) → 커스텀 종목 목록

---

## Setup

### 1. 의존성 설치
```bash
git clone https://github.com/JAY-LIM97/sentirax.git
cd sentirax
pip install -r requirements.txt
```

### 2. GitHub Actions Secrets 설정

**Secrets** (Settings → Secrets and variables → Actions → Secrets):

| Name | Value |
|------|-------|
| `HT_API_KEY` | 실전투자 앱키 |
| `HT_API_SECRET_KEY` | 실전투자 시크릿키 |
| `HT_API_FK_KEY` | 모의투자 앱키 |
| `HT_API_FK_SECRET_KEY` | 모의투자 시크릿키 |
| `PAPER_ACCOUNT_NO` | 모의투자 계좌번호 |
| `REAL_ACCOUNT_NO` | 실전투자 계좌번호 |
| `SLACK_WEBHOOK_URL` | Slack Incoming Webhook URL |

**Variables** (Settings → Secrets and variables → Actions → Variables):

| Name | Value |
|------|-------|
| `PAPER_TRADING` | `true` (모의) / `false` (실전) |

> `PAPER_TRADING`을 `false`로 변경하면 코드 수정 없이 즉시 실전 계좌로 전환됩니다.

### 3. 로컬 실행
```bash
# 해외
python cloud/run_cloud.py --swing
python cloud/run_cloud.py --scalping-opening   # 오프닝 스캔 + 학습 + 스캘핑
python cloud/run_cloud.py --scalping-continuous # 스캘핑만 (모델 이미 있는 경우)
python cloud/run_cloud.py --retrain

# 국내
python cloud/run_cloud.py --kr-swing
python cloud/run_cloud.py --kr-scalping-opening
python cloud/run_cloud.py --kr-scalping-continuous
python cloud/run_cloud.py --kr-retrain
```

---

## Tech Stack

- **ML**: scikit-learn (LogisticRegression, GradientBoosting, **SGDClassifier**)
- **Online Learning**: `partial_fit` 기반 실거래 피드백 학습, Variance-Inverted Dynamic Blending
- **Signal Engine**: 3전략 룰 앙상블 (S1 박스돌파, S2 EMA+RSI, **S3 Liquidity Sweep**)
- **Data**: yfinance (주가 + 거시경제)
- **Trading API**: 한국투자증권 Open API (해외 + 국내)
- **Cloud**: GitHub Actions (무료, 자동 스케줄, Phase1→2 체크포인트 아티팩트)
- **Alerts**: Slack Incoming Webhook (진입/청산/일일요약/재학습완료)
- **Language**: Python 3.11+

---

## License

MIT License

## Developer

**임재현 (JAY LIM)** - [@JAY-LIM97](https://github.com/JAY-LIM97)
