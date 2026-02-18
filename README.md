# Sentirax

AI-powered stock trading system — US & Korean markets, ML models, automated execution, cloud deployment.

---

## Overview

Sentirax는 머신러닝 기반 주식 자동매매 시스템입니다.

| 구분 | 해외 (미국) | 국내 (KOSPI/KOSDAQ) |
|------|------------|---------------------|
| Swing | TOP 40 대형주, 500일 LogisticRegression | 40종목 유니버스, RSI + 이동평균 |
| Scalping | 당일 급등주 TOP 20, 1분봉 GradientBoosting | 당일 급등주 TOP 20, 1분봉 GradientBoosting |
| 거래 시간 | KST 23:30~06:30 (미국 장) | KST 09:00~15:30 (한국 장) |
| 워크플로우 | 3개 (swing, scalping, retrain) | 3개 (kr-swing, kr-scalping, kr-retrain) |

---

## Architecture

```
sentirax/
├── core/
│   ├── kis_trading_api.py              # KIS API (해외 + 국내 주문/잔고)
│   ├── feature_engineer.py             # ML Feature Engineering (20+ 지표)
│   └── model_manager.py                # 모델 재학습 + 성과 추적 + 자동 교체
│
├── scripts/
│   ├── auto_trading_bot.py             # 해외 Swing 봇 (TOP 40)
│   ├── scalping_bot.py                 # 해외 Scalping 봇
│   ├── get_surging_stocks.py           # 해외 급등주 스캔 (100개 → TOP 20)
│   ├── train_scalping_model.py         # 해외 Scalping 모델 학습
│   ├── domestic_swing_bot.py           # 국내 Swing 봇 (RSI+MA)
│   ├── domestic_scalping_bot.py        # 국내 Scalping 봇
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
├── results/                     # 거래 결과 + 리포트
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
- **대상**: 당일 급등주 TOP 20 (100개 종목 스캔)
- **데이터**: 5일 1분봉 캔들
- **모델**: GradientBoostingClassifier
- **TP/SL**: 변동성 기반 동적 조정

### 국내 Swing Trading (일간)
- **대상**: KOSPI/KOSDAQ 40종목 유니버스 전체 스캔
- **전략**: RSI(14) + 5MA/20MA (ML 없이 기술적 분석, 즉시 적용)
- **매수 조건**: RSI < 35 AND 5MA > 20MA (과매도 + 상승추세)
- **매도 조건**: RSI > 70 OR 5MA < 20MA

### 국내 Scalping (분간)
- **대상**: 당일 오프닝 서지 TOP 20 (40개 유니버스 스캔)
- **모델**: GradientBoostingClassifier (`*_kr_scalping.pkl`)
- **TP/SL**: 변동성 기반 동적 조정

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
           ├─ Phase 1: 30분 대기 → 오프닝 서지 스캔 → 모델 학습 → 스캘핑 (~3hr)
           └─ Phase 2: 체크포인트 복원 → 스캘핑 계속 (~3.5hr)

[국내]
KST 08:07  [kr-daily-retrain.yml]
           ├─ 국내 급등주 스캔 (40개 유니버스)
           └─ Scalping 모델 재학습

KST 09:05  [kr-swing-trading.yml]
           └─ 국내 스윙 (RSI+MA 신호 → 매수/매도)

KST 09:35  [kr-scalping-bot.yml]
           ├─ Phase 1: 30분 대기 → 오프닝 서지 → 모델 학습 → 스캘핑 (~2.5hr)
           └─ Phase 2: 체크포인트 복원 → 스캘핑 계속 (~3hr)

* 미국/한국 휴장일 자동 스킵 (2025-2026 캘린더 내장)
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
    "swing_pct": 0.30,  "scalping_pct": 0.70,
    "swing_per_trade_pct": 0.30,  "scalping_per_trade_pct": 0.40,
    "account_balance_usd": 200000,  "account_balance_krw": 30000000
  },
  "risk": { "max_daily_loss_pct": -10.0, "stop_all_trading": false }
}
```

**주요 제어:**
- `stop_all_trading: true` → 긴급 정지
- `disabled_tickers` → 특정 종목 제외
- `forced_buy_tickers / forced_sell_tickers` → 강제 매수/매도
- `tp_override / sl_override` → TP/SL 수동 오버라이드
- `universe_tickers` (domestic_swing) → 커스텀 종목 목록 오버라이드

---

## Setup

### 1. 의존성 설치
```bash
git clone https://github.com/JAY-LIM97/sentirax.git
cd sentirax
pip install -r requirements.txt
```

### 2. 환경변수 설정 (로컬)
`.env` 파일 생성:
```env
HT_API_KEY=한국투자증권_실전_앱키
HT_API_SECRET_KEY=한국투자증권_실전_시크릿키
HT_API_FK_KEY=모의투자_앱키
HT_API_FK_SECRET_KEY=모의투자_시크릿키
```

### 3. GitHub Actions 설정

**Secrets** (Settings → Secrets and variables → Actions → Secrets):

| Name | Value |
|------|-------|
| `HT_API_KEY` | 실전투자 앱키 |
| `HT_API_SECRET_KEY` | 실전투자 시크릿키 |
| `HT_API_FK_KEY` | 모의투자 앱키 |
| `HT_API_FK_SECRET_KEY` | 모의투자 시크릿키 |
| `PAPER_ACCOUNT_NO` | 모의투자 계좌번호 |
| `REAL_ACCOUNT_NO` | 실전투자 계좌번호 |

**Variables** (Settings → Secrets and variables → Actions → Variables):

| Name | Value |
|------|-------|
| `PAPER_TRADING` | `true` (모의) / `false` (실전) |

> `PAPER_TRADING`을 `false`로 변경하면 코드 수정 없이 즉시 실전 계좌로 전환됩니다.

### 4. 로컬 실행
```bash
# 해외
python cloud/run_cloud.py --swing
python cloud/run_cloud.py --scalping-opening
python cloud/run_cloud.py --retrain

# 국내
python cloud/run_cloud.py --kr-swing
python cloud/run_cloud.py --kr-scalping-opening
python cloud/run_cloud.py --kr-retrain
```

---

## Tech Stack

- **ML**: scikit-learn (LogisticRegression, GradientBoosting)
- **Data**: yfinance (주가 + 거시경제)
- **Trading API**: 한국투자증권 Open API (해외 + 국내)
- **Cloud**: GitHub Actions (무료, 자동 스케줄)
- **Language**: Python 3.11+

---

## License

MIT License

## Developer

**임재현 (JAY LIM)** - [@JAY-LIM97](https://github.com/JAY-LIM97)
