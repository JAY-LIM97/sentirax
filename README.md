# Sentirax

AI-powered US stock trading system with ML models, automated execution, and cloud deployment.

---

## Overview

Sentirax는 머신러닝 기반 미국 주식 자동매매 시스템입니다.
- **Swing Trading**: TOP 20 대형주 대상, 500일 백테스팅 기반 LogisticRegression 모델
- **Scalping**: 일별 급등주 TOP 20 대상, 1분봉 GradientBoosting 모델
- **GitHub Actions**: 매일 자동 실행 (KST 22:00~06:30), 휴장일 자동 스킵

---

## Architecture

```
sentirax/
├── core/                        # 핵심 모듈
│   ├── kis_trading_api.py       # 한국투자증권 해외주식 API
│   ├── feature_engineer.py      # ML Feature Engineering (20+ 지표)
│   └── model_manager.py         # 모델 재학습 + 성과 추적 + 자동 교체
│
├── scripts/                     # 실행 스크립트
│   ├── auto_trading_bot.py      # Swing 자동매매 봇
│   ├── scalping_bot.py          # Scalping 자동매매 봇
│   ├── get_surging_stocks.py    # 급등주 TOP 20 스캔
│   ├── train_scalping_model.py  # Scalping 모델 학습
│   └── train_top20_500days.py   # Swing 모델 학습 (500일)
│
├── cloud/                       # 클라우드 실행
│   ├── run_cloud.py             # GitHub Actions 헤드리스 실행기
│   └── requirements-cloud.txt   # 클라우드 의존성
│
├── config/
│   └── strategy.json            # 실시간 전략 설정 (GitHub 웹에서 수정 가능)
│
├── models/                      # 학습된 ML 모델 (.pkl)
├── results/                     # 거래 결과 + 리포트
│
├── .github/workflows/           # GitHub Actions 자동화
│   ├── swing-trading.yml        # 스윙 트레이딩 (KST 23:30)
│   ├── scalping-bot.yml         # 스캘핑 (KST 23:35~06:30, 2 Phase)
│   └── daily-retrain.yml        # 일일 모델 재학습 (KST 22:00)
│
├── .env                         # API 키 (비공개)
└── RUN_ALL.bat                  # 로컬 실행용 (Swing + Scalping)
```

---

## Trading Models

### Swing Trading (일간)
- **대상**: NVDA, AAPL, GOOGL, MSFT, AMZN, META, TSLA, AVGO 등 14종목
- **데이터**: 500일 일봉 + 거시경제 지표 (VIX, 국채, 유가, 나스닥, S&P500)
- **모델**: LogisticRegression + StandardScaler
- **Features**: RSI, MA(5/20/50), 변동성, 거래량 비율, 거시경제 변화율 등 20+개
- **레이블**: 다음날 +1% 이상 = 매수(1), -1% 이하 = 매도(0)

### Scalping (분간)
- **대상**: 당일 급등주 TOP 20 (100개 종목 스캔)
- **데이터**: 5일 1분봉 캔들
- **모델**: GradientBoostingClassifier
- **TP/SL**: 변동성 기반 동적 조정 (TP max 5%, SL max 3%)
- **보유 시간**: 최대 60분
- **품질 필터**: 정확도 >= 50% AND 승률 >= 40% 인 모델만 저장

---

## Daily Execution Flow (GitHub Actions)

```
KST 22:00  [daily-retrain.yml]
           ├─ Performance tracking (전일 거래 성과)
           ├─ Model auto-switch (성과 < 25점 → 비활성화)
           ├─ Retrain 20 swing models (최신 데이터)
           └─ Git commit updated models

KST 23:30  [swing-trading.yml]
           └─ Swing trading (14종목 매수/매도)

KST 23:35  [scalping-bot.yml]
           ├─ Phase 1 (3.5hr):
           │  ├─ Surging stocks scan (100종목 → TOP 20)
           │  ├─ Train scalping models
           │  └─ Continuous scalping (210분)
           │
           └─ Phase 2 (3.5hr):
              └─ Continuous scalping (210분, 체크포인트 이어받기)

* 미국 휴장일 자동 스킵 (NYSE 2025-2026 캘린더 내장)
```

---

## Live Strategy Control

`config/strategy.json`을 GitHub 웹에서 수정하면 다음 사이클부터 즉시 반영됩니다.

```json
{
  "swing": {
    "enabled": true,
    "max_positions": 14,
    "min_probability": 0.55,
    "order_quantity": 1,
    "disabled_tickers": [],
    "forced_buy_tickers": [],
    "forced_sell_tickers": []
  },
  "scalping": {
    "enabled": true,
    "max_positions": 5,
    "min_probability": 0.55,
    "scan_interval_seconds": 60,
    "tp_override": null,
    "sl_override": null,
    "only_tickers": []
  },
  "risk": {
    "max_daily_loss_pct": -10.0,
    "stop_all_trading": false,
    "paper_trading": true
  }
}
```

**주요 제어:**
- `stop_all_trading: true` → 긴급 정지 (모든 포지션 청산)
- `disabled_tickers` → 특정 종목 제외
- `forced_buy_tickers` → 강제 매수
- `tp_override / sl_override` → TP/SL 수동 오버라이드

---

## Monitoring

### GitHub Actions UI
- **Job Summary**: 각 워크플로우 실행 결과가 Actions 탭에 표시
- **Artifacts**: `logs/` + `results/` 파일 다운로드 (7~14일 보관)

### Results Files
```
results/
├── daily_reports/report_YYYYMMDD.json    # 일일 실행 요약
├── scalping_summary_YYYYMMDD.json        # 스캘핑 거래 내역
├── scalping_checkpoint.json              # 포지션 체크포인트
├── performance/
│   ├── trade_history.csv                 # 전체 거래 기록
│   ├── daily_summary.csv                 # 일일 P&L
│   └── model_scores.json                 # 모델 점수 (0-100)
└── surging_stocks_today.csv              # 오늘의 급등주
```

---

## Setup

### 1. 의존성 설치
```bash
git clone https://github.com/JAY-LIM97/sentirax.git
cd sentirax
pip install -r requirements.txt
```

### 2. 환경변수 설정
`.env` 파일 생성:
```env
HT_API_KEY=한국투자증권_앱키
HT_API_SECRET_KEY=한국투자증권_시크릿키
HT_API_FK_KEY=모의투자_앱키
HT_API_FK_SECRET_KEY=모의투자_시크릿키
```

### 3. GitHub Actions 설정
1. Repository를 **public**으로 설정
2. Settings > Secrets에 위 4개 키 등록
3. Actions 탭에서 워크플로우 활성화

### 4. 로컬 실행
```bash
# Swing + Scalping 동시 실행 (Windows)
RUN_ALL.bat

# 개별 실행
python cloud/run_cloud.py --swing
python cloud/run_cloud.py --scalping-continuous
python cloud/run_cloud.py --retrain
python cloud/run_cloud.py --dashboard
```

---

## Tech Stack

- **ML**: scikit-learn (LogisticRegression, GradientBoosting)
- **Data**: yfinance (주가 + 거시경제)
- **Trading API**: 한국투자증권 Open API (해외주식)
- **Cloud**: GitHub Actions (무료, 월~금 자동 실행)
- **Language**: Python 3.11+

---

## License

MIT License

## Developer

**임재현 (JAY LIM)** - [@JAY-LIM97](https://github.com/JAY-LIM97)
