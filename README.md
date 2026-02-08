# 🚀 Sentirax

AI-powered stock trading system with advanced backtesting

---

## 📊 Project Overview

Sentirax는 AI 기반 주식 매매 시스템으로, 거시경제 지표와 기술적 분석을 결합하여 최적의 매매 타이밍을 포착합니다.

### ✨ 핵심 기능

- **📊 거시경제 지표 분석**: VIX, 국채수익률, 유가, 나스닥/S&P500
- **📈 기술적 지표**: RSI, 이동평균선, 거래량 분석
- **🤖 다중 LLM 지원**: Gemini, Claude, Groq
- **🎯 백테스팅 시스템**: 90일 이상 데이터 기반 성과 검증
- **📄 자동 리포트**: PDF 백테스팅 리포트 생성

---

## 🏆 검증된 성과

### 90일 백테스팅 결과 (TSLA, 2025-10-31 ~ 2026-02-06)전략           수익률     거래    승률
─────────────────────────────────
Volume        +0.82%     2회     50%
Buy & Hold    -9.95%     1회     0%
─────────────────────────────────
초과 수익:    +10.77%

**핵심 발견:**
- 거래량 기반 전략이 하락장에서 10.77% 초과 수익 달성
- 거래량이 주가 예측에 가장 강력한 지표 (상관관계 +0.245)
- Profit Factor: 1.65 (수익이 손실의 1.65배)

---

## 🏗️ 프로젝트 구조sentirax/
├── 📁 core/                    # 핵심 모듈
│   ├── config.py              # 설정 관리
│   ├── news_collector.py      # 뉴스 수집
│   ├── sentiment_analyzer.py  # AI 감성 분석
│   ├── macro_collector.py     # 거시경제 지표
│   └── evaluator.py           # 백테스팅 엔진
│
├── 📁 collectors/             # 데이터 수집
│   └── optimized_collector.py # 통합 수집기
│
├── 📁 scripts/                # 실행 스크립트
│   ├── main.py               # 실시간 분석
│   ├── collect_90days.py     # 90일 데이터 수집
│   ├── backtest_all.py       # 백테스팅 실행
│   ├── report_generator.py   # PDF 리포트 생성
│   └── visualizer.py         # 차트 생성
│
├── 📁 data/                   # 데이터 저장소
├── 📁 results/                # 결과물 (차트, 리포트)
├── 📁 docs/                   # 문서
│   ├── CHANGELOG.md
│   └── DEVELOPMENT.md
│
├── .env                       # API 키 (비공개)
├── .gitignore
├── LICENSE                    # MIT License
├── README.md
└── requirements.txt

---

## 🚀 Quick Start

### 1. 설치
```bash저장소 클론
git clone https://github.com/JAY-LIM97/sentirax.git
cd sentirax가상환경 생성
python -m venv venv
.\venv\Scripts\Activate.ps1  # Windows
source venv/bin/activate      # Mac/Linux패키지 설치
pip install -r requirements.txt

### 2. 환경 설정

`.env` 파일 생성:
```envNews API
NEWSAPI_KEY=your_newsapi_keyAI Models
GEMINI_API_KEY=your_gemini_key
ANTHROPIC_API_KEY=your_claude_key
GROQ_API_KEY=your_groq_keyLLM Provider (gemini/claude/groq)
LLM_PROVIDER=gemini

### 3. 실행
```bashcd scripts실시간 분석
python main.py90일 데이터 수집
python collect_90days.py백테스팅
python backtest_all.py리포트 생성
python report_generator.py차트 생성
python visualizer.py

---

## 📊 지원 기능

### 데이터 수집
- **뉴스**: NewsAPI (최근 30일)
- **거시경제**: yfinance (VIX, 국채, 유가 등)
- **기술지표**: RSI, MA, 볼린저밴드, 거래량

### 분석 전략
1. **감성 기반**: AI 뉴스 감성 분석
2. **거래량 기반**: 거래량 급증/감소 패턴
3. **복합 신호**: 다중 지표 결합

### 백테스팅
- 90일 이상 데이터 지원
- 손절/익절 전략
- 승률, Profit Factor, 최대낙폭 계산
- Buy & Hold 대비 성과 비교

---

## 🎯 거래 전략

### Volume Strategy (권장)
```python매수 조건:

거래량 1.3배 이상 급증
VIX < 25 (안정적 시장)
매도 조건:

거래량 0.9배 이하 감소
또는 손절 -10%
또는 익절 +20%


**성과:** Buy & Hold 대비 +10.77% 초과 수익 (90일)

---

## 📈 사용 예시

### 실시간 분석
```pythonfrom core.news_collector import NewsCollector
from core.sentiment_analyzer import SentimentAnalyzer뉴스 수집
collector = NewsCollector()
news = collector.get_stock_news("TSLA", "Tesla", limit=20)AI 감성 분석
analyzer = SentimentAnalyzer()
result = analyzer.analyze_news_sentiment_batch(news_text, "TSLA", 20)print(f"감성 점수: {result['overall_score']}")
print(f"추천: {result['recommendation']}")

### 백테스팅
```pythonfrom core.evaluator import BacktestEvaluator
import pandas as pd데이터 로드
df = pd.read_csv('data/tsla_optimized_90days.csv', index_col=0)평가
evaluator = BacktestEvaluator(df)
evaluator.compare_all_strategies(initial_capital=10000)

---

## ⚙️ 설정

`core/config.py`:
```python뉴스 수집
NEWS_DAYS_BACK = 7
NEWS_LIMIT = 20거래 전략
BULLISH_THRESHOLD = 0.3
BEARISH_THRESHOLD = -0.3LLM 설정
LLM_PROVIDER = 'gemini'

---

## 📊 백테스팅 지표

- **Total Return**: 총 수익률
- **Excess Return**: Buy & Hold 대비 초과 수익
- **Win Rate**: 승률
- **Profit Factor**: 평균 수익 / 평균 손실
- **Max Drawdown**: 최대 낙폭
- **Sharpe Ratio**: 위험 조정 수익률

---

## 🔮 로드맵

### Phase 1: 핵심 시스템 ✅ (완료)
- [x] AI 감성 분석
- [x] 거시경제 지표 수집
- [x] 백테스팅 엔진
- [x] PDF 리포트 생성

### Phase 2: 전략 고도화 🔄 (진행 중)
- [ ] 정교한 매매 전략 (손절/익절)
- [ ] 포지션 사이징
- [ ] 리스크 관리

### Phase 3: 실전 배포 ⏳ (예정)
- [ ] 실시간 모니터링
- [ ] 텔레그램 알림
- [ ] 웹 대시보드
- [ ] 자동 매매 연동

### Phase 4: 확장 🌟 (미래)
- [ ] 다중 종목 분석
- [ ] 포트폴리오 최적화
- [ ] 소셜 감성 분석 (Reddit, Twitter)
- [ ] 머신러닝 모델

---

## 🛠️ 기술 스택

- **Language**: Python 3.14+
- **AI Models**: Google Gemini, Anthropic Claude, Groq
- **Data**: NewsAPI, yfinance
- **Analysis**: pandas, numpy, matplotlib, seaborn
- **Reporting**: FPDF

---

## 📄 라이선스

MIT License - 자세한 내용은 [LICENSE](LICENSE) 파일 참고

---

## 👨‍💻 개발자

**임재현 (JAY LIM)**
- GitHub: [@JAY-LIM97](https://github.com/JAY-LIM97)
- Project: Sentirax - AI Stock Trading System

---

## 🙏 기여

이슈 및 PR 환영합니다!

1. Fork the Project
2. Create your Feature Branch (`git checkout -b feature/AmazingFeature`)
3. Commit your Changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the Branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

---

---

## 📞 문의

프로젝트 관련 문의사항은 GitHub Issues를 이용해주세요.

---

**⭐ Star this repo if you find it useful!**