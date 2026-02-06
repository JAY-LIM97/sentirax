# 🚀 Sentirax - AI Stock Sentiment Analysis

AI 기반 주식 뉴스 감성 분석 시스템

## 📌 프로젝트 개요

실시간 뉴스를 수집하고 AI(Gemini/Claude)를 활용하여 특정 주식에 미칠 영향을 자동으로 분석하는 시스템입니다.

## ✨ 주요 기능

- 📰 **뉴스 자동 수집**: NewsAPI를 통한 실시간 뉴스 크롤링
- 🤖 **AI 감성 분석**: Google Gemini / Anthropic Claude 지원
- 📊 **배치 처리**: 대량의 뉴스를 효율적으로 처리
- 💡 **투자 인사이트**: 매수/매도/관망 추천 제공

## 🛠️ 기술 스택

- **Language**: Python 3.14
- **AI Models**: Google Gemini, Anthropic Claude
- **APIs**: NewsAPI, yfinance
- **Libraries**: requests, google-generativeai, anthropic

## 📦 설치 방법
```bash
# 저장소 클론
git clone https://github.com/your-username/sentirax.git
cd sentirax

# 가상환경 생성 및 활성화
python -m venv venv
.\venv\Scripts\Activate.ps1  # Windows
# source venv/bin/activate    # Mac/Linux

# 패키지 설치
pip install -r requirements.txt
```

## 🔑 환경 설정

`.env` 파일 생성:
```env
NEWSAPI_KEY=your_newsapi_key
GEMINI_API_KEY=your_gemini_key
ANTHROPIC_API_KEY=your_claude_key  # 선택사항
```

### API 키 발급

- **NewsAPI**: https://newsapi.org
- **Google Gemini**: https://aistudio.google.com/apikey
- **Anthropic Claude**: https://console.anthropic.com

## 🚀 사용 방법
```bash
python main.py
```

### 설정 변경

`config.py`에서 다음을 조정할 수 있습니다:
```python
NEWS_DAYS_BACK = 7        # 분석할 뉴스 기간
NEWS_LIMIT = 20           # 수집할 뉴스 개수
LLM_PROVIDER = 'gemini'   # AI 모델 선택
BULLISH_THRESHOLD = 0.3   # 매수 신호 기준
BEARISH_THRESHOLD = -0.3  # 매도 신호 기준
```

## 📊 출력 예시
```
🎯 Sentirax 분석 리포트: TSLA
🤖 분석 엔진: GEMINI
============================================================

📈 종합 감성 점수: 0.24 / 1.0
💡 투자 추천: 관망 (HOLD)
📊 분석된 뉴스: 20개

📰 주요 뉴스 분석 (상위 10개):
------------------------------------------------------------
🟢 뉴스 5: +0.60
   → xAI 투자로 AI 시장 확장 긍정적

🔴 뉴스 12: -0.45
   → Waymo 경쟁 심화로 자율주행 우려
...
```

## 🗂️ 프로젝트 구조
```
sentirax/
├── config.py              # 설정 관리
├── news_collector.py      # 뉴스 수집 모듈
├── sentiment_analyzer.py  # AI 감성 분석
├── main.py               # 메인 실행 파일
├── requirements.txt      # 패키지 의존성
├── .env                  # API 키 (gitignore)
└── README.md
```

## 🔮 향후 계획

- [ ] 웹 대시보드 구축 (Flask/Streamlit)
- [ ] 데이터베이스 연동 (PostgreSQL)
- [ ] 실시간 모니터링 & 알림 시스템
- [ ] 다중 주식 동시 분석
- [ ] 백테스팅 기능

## 📄 라이센스

MIT License

## 👨‍💻 개발자

**JAY** - 예비 창업자, AI/Fintech 관심

---

⭐ 이 프로젝트가 도움이 되었다면 Star를 눌러주세요!