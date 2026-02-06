import os
from dotenv import load_dotenv

load_dotenv()

class Config:
    """API 키 및 설정 관리"""
    
    # API Keys
    NEWSAPI_KEY = os.getenv("NEWSAPI_KEY")
    ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
    GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")  # 추가
    
    # LLM 선택 ('claude' 또는 'gemini')
    LLM_PROVIDER = 'gemini'  # 기본값을 gemini로 변경
    
    # 분석 설정
    NEWS_DAYS_BACK = 7
    NEWS_LIMIT = 20
    
    # 감성 점수 기준
    BULLISH_THRESHOLD = 0.3
    BEARISH_THRESHOLD = -0.3