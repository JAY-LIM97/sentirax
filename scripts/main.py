import sys
import os

# 프로젝트 루트 추가
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from core.news_collector import NewsCollector
from core.sentiment_analyzer import SentimentAnalyzer
from core.config import Config
import yfinance as yf

def main():
    """Sentirax 실시간 분석"""
    
    print("🚀 Sentirax 실시간 주식 분석\n")
    
    symbol = "TSLA"
    company_name = "Tesla"
    
    # 현재가
    print(f"📊 {symbol} 현재가 조회 중...")
    try:
        ticker = yf.Ticker(symbol)
        current_price = ticker.history(period="1d")['Close'].iloc[-1]
        print(f"✅ 현재가: ${current_price:.2f}\n")
    except:
        print("⚠️ 현재가 조회 실패\n")
    
    # 뉴스 수집
    print(f"📰 최근 뉴스 수집 중...")
    collector = NewsCollector()
    news = collector.get_stock_news(symbol, company_name, limit=Config.NEWS_LIMIT)
    
    if not news:
        print("❌ 뉴스가 없습니다.")
        return
    
    print(f"✅ {len(news)}개 뉴스 수집\n")
    
    # 미리보기
    print("📋 수집된 뉴스:")
    print("-" * 60)
    for i, article in enumerate(news[:3], 1):
        print(f"{i}. {article['title']}")
        print(f"   출처: {article['source']['name']}\n")
    
    # 감성 분석
    print("🤖 AI 감성 분석 시작...\n")
    analyzer = SentimentAnalyzer()
    news_text = collector.format_news_for_analysis(news)
    result = analyzer.analyze_news_sentiment_batch(news_text, symbol, len(news))
    
    # 리포트
    report = analyzer.format_analysis_report(result, symbol)
    print(report)
    
    # 추천
    rec = result['recommendation']
    if rec == 'BUY':
        print("✅ 권장 행동: 매수 포지션 고려")
    elif rec == 'SELL':
        print("⚠️ 권장 행동: 매도 또는 포지션 축소")
    else:
        print("⏸️ 권장 행동: 추가 정보 수집 후 재평가")

if __name__ == "__main__":
    main()