from news_collector import NewsCollector
from sentiment_analyzer import SentimentAnalyzer
from config import Config
import yfinance as yf

def main():
    """Sentirax 메인 실행 함수"""
    
    print("🚀 Sentirax AI 주식 분석 시스템 시작...\n")
    
    # 분석할 주식 설정
    symbol = "TSLA"
    company_name = "Tesla"
    
    # 1. 현재가 가져오기
    print(f"📊 {symbol} 현재가 조회 중...")
    try:
        ticker = yf.Ticker(symbol)
        current_price = ticker.history(period="1d")['Close'].iloc[-1]
        print(f"✅ 현재가: ${current_price:.2f} USD\n")
    except:
        current_price = "N/A"
        print("⚠️ 현재가 조회 실패\n")
    
    # 2. 뉴스 수집
    print(f"📰 최근 {Config.NEWS_DAYS_BACK}일간 뉴스 수집 중...")
    collector = NewsCollector()
    news_articles = collector.get_stock_news(symbol, company_name)
    
    if not news_articles:
        print("❌ 수집된 뉴스가 없습니다.")
        return
    
    news_count = len(news_articles)
    print(f"✅ {news_count}개의 뉴스 수집 완료\n")
    
    # 3. AI 감성 분석 (배치)
    print(f"🤖 AI 분석 시작...\n")
    analyzer = SentimentAnalyzer()
    
    news_text = collector.format_news_for_analysis(news_articles)
    
    # 배치 분석 사용
    analysis_result = analyzer.analyze_news_sentiment_batch(
        news_text, 
        symbol, 
        news_count
    )
    
    # 4. 결과 출력
    report = analyzer.format_analysis_report(analysis_result, symbol)
    print(report)
    
    # 5. 액션 아이템
    recommendation = analysis_result['recommendation']
    if recommendation == 'BUY':
        print("✅ 권장 행동: 매수 포지션 고려")
    elif recommendation == 'SELL':
        print("⚠️ 권장 행동: 매도 또는 포지션 축소 고려")
    else:
        print("⏸️ 권장 행동: 추가 정보 수집 후 재평가")

if __name__ == "__main__":
    main()