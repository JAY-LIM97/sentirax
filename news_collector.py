import requests
from datetime import datetime, timedelta
from typing import List, Dict
from config import Config

class NewsCollector:
    """NewsAPI를 통한 뉴스 수집"""
    
    def __init__(self, api_key: str = None):
        self.api_key = api_key or Config.NEWSAPI_KEY
        self.base_url = "https://newsapi.org/v2/everything"
    
    def get_stock_news(self, symbol: str, company_name: str = None, 
                      days_back: int = None, limit: int = None) -> List[Dict]:
        """특정 주식의 뉴스 수집"""
        days_back = days_back or Config.NEWS_DAYS_BACK
        limit = limit or Config.NEWS_LIMIT
        
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days_back)
        
        # 검색 쿼리 최적화
        if company_name:
            query = f'({symbol} stock) OR ({company_name} stock)'
        else:
            query = f'{symbol} stock'
        
        params = {
            'q': query,
            'from': start_date.strftime('%Y-%m-%d'),
            'to': end_date.strftime('%Y-%m-%d'),
            'sortBy': 'publishedAt',
            'language': 'en',
            'apiKey': self.api_key
        }
        
        try:
            response = requests.get(self.base_url, params=params, timeout=10)
            response.raise_for_status()
            
            articles = response.json().get('articles', [])
            return articles[:limit]
            
        except requests.exceptions.RequestException as e:
            print(f"❌ 뉴스 수집 실패: {e}")
            return []
    
    def format_news_for_analysis(self, articles: List[Dict]) -> str:
        """Claude API 분석을 위한 뉴스 포맷팅"""
        if not articles:
            return "수집된 뉴스가 없습니다."
        
        formatted = "다음은 최근 뉴스 헤드라인입니다:\n\n"
        
        for i, article in enumerate(articles, 1):
            title = article.get('title', '제목 없음')
            source = article.get('source', {}).get('name', '출처 불명')
            date = article.get('publishedAt', '')
            
            formatted += f"{i}. {title}\n"
            formatted += f"   출처: {source} | 날짜: {date}\n\n"
        
        return formatted