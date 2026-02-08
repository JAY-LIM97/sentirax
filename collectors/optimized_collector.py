import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List
import time
import sys
import os
import requests

# 프로젝트 루트 추가
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from core.news_collector import NewsCollector
from core.sentiment_analyzer import SentimentAnalyzer
from core.macro_collector import MacroDataCollector
from core.config import Config

class OptimizedCollector:
    """최적화된 데이터 수집기 (뉴스 최소화 + 거시/기술 중심)"""
    
    def __init__(self):
        self.news_collector = NewsCollector()
        self.sentiment_analyzer = SentimentAnalyzer()
        self.macro_collector = MacroDataCollector()
    
    def collect_optimized_data(self, symbol: str, company_name: str,
                               full_days: int = 90,
                               news_days: int = 14) -> pd.DataFrame:
        """
        최적화된 데이터 수집
        
        Args:
            symbol: 주식 심볼 (TSLA)
            company_name: 회사명 (Tesla)
            full_days: 전체 분석 기간 (거시+기술)
            news_days: 뉴스 수집 기간 (최근 N일만)
        """
        
        print("="*60)
        print("🚀 Sentirax Optimized Data Collection")
        print("="*60)
        print(f"📊 전체 분석 기간: {full_days}일")
        print(f"📰 뉴스 수집 기간: 최근 {news_days}일만")
        print("="*60 + "\n")
        
        end_date = datetime.now()
        full_start = end_date - timedelta(days=full_days + 10)
        news_start = end_date - timedelta(days=news_days)
        
        full_start_str = full_start.strftime('%Y-%m-%d')
        news_start_str = news_start.strftime('%Y-%m-%d')
        end_str = end_date.strftime('%Y-%m-%d')
        
        # ===== STEP 1: 주가 데이터 (90일) =====
        print("💹 STEP 1/4: 주가 데이터 수집 (90일)")
        print("-"*60)
        stock_df = self._get_stock_data(symbol, full_start_str, end_str)
        print(f"✅ {len(stock_df)}일 주가 데이터 수집 완료\n")
        
        # ===== STEP 2: 거시경제 지표 (90일) =====
        print("📊 STEP 2/4: 거시경제 지표 수집 (90일)")
        print("-"*60)
        macro_df = self.macro_collector.collect_macro_data(
            full_start_str, end_str
        )
        
        # ===== STEP 3: 기술적 지표 (90일) =====
        print("📈 STEP 3/4: 기술적 지표 계산 (90일)")
        print("-"*60)
        tech_df = self.macro_collector.get_technical_indicators(
            symbol, full_start_str, end_str
        )
        
        # ===== STEP 4: 뉴스 감성 분석 (최근 14일만) =====
        print("📰 STEP 4/4: 뉴스 감성 분석 (최근 14일)")
        print("-"*60)
        print("⚡ 주간 단위 수집으로 API 절약\n")
        
        news_df = self._collect_weekly_news(
            symbol, company_name, news_start, end_date
        )
        
        # ===== 데이터 통합 =====
        print("\n🔗 데이터 통합 중...")
        print("-"*60)
        
        combined = self._merge_all_data(stock_df, macro_df, tech_df, news_df)
        
        print(f"✅ 통합 완료: {len(combined)}일 × {len(combined.columns)}개 특징\n")
        
        # 특징 목록
        print("📋 수집된 특징 (Features):")
        categories = {
            '주가': ['Close', 'Volume', 'next_day_return'],
            '뉴스': ['sentiment_score', 'news_count'],
            '거시경제': ['vix', 'treasury_10y', 'oil', 'nasdaq', 'sp500'],
            '기술지표': ['rsi', 'ma_5', 'ma_20', 'ma_50', 'volume_ratio']
        }
        
        for category, features in categories.items():
            available = [f for f in features if f in combined.columns]
            if available:
                print(f"   {category:8s}: {', '.join(available)}")
        
        print("\n" + "="*60)
        
        return combined
    
    def _get_stock_data(self, symbol: str, start: str, end: str) -> pd.DataFrame:
        """주가 데이터"""
        ticker = yf.Ticker(symbol)
        df = ticker.history(start=start, end=end)
        
        # 다음날 수익률
        df['next_day_return'] = df['Close'].pct_change(1).shift(-1) * 100
        
        # 날짜 문자열로
        df.index = df.index.strftime('%Y-%m-%d')
        
        return df[['Close', 'Volume', 'next_day_return']]
    
    def _collect_weekly_news(self, symbol: str, company_name: str,
                            start_date: datetime, end_date: datetime) -> pd.DataFrame:
        """
        주간 단위 뉴스 수집 (API 절약)
        14일 = 2주 = 2번 API 호출만!
        """
        
        news_data = []
        current = start_date
        week_num = 1
        
        while current < end_date:
            week_end = min(current + timedelta(days=7), end_date)
            
            print(f"📅 Week {week_num}: {current.strftime('%Y-%m-%d')} ~ {week_end.strftime('%Y-%m-%d')}")
            
            # 1주일치 뉴스 한 번에 수집
            articles = self._get_news_for_period(
                symbol, company_name, current, week_end
            )
            
            if articles:
                # 감성 분석
                news_text = self.news_collector.format_news_for_analysis(articles)
                sentiment = self.sentiment_analyzer.analyze_news_sentiment_batch(
                    news_text, symbol, len(articles)
                )
                
                # 이번 주 모든 날짜에 같은 감성 점수 적용
                days_in_week = (week_end - current).days
                for day_offset in range(days_in_week):
                    date = current + timedelta(days=day_offset)
                    if date.weekday() < 5:  # 주말 제외
                        news_data.append({
                            'date': date.strftime('%Y-%m-%d'),
                            'sentiment_score': sentiment['overall_score'],
                            'news_count': len(articles)
                        })
                
                print(f"   ✅ {len(articles)}개 뉴스 | 감성: {sentiment['overall_score']:+.2f}")
            else:
                print(f"   ⚠️ 뉴스 없음")
            
            current = week_end
            week_num += 1
            time.sleep(1)  # API 제한 방지
        
        df = pd.DataFrame(news_data)
        
        if len(df) > 0:
            df = df.set_index('date')
            print(f"\n✅ 뉴스 분석 완료: {len(df)}일 데이터")
        else:
            print(f"\n⚠️ 뉴스 데이터 없음 - 거시+기술 지표만 사용")
            df = pd.DataFrame(columns=['sentiment_score', 'news_count'])
        
        return df
    
    def _get_news_for_period(self, symbol: str, company_name: str,
                             start: datetime, end: datetime) -> List[Dict]:
        """기간 내 뉴스 수집"""
        
        url = "https://newsapi.org/v2/everything"
        
        params = {
            'q': f'({company_name} OR {symbol}) AND stock',
            'from': start.strftime('%Y-%m-%d'),
            'to': end.strftime('%Y-%m-%d'),
            'sortBy': 'relevancy',
            'language': 'en',
            'apiKey': Config.NEWSAPI_KEY,
            'pageSize': 30
        }
        
        try:
            response = requests.get(url, params=params, timeout=10)
            response.raise_for_status()
            data = response.json()
            return data.get('articles', [])
        except:
            return []
    
    def _merge_all_data(self, stock_df: pd.DataFrame, macro_df: pd.DataFrame,
                       tech_df: pd.DataFrame, news_df: pd.DataFrame) -> pd.DataFrame:
        """모든 데이터 통합"""
        
        # 모두 join
        combined = stock_df.join([macro_df, tech_df, news_df], how='left')
        
        # 뉴스가 없는 날은 0으로
        if 'sentiment_score' in combined.columns:
            combined['sentiment_score'] = combined['sentiment_score'].fillna(0)
        else:
            combined['sentiment_score'] = 0
            
        if 'news_count' in combined.columns:
            combined['news_count'] = combined['news_count'].fillna(0)
        else:
            combined['news_count'] = 0
        
        # 다른 NaN은 앞뒤 값으로 채우기
        combined = combined.ffill().bfill()
        
        # 마지막 행 제거 (next_day_return이 NaN)
        combined = combined[:-1]
        
        return combined
    
    def save_data(self, df: pd.DataFrame, filepath: str):
        """데이터 저장"""
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        df.to_csv(filepath)
        
        print(f"\n💾 저장 완료: {filepath}")
        print(f"   • 기간: {df.index[0]} ~ {df.index[-1]}")
        print(f"   • 크기: {len(df)} rows × {len(df.columns)} columns")
        if os.path.exists(filepath):
            print(f"   • 파일 크기: {os.path.getsize(filepath) / 1024:.1f} KB")