# core/__init__.py
#from .config import Config
from .news_collector import NewsCollector
from .sentiment_analyzer import SentimentAnalyzer
from .macro_collector import MacroDataCollector
from .evaluator import BacktestEvaluator

__all__ = [
    'Config',
    'NewsCollector',
    'SentimentAnalyzer',
    'MacroDataCollector',
    'BacktestEvaluator'
]