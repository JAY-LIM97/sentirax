import sys
import os

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from collectors.optimized_collector import OptimizedCollector

print("""
╔══════════════════════════════════════════════════════════╗
║                                                          ║
║       🚀 Sentirax 90-Day Data Collection 🚀            ║
║                                                          ║
╚══════════════════════════════════════════════════════════╝
""")

collector = OptimizedCollector()

df = collector.collect_optimized_data(
    symbol="TSLA",
    company_name="Tesla",
    full_days=90,
    news_days=14
)

# 저장 경로 수정
save_path = os.path.join(project_root, 'data', 'tsla_optimized_90days.csv')
collector.save_data(df, save_path)

print("\n✅ 90일 데이터 수집 완료!")
print(f"   저장: {save_path}")
print("   다음: python backtest_all.py")