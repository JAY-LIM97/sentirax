import sys
import os

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

import pandas as pd
from core.evaluator import BacktestEvaluator

print("""
╔══════════════════════════════════════════════════════════╗
║                                                          ║
║          📊 Sentirax Backtesting System 📊              ║
║                                                          ║
╚══════════════════════════════════════════════════════════╝
""")

# 데이터 파일 찾기
data_dir = os.path.join(project_root, 'data')
data_files = [
    os.path.join(data_dir, 'tsla_optimized_90days.csv'),
    os.path.join(data_dir, 'tsla_backtest_30days.csv')
]

data_file = None
for f in data_files:
    if os.path.exists(f):
        data_file = f
        break

if not data_file:
    print("❌ 데이터 파일이 없습니다!")
    print("먼저 scripts/collect_90days.py를 실행하세요.")
    exit()

print(f"📂 데이터: {os.path.basename(data_file)}\n")

df = pd.read_csv(data_file, index_col=0)

print(f"📊 데이터 정보:")
print(f"   기간: {df.index[0]} ~ {df.index[-1]}")
print(f"   일수: {len(df)}일")
print(f"   특징: {len(df.columns)}개\n")

evaluator = BacktestEvaluator(df)

# 상관관계
evaluator.calculate_correlation()

# 전략 비교
evaluator.compare_all_strategies(initial_capital=10000)