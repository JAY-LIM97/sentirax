import sys
import os

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')

# ... (기존 코드)

# 파일 경로 수정
data_files = [
    '../data/tsla_optimized_90days.csv',
    '../data/tsla_backtest_30days.csv'
]

# 저장 경로 수정
os.makedirs('../results', exist_ok=True)
chart_path = '../results/backtest_chart.png'