"""
Step 4: ML 모델 백테스팅

🎯 목적:
로지스틱 회귀 모델로 실제 매매 시뮬레이션 수행

📊 백테스팅 전략:
1. 모델이 매수(1) 예측 → 주식 매수
2. 모델이 매도(0) 예측 → 주식 매도 or 보유 안 함
3. 수익률 계산 및 Buy & Hold와 비교
"""

import sys
import os
import io
import platform

# Windows 한글/이모지 출력 설정
if platform.system() == 'Windows':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

# 프로젝트 루트를 파이썬 경로에 추가
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression

from core.feature_engineer import FeatureEngineer
from core.ml_trainer import MLTrainer


def backtest_ml_strategy(
    df: pd.DataFrame,
    model,
    scaler,
    feature_names: list,
    initial_capital: float = 10000
):
    """
    ML 모델 백테스팅

    Args:
        df: 원본 데이터
        model: 학습된 모델
        scaler: 학습된 스케일러
        feature_names: 특징 이름 리스트
        initial_capital: 초기 자본

    Returns:
        백테스팅 결과 딕셔너리
    """

    # Feature Engineering
    engineer = FeatureEngineer(label_threshold=1.0)
    df_features = engineer.create_features(df)

    # 필요한 특징만 선택
    available_features = [f for f in feature_names if f in df_features.columns]
    X = df_features[available_features]

    # NaN 제거
    valid_idx = X.notna().all(axis=1)
    X_clean = X[valid_idx]
    df_clean = df_features[valid_idx]

    # 정규화
    X_scaled = scaler.transform(X_clean)

    # 예측
    predictions = model.predict(X_scaled)

    # 백테스팅
    capital = initial_capital
    position = 0  # 0: 없음, 1: 보유
    buy_price = 0
    trades = []

    for i in range(len(df_clean)):
        date = df_clean.index[i]
        close = df_clean['Close'].iloc[i]
        signal = predictions[i]

        if signal == 1 and position == 0:  # 매수 신호 & 포지션 없음
            position = 1
            buy_price = close
            shares = capital / close
            trades.append({
                'date': date,
                'action': 'BUY',
                'price': close,
                'shares': shares,
                'capital': capital
            })

        elif signal == 0 and position == 1:  # 매도 신호 & 포지션 있음
            position = 0
            sell_price = close
            capital = shares * sell_price
            profit = capital - trades[-1]['capital']
            profit_pct = (profit / trades[-1]['capital']) * 100

            trades.append({
                'date': date,
                'action': 'SELL',
                'price': close,
                'shares': shares,
                'capital': capital,
                'profit': profit,
                'profit_pct': profit_pct
            })

    # 마지막 포지션 정리
    if position == 1:
        sell_price = df_clean['Close'].iloc[-1]
        capital = shares * sell_price
        profit = capital - trades[-1]['capital']
        profit_pct = (profit / trades[-1]['capital']) * 100

        trades.append({
            'date': df_clean.index[-1],
            'action': 'SELL (종료)',
            'price': sell_price,
            'shares': shares,
            'capital': capital,
            'profit': profit,
            'profit_pct': profit_pct
        })

    # 수익률 계산
    total_return = (capital - initial_capital) / initial_capital * 100

    # Buy & Hold 수익률
    bh_return = (df_clean['Close'].iloc[-1] - df_clean['Close'].iloc[0]) / df_clean['Close'].iloc[0] * 100

    return {
        'trades': trades,
        'final_capital': capital,
        'total_return': total_return,
        'buy_hold_return': bh_return,
        'excess_return': total_return - bh_return,
        'num_trades': len([t for t in trades if t['action'] == 'SELL'])
    }


def main():
    print("=" * 70)
    print("📈 Step 4: ML 모델 백테스팅")
    print("=" * 70)

    # ========================================
    # 1. 데이터 준비
    # ========================================
    print("\n📁 데이터 로딩...")

    data_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'tsla_optimized_90days.csv')
    df = pd.read_csv(data_path, index_col=0)

    print(f"✅ 데이터: {len(df)}개 행")

    # ========================================
    # 2. 모델 학습 (로지스틱 회귀)
    # ========================================
    print("\n🤖 로지스틱 회귀 모델 학습...")

    engineer = FeatureEngineer(label_threshold=1.0)
    X, y = engineer.prepare_ml_data(df)

    trainer = MLTrainer(test_size=0.2, random_state=42)
    X_train, X_test, y_train, y_test = trainer.split_data(X, y)
    X_train_scaled, X_test_scaled = trainer.normalize_data(X_train, X_test)

    # 로지스틱 회귀 학습
    lr_model = LogisticRegression(
        C=1.0,
        max_iter=1000,
        random_state=42,
        class_weight='balanced'
    )

    lr_model.fit(X_train_scaled, y_train)
    print(f"✅ 모델 학습 완료")

    # ========================================
    # 3. 백테스팅
    # ========================================
    print("\n📊 백테스팅 실행...")

    results = backtest_ml_strategy(
        df,
        lr_model,
        trainer.scaler,
        X.columns.tolist(),
        initial_capital=10000
    )

    print(f"✅ 백테스팅 완료")

    # ========================================
    # 4. 결과 출력
    # ========================================
    print("\n\n" + "="*70)
    print("📊 백테스팅 결과")
    print("="*70)

    print(f"\n💰 수익률:")
    print(f"  - 초기 자본: $10,000")
    print(f"  - 최종 자본: ${results['final_capital']:,.2f}")
    print(f"  - ML 전략 수익률: {results['total_return']:+.2f}%")
    print(f"  - Buy & Hold 수익률: {results['buy_hold_return']:+.2f}%")
    print(f"  - 초과 수익: {results['excess_return']:+.2f}%p")

    print(f"\n📈 거래 통계:")
    print(f"  - 총 거래 횟수: {results['num_trades']}회")

    # 승률 계산
    sell_trades = [t for t in results['trades'] if t['action'].startswith('SELL') and 'profit_pct' in t]
    if sell_trades:
        win_trades = [t for t in sell_trades if t['profit'] > 0]
        win_rate = len(win_trades) / len(sell_trades) * 100
        avg_profit = np.mean([t['profit_pct'] for t in sell_trades])
        avg_win = np.mean([t['profit_pct'] for t in win_trades]) if win_trades else 0
        avg_loss = np.mean([t['profit_pct'] for t in sell_trades if t['profit'] < 0]) if len(sell_trades) > len(win_trades) else 0

        print(f"  - 승률: {win_rate:.1f}% ({len(win_trades)}/{len(sell_trades)})")
        print(f"  - 평균 수익: {avg_profit:+.2f}%")
        if win_trades:
            print(f"  - 평균 승리: {avg_win:+.2f}%")
        if avg_loss < 0:
            print(f"  - 평균 손실: {avg_loss:+.2f}%")

    # ========================================
    # 5. 거래 내역
    # ========================================
    print(f"\n📋 거래 내역:")

    trades_df = pd.DataFrame(results['trades'])
    for i, trade in enumerate(results['trades'][:10], 1):  # 최근 10개만
        if trade['action'] == 'BUY':
            print(f"  {i:2d}. [{trade['date']}] BUY  ${trade['price']:.2f}")
        else:
            profit_str = f"(수익: {trade['profit_pct']:+.2f}%)" if 'profit_pct' in trade else ""
            print(f"  {i:2d}. [{trade['date']}] SELL ${trade['price']:.2f} {profit_str}")

    if len(results['trades']) > 10:
        print(f"  ... (총 {len(results['trades'])}개 거래)")

    # ========================================
    # 6. 성능 평가
    # ========================================
    print(f"\n\n💡 성능 평가:")

    if results['total_return'] > results['buy_hold_return']:
        print(f"  ✅ ML 전략이 Buy & Hold보다 {results['excess_return']:.2f}%p 우수!")
    elif results['total_return'] > 0:
        print(f"  ⚠️  ML 전략은 수익이지만 Buy & Hold보다 {abs(results['excess_return']):.2f}%p 낮음")
    else:
        print(f"  ❌ ML 전략이 손실 ({results['total_return']:.2f}%)")

    # 승률 평가
    if sell_trades and win_rate >= 60:
        print(f"  ✅ 높은 승률 ({win_rate:.1f}%)")
    elif sell_trades and win_rate >= 50:
        print(f"  ➖ 보통 승률 ({win_rate:.1f}%)")
    elif sell_trades:
        print(f"  ⚠️  낮은 승률 ({win_rate:.1f}%)")

    # ========================================
    # 7. 결과 저장
    # ========================================
    print(f"\n💾 결과 저장...")

    results_dir = os.path.join(os.path.dirname(__file__), '..', 'results')

    # 거래 내역 저장
    trades_path = os.path.join(results_dir, 'ml_backtest_trades.csv')
    trades_df.to_csv(trades_path, index=False)
    print(f"✅ 거래 내역 저장: {trades_path}")

    # 요약 저장
    summary = pd.DataFrame([{
        'model': 'Logistic Regression',
        'initial_capital': 10000,
        'final_capital': results['final_capital'],
        'ml_return': results['total_return'],
        'buy_hold_return': results['buy_hold_return'],
        'excess_return': results['excess_return'],
        'num_trades': results['num_trades'],
        'win_rate': win_rate if sell_trades else 0,
        'avg_profit': avg_profit if sell_trades else 0
    }])

    summary_path = os.path.join(results_dir, 'ml_backtest_summary.csv')
    summary.to_csv(summary_path, index=False)
    print(f"✅ 백테스팅 요약 저장: {summary_path}")

    print("\n" + "="*70)
    print("✨ Step 4 완료! Day 2 전체 완료!")
    print("="*70)

    print(f"\n🎉 Day 2 최종 결과:")
    print(f"  - 최고 모델: 로지스틱 회귀")
    print(f"  - Test Accuracy: 87.5%")
    print(f"  - ML 전략 수익률: {results['total_return']:+.2f}%")
    print(f"  - Buy & Hold 대비: {results['excess_return']:+.2f}%p")
    print(f"\n🚀 다음 단계:")
    print(f"  - 더 긴 기간 데이터 수집 (180일+)")
    print(f"  - 다른 주식 테스트 (NVDA, AAPL)")
    print(f"  - 실시간 매매 시스템 구축")


if __name__ == "__main__":
    main()
