"""
다중 종목 ML 백테스팅

🎯 목적:
- 여러 종목에 대해 로지스틱 회귀 학습
- 성능 비교 및 최적 종목 찾기
- 모델 저장 및 재사용

📊 테스트 종목:
- TSLA, NVDA, AAPL, MSFT
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
import pickle
from datetime import datetime
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

from core.feature_engineer import FeatureEngineer
from core.ml_trainer import MLTrainer


def train_and_backtest_stock(ticker: str, data_file: str):
    """
    단일 종목 학습 및 백테스팅

    Args:
        ticker: 종목 티커
        data_file: 데이터 파일 경로

    Returns:
        결과 딕셔너리
    """

    print(f"\n{'='*70}")
    print(f"🤖 {ticker} 학습 및 백테스팅")
    print(f"{'='*70}")

    try:
        # 데이터 로드
        df = pd.read_csv(data_file, index_col=0)
        print(f"✅ 데이터 로드: {len(df)}개 행")

        # Feature Engineering
        engineer = FeatureEngineer(label_threshold=1.0)
        X, y = engineer.prepare_ml_data(df)

        if len(X) < 20:
            print(f"⚠️  데이터가 너무 적습니다 ({len(X)}개). 최소 20개 필요")
            return None

        # Train/Test 분할
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

        # 평가
        y_pred_train = lr_model.predict(X_train_scaled)
        y_pred_test = lr_model.predict(X_test_scaled)

        train_acc = accuracy_score(y_train, y_pred_train)
        test_acc = accuracy_score(y_test, y_pred_test)
        test_precision = precision_score(y_test, y_pred_test, zero_division=0)
        test_recall = recall_score(y_test, y_pred_test, zero_division=0)
        test_f1 = f1_score(y_test, y_pred_test, zero_division=0)
        overfit_gap = train_acc - test_acc

        print(f"\n📊 모델 성능:")
        print(f"  - Train Accuracy: {train_acc*100:.2f}%")
        print(f"  - Test Accuracy: {test_acc*100:.2f}%")
        print(f"  - Test F1-Score: {test_f1:.3f}")
        print(f"  - Overfit Gap: {overfit_gap*100:.1f}%p")

        # 백테스팅
        print(f"\n📈 백테스팅 실행...")

        # Feature Engineering for all data
        df_features = engineer.create_features(df)
        X_all = df_features[X.columns.tolist()]
        valid_idx = X_all.notna().all(axis=1)
        X_clean = X_all[valid_idx]
        df_clean = df_features[valid_idx]

        # 정규화
        X_scaled = trainer.scaler.transform(X_clean)

        # 예측
        predictions = lr_model.predict(X_scaled)

        # 백테스팅 시뮬레이션
        capital = 10000
        position = 0
        buy_price = 0
        trades = []
        shares = 0

        for i in range(len(df_clean)):
            close = df_clean['Close'].iloc[i]
            signal = predictions[i]

            if signal == 1 and position == 0:  # 매수
                position = 1
                buy_price = close
                shares = capital / close
                trades.append({
                    'action': 'BUY',
                    'price': close,
                    'capital': capital
                })

            elif signal == 0 and position == 1:  # 매도
                position = 0
                capital = shares * close
                profit = capital - trades[-1]['capital']

                trades.append({
                    'action': 'SELL',
                    'price': close,
                    'capital': capital,
                    'profit': profit
                })

        # 마지막 포지션 정리
        if position == 1:
            capital = shares * df_clean['Close'].iloc[-1]
            profit = capital - trades[-1]['capital']
            trades.append({
                'action': 'SELL (종료)',
                'price': df_clean['Close'].iloc[-1],
                'capital': capital,
                'profit': profit
            })

        # 수익률 계산
        ml_return = (capital - 10000) / 10000 * 100
        bh_return = (df_clean['Close'].iloc[-1] / df_clean['Close'].iloc[0] - 1) * 100
        excess_return = ml_return - bh_return

        # 승률 계산
        sell_trades = [t for t in trades if t['action'].startswith('SELL') and 'profit' in t]
        num_trades = len(sell_trades)
        win_trades = [t for t in sell_trades if t['profit'] > 0]
        win_rate = (len(win_trades) / num_trades * 100) if num_trades > 0 else 0

        print(f"\n💰 백테스팅 결과:")
        print(f"  - ML 전략 수익률: {ml_return:+.2f}%")
        print(f"  - Buy & Hold 수익률: {bh_return:+.2f}%")
        print(f"  - 초과 수익: {excess_return:+.2f}%p")
        print(f"  - 거래 횟수: {num_trades}회")
        print(f"  - 승률: {win_rate:.1f}%")

        # 모델 저장
        models_dir = os.path.join(os.path.dirname(__file__), '..', 'models')
        os.makedirs(models_dir, exist_ok=True)

        model_data = {
            'model': lr_model,
            'scaler': trainer.scaler,
            'feature_names': X.columns.tolist(),
            'ticker': ticker,
            'train_date': datetime.now().strftime('%Y-%m-%d'),
            'performance': {
                'test_accuracy': test_acc,
                'test_f1': test_f1,
                'ml_return': ml_return,
                'excess_return': excess_return
            }
        }

        model_path = os.path.join(models_dir, f'{ticker.lower()}_logistic_regression.pkl')
        with open(model_path, 'wb') as f:
            pickle.dump(model_data, f)

        print(f"\n💾 모델 저장: {model_path}")

        return {
            'ticker': ticker,
            'success': True,
            'train_accuracy': train_acc,
            'test_accuracy': test_acc,
            'test_precision': test_precision,
            'test_recall': test_recall,
            'test_f1': test_f1,
            'overfit_gap': overfit_gap,
            'ml_return': ml_return,
            'bh_return': bh_return,
            'excess_return': excess_return,
            'num_trades': num_trades,
            'win_rate': win_rate,
            'model_path': model_path
        }

    except Exception as e:
        print(f"\n❌ 오류 발생: {e}")
        import traceback
        traceback.print_exc()

        return {
            'ticker': ticker,
            'success': False,
            'error': str(e)
        }


def main():
    print("=" * 70)
    print("🚀 다중 종목 ML 백테스팅")
    print("=" * 70)

    # 테스트할 종목들
    stocks = {
        'TSLA': 'tsla_longterm_180days.csv',
        'NVDA': 'nvda_longterm_180days.csv',
        'AAPL': 'aapl_longterm_180days.csv',
        'MSFT': 'msft_longterm_180days.csv'
    }

    data_dir = os.path.join(os.path.dirname(__file__), '..', 'data')

    results = []

    for ticker, filename in stocks.items():
        data_file = os.path.join(data_dir, filename)

        if not os.path.exists(data_file):
            print(f"\n⚠️  {ticker} 데이터 파일 없음: {filename}")
            continue

        result = train_and_backtest_stock(ticker, data_file)

        if result:
            results.append(result)

    # 결과 비교
    print("\n\n" + "=" * 70)
    print("📊 종목별 성능 비교")
    print("=" * 70)

    if results:
        # 성공한 결과만
        success_results = [r for r in results if r['success']]

        if success_results:
            # DataFrame으로 변환
            comparison_df = pd.DataFrame(success_results)

            # 주요 지표만 선택
            display_cols = ['ticker', 'test_accuracy', 'test_f1', 'overfit_gap',
                          'ml_return', 'bh_return', 'excess_return', 'win_rate']

            comparison_display = comparison_df[display_cols].copy()

            # 포맷팅
            comparison_display['test_accuracy'] = comparison_display['test_accuracy'].apply(lambda x: f"{x*100:.1f}%")
            comparison_display['test_f1'] = comparison_display['test_f1'].apply(lambda x: f"{x:.3f}")
            comparison_display['overfit_gap'] = comparison_display['overfit_gap'].apply(lambda x: f"{x*100:.1f}%p")
            comparison_display['ml_return'] = comparison_display['ml_return'].apply(lambda x: f"{x:+.2f}%")
            comparison_display['bh_return'] = comparison_display['bh_return'].apply(lambda x: f"{x:+.2f}%")
            comparison_display['excess_return'] = comparison_display['excess_return'].apply(lambda x: f"{x:+.2f}%p")
            comparison_display['win_rate'] = comparison_display['win_rate'].apply(lambda x: f"{x:.1f}%")

            # 컬럼명 한글화
            comparison_display.columns = ['종목', 'Test 정확도', 'F1-Score', '과적합',
                                        'ML 수익률', 'B&H 수익률', '초과 수익', '승률']

            print("\n")
            print(comparison_display.to_string(index=False))

            # 최고 성능 종목
            best_by_return = comparison_df.loc[comparison_df['ml_return'].idxmax()]
            best_by_f1 = comparison_df.loc[comparison_df['test_f1'].idxmax()]

            print(f"\n\n🏆 최고 성능:")
            print(f"  - 최고 수익률: {best_by_return['ticker']} ({best_by_return['ml_return']:+.2f}%)")
            print(f"  - 최고 F1-Score: {best_by_f1['ticker']} ({best_by_f1['test_f1']:.3f})")

            # 결과 저장
            results_dir = os.path.join(os.path.dirname(__file__), '..', 'results')
            os.makedirs(results_dir, exist_ok=True)

            comparison_path = os.path.join(results_dir, 'multi_stock_comparison.csv')
            comparison_df.to_csv(comparison_path, index=False)
            print(f"\n💾 비교 결과 저장: {comparison_path}")

    print("\n" + "=" * 70)
    print("✨ 다중 종목 백테스팅 완료!")
    print("=" * 70)


if __name__ == "__main__":
    main()
