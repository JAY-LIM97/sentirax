"""
TOP 10 종목 모델 학습 및 백테스팅

🎯 목적:
- 나스닥 거래량 TOP 10 종목 전체 학습
- 각 종목별 모델 저장
- 통합 성능 리포트 생성

📊 대상:
NVDA, INTC, TSLA, AMZN, AAPL, NFLX, MSFT, AMD, MU, GOOGL
"""

import sys
import os
import io

# Windows 한글/이모지 출력 설정
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import numpy as np
import pickle
from datetime import datetime
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, f1_score

from core.feature_engineer import FeatureEngineer

# TOP 10 종목
TOP10_TICKERS = ['NVDA', 'INTC', 'TSLA', 'AMZN', 'AAPL', 'NFLX', 'MSFT', 'AMD', 'MU', 'GOOGL']


def train_and_backtest(ticker: str, save_model: bool = True):
    """
    단일 종목 학습 및 백테스팅

    Args:
        ticker: 종목 티커
        save_model: 모델 저장 여부

    Returns:
        결과 딕셔너리
    """

    print(f"\n{'='*70}")
    print(f"🎯 {ticker} 모델 학습 및 백테스팅")
    print(f"{'='*70}")

    try:
        # 1. 데이터 로드
        print(f"\n1️⃣ 데이터 로드...")
        data_dir = os.path.join(os.path.dirname(__file__), '..', 'data')
        data_file = os.path.join(data_dir, f'{ticker.lower()}_top10_200days.csv')

        if not os.path.exists(data_file):
            print(f"❌ 데이터 파일 없음: {data_file}")
            return None

        df = pd.read_csv(data_file, index_col=0)
        df.index = pd.to_datetime(df.index, utc=True).tz_localize(None)
        print(f"  ✅ {len(df)}개 행 로드")

        # 2. Feature Engineering
        print(f"\n2️⃣ Feature Engineering...")
        engineer = FeatureEngineer(label_threshold=1.0)

        # prepare_ml_data 사용 (자동으로 features와 labels 생성)
        X_clean, y_clean = engineer.prepare_ml_data(df)

        # 해당 인덱스로 df도 필터링
        df_clean = df.loc[X_clean.index]

        feature_cols = X_clean.columns.tolist()

        # 3. Train/Test Split (80/20)
        print(f"\n3️⃣ Train/Test Split (80/20)...")
        split_idx = int(len(X_clean) * 0.8)

        X_train = X_clean.iloc[:split_idx]
        y_train = y_clean.iloc[:split_idx]
        X_test = X_clean.iloc[split_idx:]
        y_test = y_clean.iloc[split_idx:]
        df_test = df_clean.iloc[split_idx:]

        print(f"  - Train: {len(X_train)}개 ({X_train.index[0].date()} ~ {X_train.index[-1].date()})")
        print(f"  - Test:  {len(X_test)}개 ({X_test.index[0].date()} ~ {X_test.index[-1].date()})")

        # 4. 정규화
        print(f"\n4️⃣ 정규화 (StandardScaler)...")
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        print(f"  ✅ 완료")

        # 5. 모델 학습
        print(f"\n5️⃣ Logistic Regression 학습...")
        model = LogisticRegression(random_state=42, max_iter=1000)
        model.fit(X_train_scaled, y_train)
        print(f"  ✅ 학습 완료")

        # 6. 예측
        print(f"\n6️⃣ 예측 및 평가...")
        y_train_pred = model.predict(X_train_scaled)
        y_test_pred = model.predict(X_test_scaled)

        train_accuracy = accuracy_score(y_train, y_train_pred)
        test_accuracy = accuracy_score(y_test, y_test_pred)
        test_f1 = f1_score(y_test, y_test_pred, zero_division=0)

        print(f"  - Train 정확도: {train_accuracy*100:.2f}%")
        print(f"  - Test 정확도:  {test_accuracy*100:.2f}%")
        print(f"  - Test F1-Score: {test_f1:.3f}")

        # 7. 백테스팅
        print(f"\n7️⃣ 백테스팅...")

        # ML 전략 수익률
        df_test_copy = df_test.copy()
        df_test_copy['prediction'] = y_test_pred
        df_test_copy['actual_return'] = df_test_copy['Close'].pct_change().shift(-1) * 100

        # 매수 신호일 때만 수익 계산
        df_test_copy['strategy_return'] = 0.0
        df_test_copy.loc[df_test_copy['prediction'] == 1, 'strategy_return'] = df_test_copy['actual_return']

        # 누적 수익률
        ml_cumulative_return = (1 + df_test_copy['strategy_return'] / 100).prod() - 1
        ml_return_pct = ml_cumulative_return * 100

        # Buy & Hold 수익률
        bh_return_pct = (df_test['Close'].iloc[-1] / df_test['Close'].iloc[0] - 1) * 100

        # 초과 수익
        excess_return = ml_return_pct - bh_return_pct

        # 승률
        winning_trades = df_test_copy[
            (df_test_copy['prediction'] == 1) &
            (df_test_copy['actual_return'] > 0)
        ]
        total_trades = (df_test_copy['prediction'] == 1).sum()
        win_rate = len(winning_trades) / total_trades * 100 if total_trades > 0 else 0

        print(f"  - ML Strategy:  {ml_return_pct:+.2f}%")
        print(f"  - Buy & Hold:   {bh_return_pct:+.2f}%")
        print(f"  - 초과 수익:    {excess_return:+.2f}%p")
        print(f"  - 승률:         {win_rate:.1f}% ({len(winning_trades)}/{total_trades})")

        # 8. 모델 저장
        if save_model:
            print(f"\n8️⃣ 모델 저장...")
            models_dir = os.path.join(os.path.dirname(__file__), '..', 'models')
            os.makedirs(models_dir, exist_ok=True)

            model_data = {
                'model': model,
                'scaler': scaler,
                'feature_names': feature_cols,
                'train_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'performance': {
                    'train_accuracy': train_accuracy,
                    'test_accuracy': test_accuracy,
                    'test_f1': test_f1,
                    'ml_return': ml_return_pct,
                    'bh_return': bh_return_pct,
                    'excess_return': excess_return,
                    'win_rate': win_rate,
                    'num_trades': total_trades
                }
            }

            model_path = os.path.join(models_dir, f'{ticker.lower()}_top10_logistic.pkl')
            with open(model_path, 'wb') as f:
                pickle.dump(model_data, f)

            print(f"  ✅ 저장: {model_path}")

        print(f"\n✅ {ticker} 완료!")

        return {
            'ticker': ticker,
            'success': True,
            'train_accuracy': train_accuracy,
            'test_accuracy': test_accuracy,
            'test_f1': test_f1,
            'ml_return': ml_return_pct,
            'bh_return': bh_return_pct,
            'excess_return': excess_return,
            'win_rate': win_rate,
            'num_trades': total_trades,
            'train_samples': len(X_train),
            'test_samples': len(X_test)
        }

    except Exception as e:
        print(f"\n❌ 오류: {e}")
        import traceback
        traceback.print_exc()

        return {
            'ticker': ticker,
            'success': False,
            'error': str(e)
        }


def main():
    print("=" * 70)
    print("🚀 TOP 10 종목 모델 학습 및 백테스팅")
    print("=" * 70)

    print(f"\n📋 대상 종목 ({len(TOP10_TICKERS)}개):")
    for i, ticker in enumerate(TOP10_TICKERS, 1):
        print(f"  {i:2d}. {ticker}")

    results = []

    for ticker in TOP10_TICKERS:
        result = train_and_backtest(ticker, save_model=True)
        if result:
            results.append(result)

    # 최종 요약
    print("\n\n" + "=" * 70)
    print("📊 최종 요약")
    print("=" * 70)

    success_results = [r for r in results if r['success']]
    failed_results = [r for r in results if not r['success']]

    if success_results:
        print(f"\n✅ 성공 ({len(success_results)}개):")
        print(f"\n{'Ticker':<8} {'Test Acc':>10} {'F1':>8} {'ML Ret':>10} {'B&H':>10} {'Excess':>10} {'Win%':>8} {'Trades':>8}")
        print("-" * 70)

        for r in success_results:
            print(f"{r['ticker']:<8} {r['test_accuracy']*100:>9.2f}% "
                  f"{r['test_f1']:>8.3f} {r['ml_return']:>9.2f}% "
                  f"{r['bh_return']:>9.2f}% {r['excess_return']:>9.2f}%p "
                  f"{r['win_rate']:>7.1f}% {r['num_trades']:>8d}")

    if failed_results:
        print(f"\n❌ 실패 ({len(failed_results)}개):")
        for r in failed_results:
            print(f"  - {r['ticker']}: {r.get('error', 'Unknown')}")

    # 통계
    if success_results:
        print(f"\n📈 통계:")
        print(f"  - 총 종목: {len(TOP10_TICKERS)}개")
        print(f"  - 성공: {len(success_results)}개")
        print(f"  - 실패: {len(failed_results)}개")
        print(f"  - 성공률: {len(success_results)/len(TOP10_TICKERS)*100:.1f}%")

        avg_test_acc = np.mean([r['test_accuracy'] for r in success_results])
        avg_f1 = np.mean([r['test_f1'] for r in success_results])
        avg_ml_return = np.mean([r['ml_return'] for r in success_results])
        avg_bh_return = np.mean([r['bh_return'] for r in success_results])
        avg_excess = np.mean([r['excess_return'] for r in success_results])
        avg_win_rate = np.mean([r['win_rate'] for r in success_results])

        print(f"\n📊 평균:")
        print(f"  - Test 정확도: {avg_test_acc*100:.2f}%")
        print(f"  - F1-Score: {avg_f1:.3f}")
        print(f"  - ML 수익률: {avg_ml_return:+.2f}%")
        print(f"  - B&H 수익률: {avg_bh_return:+.2f}%")
        print(f"  - 초과 수익: {avg_excess:+.2f}%p")
        print(f"  - 승률: {avg_win_rate:.1f}%")

        # 최고 성능
        best_ml = max(success_results, key=lambda x: x['ml_return'])
        best_excess = max(success_results, key=lambda x: x['excess_return'])
        best_f1 = max(success_results, key=lambda x: x['test_f1'])

        print(f"\n🏆 최고 성능:")
        print(f"  - 최고 ML 수익률: {best_ml['ticker']} ({best_ml['ml_return']:+.2f}%)")
        print(f"  - 최고 초과 수익: {best_excess['ticker']} ({best_excess['excess_return']:+.2f}%p)")
        print(f"  - 최고 F1-Score: {best_f1['ticker']} ({best_f1['test_f1']:.3f})")

        # 결과 저장
        print(f"\n💾 결과 저장...")
        results_dir = os.path.join(os.path.dirname(__file__), '..', 'results')
        os.makedirs(results_dir, exist_ok=True)

        comparison_df = pd.DataFrame(success_results)
        comparison_file = os.path.join(results_dir, 'top10_model_comparison.csv')
        comparison_df.to_csv(comparison_file, index=False)

        print(f"  ✅ {comparison_file}")

    print("\n" + "=" * 70)
    print("✨ 모델 학습 완료!")
    print("=" * 70)


if __name__ == "__main__":
    main()
