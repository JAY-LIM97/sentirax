"""
TOP 20 종목 500일 데이터 모델 학습 및 백테스팅

🎯 목적:
- 엄격한 백테스팅
- 손실 위험 최소화
- 높은 정확도만 선택
"""

import sys
import os
import io

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import numpy as np
import pickle
from datetime import datetime
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score

from core.feature_engineer import FeatureEngineer

# TOP 20 종목
TOP20_TICKERS = ['NVDA', 'AAPL', 'GOOGL', 'MSFT', 'AMZN', 'META', 'TSLA', 'AVGO', 'WMT', 'LLY',
                 'JPM', 'XOM', 'JNJ', 'ORCL', 'COST', 'ABBV', 'HD', 'BAC', 'PG', 'CVX']


def train_and_backtest(ticker: str) -> dict:
    """모델 학습 및 백테스팅"""

    print(f"\n{'='*70}")
    print(f"🎯 {ticker} 모델 학습 및 백테스팅")
    print(f"{'='*70}")

    try:
        # 1. 데이터 로드
        print(f"\n1️⃣  데이터 로드...")
        data_dir = os.path.join(os.path.dirname(__file__), '..', 'data')
        data_file = os.path.join(data_dir, f'{ticker.lower()}_top20_500days.csv')

        if not os.path.exists(data_file):
            print(f"❌ 데이터 파일 없음")
            return None

        df = pd.read_csv(data_file, index_col=0)
        df.index = pd.to_datetime(df.index, utc=True).tz_localize(None)
        print(f"  ✅ {len(df)}개 행")

        # 2. Feature Engineering
        print(f"\n2️⃣  Feature Engineering...")
        engineer = FeatureEngineer(label_threshold=1.0)
        X_clean, y_clean = engineer.prepare_ml_data(df)

        df_clean = df.loc[X_clean.index]
        feature_cols = X_clean.columns.tolist()

        # 3. Train/Test Split (80/20)
        print(f"\n3️⃣  Train/Test Split (80/20)...")
        split_idx = int(len(X_clean) * 0.8)

        X_train = X_clean.iloc[:split_idx]
        y_train = y_clean.iloc[:split_idx]
        X_test = X_clean.iloc[split_idx:]
        y_test = y_clean.iloc[split_idx:]
        df_test = df_clean.iloc[split_idx:]

        print(f"  - Train: {len(X_train)}개")
        print(f"  - Test:  {len(X_test)}개")

        # 4. 정규화
        print(f"\n4️⃣  정규화...")
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        # 5. 모델 학습
        print(f"\n5️⃣  Logistic Regression 학습...")
        model = LogisticRegression(random_state=42, max_iter=1000)
        model.fit(X_train_scaled, y_train)

        # 6. 예측
        print(f"\n6️⃣  예측 및 평가...")
        y_test_pred = model.predict(X_test_scaled)

        test_accuracy = accuracy_score(y_test, y_test_pred)
        test_f1 = f1_score(y_test, y_test_pred, zero_division=0)

        print(f"  - Test 정확도: {test_accuracy*100:.2f}%")
        print(f"  - Test F1: {test_f1:.3f}")

        # 7. 백테스팅
        print(f"\n7️⃣  백테스팅...")

        df_test_copy = df_test.copy()
        df_test_copy['prediction'] = y_test_pred
        df_test_copy['actual_return'] = df_test_copy['Close'].pct_change().shift(-1) * 100

        df_test_copy['strategy_return'] = 0.0
        df_test_copy.loc[df_test_copy['prediction'] == 1, 'strategy_return'] = df_test_copy['actual_return']

        ml_cumulative_return = (1 + df_test_copy['strategy_return'] / 100).prod() - 1
        ml_return_pct = ml_cumulative_return * 100

        bh_return_pct = (df_test['Close'].iloc[-1] / df_test['Close'].iloc[0] - 1) * 100

        excess_return = ml_return_pct - bh_return_pct

        winning_trades = df_test_copy[
            (df_test_copy['prediction'] == 1) &
            (df_test_copy['actual_return'] > 0)
        ]
        total_trades = (df_test_copy['prediction'] == 1).sum()
        win_rate = len(winning_trades) / total_trades * 100 if total_trades > 0 else 0

        print(f"  - ML Strategy: {ml_return_pct:+.2f}%")
        print(f"  - Buy & Hold:  {bh_return_pct:+.2f}%")
        print(f"  - 초과 수익:   {excess_return:+.2f}%p")
        print(f"  - 승률:        {win_rate:.1f}%")

        # 🚨 리스크 체크
        risk_level = "LOW"
        if ml_return_pct < -20:
            risk_level = "HIGH"
        elif ml_return_pct < -10:
            risk_level = "MEDIUM"

        print(f"  - ⚠️  리스크: {risk_level}")

        # 8. 모델 저장
        if test_accuracy >= 0.45 and ml_return_pct > -15:  # 엄격한 기준
            print(f"\n8️⃣  모델 저장...")
            models_dir = os.path.join(os.path.dirname(__file__), '..', 'models')
            os.makedirs(models_dir, exist_ok=True)

            model_data = {
                'model': model,
                'scaler': scaler,
                'feature_names': feature_cols,
                'train_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'performance': {
                    'train_samples': len(X_train),
                    'test_samples': len(X_test),
                    'test_accuracy': test_accuracy,
                    'test_f1': test_f1,
                    'ml_return': ml_return_pct,
                    'bh_return': bh_return_pct,
                    'excess_return': excess_return,
                    'win_rate': win_rate,
                    'num_trades': total_trades,
                    'risk_level': risk_level
                }
            }

            model_path = os.path.join(models_dir, f'{ticker.lower()}_top20_500d.pkl')
            with open(model_path, 'wb') as f:
                pickle.dump(model_data, f)

            print(f"  ✅ 저장: {model_path}")
        else:
            print(f"\n⚠️  성능 미달로 모델 저장 안 함")

        print(f"\n✅ {ticker} 완료!")

        return {
            'ticker': ticker,
            'success': True,
            'test_accuracy': test_accuracy,
            'test_f1': test_f1,
            'ml_return': ml_return_pct,
            'bh_return': bh_return_pct,
            'excess_return': excess_return,
            'win_rate': win_rate,
            'num_trades': total_trades,
            'risk_level': risk_level,
            'saved': test_accuracy >= 0.45 and ml_return_pct > -15
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
    print("🚀 TOP 20 종목 500일 모델 학습 및 백테스팅")
    print("=" * 70)

    results = []

    for ticker in TOP20_TICKERS:
        result = train_and_backtest(ticker)
        if result:
            results.append(result)

    # 최종 요약
    print("\n\n" + "=" * 70)
    print("📊 최종 요약")
    print("=" * 70)

    success_results = [r for r in results if r['success']]

    if success_results:
        print(f"\n✅ 성공 ({len(success_results)}개):")
        print(f"\n{'Ticker':<8} {'Acc':>8} {'ML Ret':>10} {'B&H':>10} {'Excess':>10} {'Win%':>8} {'리스크':<10} {'저장'}")
        print("-" * 70)

        for r in success_results:
            saved_mark = "✅" if r.get('saved', False) else "❌"
            print(f"{r['ticker']:<8} {r['test_accuracy']*100:>7.2f}% "
                  f"{r['ml_return']:>9.2f}% {r['bh_return']:>9.2f}% "
                  f"{r['excess_return']:>9.2f}%p {r['win_rate']:>7.1f}% "
                  f"{r['risk_level']:<10} {saved_mark}")

        # 통계
        saved_models = [r for r in success_results if r.get('saved', False)]
        print(f"\n📈 통계:")
        print(f"  - 총 종목: {len(TOP20_TICKERS)}개")
        print(f"  - 학습 성공: {len(success_results)}개")
        print(f"  - 모델 저장: {len(saved_models)}개")

        if saved_models:
            avg_ml = np.mean([r['ml_return'] for r in saved_models])
            avg_excess = np.mean([r['excess_return'] for r in saved_models])
            avg_win = np.mean([r['win_rate'] for r in saved_models])

            print(f"\n💡 저장된 모델 평균:")
            print(f"  - 평균 ML 수익률: {avg_ml:+.2f}%")
            print(f"  - 평균 초과 수익: {avg_excess:+.2f}%p")
            print(f"  - 평균 승률: {avg_win:.1f}%")

        # 결과 저장
        results_dir = os.path.join(os.path.dirname(__file__), '..', 'results')
        os.makedirs(results_dir, exist_ok=True)

        comparison_df = pd.DataFrame(success_results)
        comparison_file = os.path.join(results_dir, 'top20_500d_comparison.csv')
        comparison_df.to_csv(comparison_file, index=False)

        print(f"\n💾 결과 저장: {comparison_file}")

    print("\n" + "=" * 70)
    print("✨ 학습 완료!")
    print("=" * 70)


if __name__ == "__main__":
    main()
