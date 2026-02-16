"""
Step 2: 특징 선택 후 재학습

🎯 목적:
상위 10개 중요 특징만 사용하여 과적합 방지

📊 이전 Feature Importance 상위 10개:
1. oil (11.55%)
2. volatility_5d (10.21%)
3. vix_lag1 (10.02%)
4. volume_ratio (8.41%)
5. ma_5 (8.24%)
6. treasury_10y (5.38%)
7. vix (5.33%)
8. ma_20 (4.78%)
9. return_20d (4.66%)
10. volume_ma_20 (4.58%)
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

from core.feature_engineer import FeatureEngineer
from core.ml_trainer import MLTrainer


def main():
    print("=" * 70)
    print("📊 Step 2: 특징 선택 후 재학습")
    print("=" * 70)

    # ========================================
    # 1. 데이터 준비
    # ========================================
    print("\n📁 데이터 로딩...")

    data_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'tsla_optimized_90days.csv')
    df = pd.read_csv(data_path, index_col=0)

    engineer = FeatureEngineer(label_threshold=1.0)
    X_full, y = engineer.prepare_ml_data(df)

    print(f"✅ 전체 특징: {X_full.shape[1]}개")

    # ========================================
    # 2. 상위 10개 특징 선택
    # ========================================
    print("\n🔍 상위 10개 특징 선택...")

    # 이전 결과에서 확인된 상위 10개
    top_features = [
        'oil',
        'volatility_5d',
        'vix_lag1',
        'volume_ratio',
        'ma_5',
        'treasury_10y',
        'vix',
        'ma_20',
        'return_20d',
        'volume_ma_20'
    ]

    # 실제로 존재하는 특징만 선택
    selected_features = [f for f in top_features if f in X_full.columns]
    X_selected = X_full[selected_features]

    print(f"✅ 선택된 특징 ({len(selected_features)}개):")
    for i, feat in enumerate(selected_features, 1):
        print(f"  {i:2d}. {feat}")

    # ========================================
    # 3. 최적 하이퍼파라미터로 학습 (전체 특징)
    # ========================================
    print("\n\n" + "="*70)
    print("🌲 [비교 A] 전체 특징 (26개) + 최적 하이퍼파라미터")
    print("="*70)

    trainer_full = MLTrainer(test_size=0.2, random_state=42)
    trainer_full.feature_names = X_full.columns.tolist()

    X_train_full, X_test_full, y_train, y_test = trainer_full.split_data(X_full, y)
    X_train_full_scaled, X_test_full_scaled = trainer_full.normalize_data(X_train_full, X_test_full)

    # Step 1에서 찾은 최적 파라미터
    trainer_full.train_random_forest(
        X_train_full_scaled,
        y_train,
        n_estimators=100,
        max_depth=5,
        min_samples_split=10,
        min_samples_leaf=10
    )

    results_full = trainer_full.evaluate_model(
        X_test_full_scaled,
        y_test,
        X_train_full_scaled,
        y_train
    )

    # ========================================
    # 4. 최적 하이퍼파라미터로 학습 (선택된 특징)
    # ========================================
    print("\n\n" + "="*70)
    print(f"🎯 [비교 B] 선택된 특징 ({len(selected_features)}개) + 최적 하이퍼파라미터")
    print("="*70)

    trainer_selected = MLTrainer(test_size=0.2, random_state=42)
    trainer_selected.feature_names = X_selected.columns.tolist()

    X_train_sel, X_test_sel, y_train, y_test = trainer_selected.split_data(X_selected, y)
    X_train_sel_scaled, X_test_sel_scaled = trainer_selected.normalize_data(X_train_sel, X_test_sel)

    trainer_selected.train_random_forest(
        X_train_sel_scaled,
        y_train,
        n_estimators=100,
        max_depth=5,
        min_samples_split=10,
        min_samples_leaf=10
    )

    results_selected = trainer_selected.evaluate_model(
        X_test_sel_scaled,
        y_test,
        X_train_sel_scaled,
        y_train
    )

    # ========================================
    # 5. 결과 비교
    # ========================================
    print("\n\n" + "="*70)
    print("📊 최종 비교")
    print("="*70)

    comparison = pd.DataFrame({
        '지표': ['특징 수', 'Train Accuracy', 'Test Accuracy', 'Test Precision',
                'Test Recall', 'Test F1-Score', 'Overfit Gap'],
        '전체 특징 (26개)': [
            26,
            f"{results_full['train_accuracy']*100:.2f}%",
            f"{results_full['test_accuracy']*100:.2f}%",
            f"{results_full['test_precision']*100:.2f}%",
            f"{results_full['test_recall']*100:.2f}%",
            f"{results_full['test_f1']:.3f}",
            f"{(results_full['train_accuracy'] - results_full['test_accuracy'])*100:.1f}%p"
        ],
        f'선택된 특징 ({len(selected_features)}개)': [
            len(selected_features),
            f"{results_selected['train_accuracy']*100:.2f}%",
            f"{results_selected['test_accuracy']*100:.2f}%",
            f"{results_selected['test_precision']*100:.2f}%",
            f"{results_selected['test_recall']*100:.2f}%",
            f"{results_selected['test_f1']:.3f}",
            f"{(results_selected['train_accuracy'] - results_selected['test_accuracy'])*100:.1f}%p"
        ]
    })

    print("\n")
    print(comparison.to_string(index=False))

    # ========================================
    # 6. 인사이트
    # ========================================
    print("\n\n💡 인사이트:")

    # Test Accuracy 비교
    acc_diff = results_selected['test_accuracy'] - results_full['test_accuracy']
    if acc_diff > 0:
        print(f"  ✅ 특징 선택으로 Test Accuracy {acc_diff*100:.1f}%p 향상!")
    elif acc_diff < -0.05:
        print(f"  ⚠️  특징 선택으로 Test Accuracy {abs(acc_diff)*100:.1f}%p 하락")
    else:
        print(f"  ➖ Test Accuracy 비슷 (차이 {abs(acc_diff)*100:.1f}%p)")

    # Overfit Gap 비교
    gap_full = results_full['train_accuracy'] - results_full['test_accuracy']
    gap_selected = results_selected['train_accuracy'] - results_selected['test_accuracy']
    gap_diff = gap_full - gap_selected

    if gap_diff > 0.05:
        print(f"  ✅ 과적합 {gap_diff*100:.1f}%p 감소! (더 안정적)")
    elif gap_diff < -0.05:
        print(f"  ⚠️  과적합 {abs(gap_diff)*100:.1f}%p 증가")
    else:
        print(f"  ➖ 과적합 정도 비슷")

    # F1-Score 비교
    f1_diff = results_selected['test_f1'] - results_full['test_f1']
    if f1_diff > 0:
        print(f"  ✅ F1-Score {f1_diff:.3f} 향상!")
    else:
        print(f"  ➖ F1-Score {abs(f1_diff):.3f} 하락")

    # ========================================
    # 7. 결과 저장
    # ========================================
    print(f"\n💾 결과 저장...")

    results_dir = os.path.join(os.path.dirname(__file__), '..', 'results')

    # 비교 결과 저장
    comparison_path = os.path.join(results_dir, 'feature_selection_comparison.csv')
    comparison.to_csv(comparison_path, index=False)
    print(f"✅ 비교 결과 저장: {comparison_path}")

    # 최고 성능 모델 저장
    if results_selected['test_f1'] >= results_full['test_f1']:
        print(f"\n🏆 선택된 특징 모델이 더 우수!")
        models_dir = os.path.join(os.path.dirname(__file__), '..', 'models')
        trainer_selected.save_model(save_dir=models_dir, model_name='rf_selected_features')
    else:
        print(f"\n🏆 전체 특징 모델이 더 우수")

    print("\n" + "="*70)
    print("✨ Step 2 완료!")
    print("="*70)


if __name__ == "__main__":
    main()
