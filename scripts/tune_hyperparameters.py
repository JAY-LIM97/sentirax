"""
하이퍼파라미터 튜닝 스크립트

🎯 목적:
과적합 문제 해결을 위한 최적 하이퍼파라미터 찾기

📊 시도할 조합:
- max_depth: 3, 5, 7 (현재 10에서 줄이기)
- min_samples_split: 10, 15, 20 (현재 5에서 늘리기)
- n_estimators: 50, 100 (트리 개수)
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
from itertools import product

from core.feature_engineer import FeatureEngineer
from core.ml_trainer import MLTrainer


def main():
    print("=" * 70)
    print("🔧 Step 1: 하이퍼파라미터 튜닝")
    print("=" * 70)

    # ========================================
    # 1. 데이터 준비
    # ========================================
    print("\n📁 데이터 로딩 및 Feature Engineering...")

    data_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'tsla_optimized_90days.csv')
    df = pd.read_csv(data_path, index_col=0)

    engineer = FeatureEngineer(label_threshold=1.0)
    X, y = engineer.prepare_ml_data(df)

    feature_names = X.columns.tolist()

    # ========================================
    # 2. 하이퍼파라미터 그리드 정의
    # ========================================
    print("\n⚙️  하이퍼파라미터 그리드 설정...")

    param_grid = {
        'n_estimators': [50, 100],
        'max_depth': [3, 5, 7],
        'min_samples_split': [10, 15, 20],
        'min_samples_leaf': [5, 7, 10]
    }

    print(f"  - n_estimators: {param_grid['n_estimators']}")
    print(f"  - max_depth: {param_grid['max_depth']}")
    print(f"  - min_samples_split: {param_grid['min_samples_split']}")
    print(f"  - min_samples_leaf: {param_grid['min_samples_leaf']}")

    # 전체 조합 수
    total_combinations = (len(param_grid['n_estimators']) *
                         len(param_grid['max_depth']) *
                         len(param_grid['min_samples_split']) *
                         len(param_grid['min_samples_leaf']))

    print(f"\n  총 {total_combinations}개 조합 테스트")

    # ========================================
    # 3. 그리드 서치
    # ========================================
    print("\n🔍 그리드 서치 시작...\n")

    results = []
    best_score = 0
    best_params = None

    for i, (n_est, max_d, min_split, min_leaf) in enumerate(
        product(
            param_grid['n_estimators'],
            param_grid['max_depth'],
            param_grid['min_samples_split'],
            param_grid['min_samples_leaf']
        ), 1
    ):
        print(f"[{i}/{total_combinations}] 테스트 중: n_est={n_est}, max_depth={max_d}, "
              f"min_split={min_split}, min_leaf={min_leaf}")

        # Trainer 초기화
        trainer = MLTrainer(test_size=0.2, random_state=42)
        trainer.feature_names = feature_names

        # Train/Test 분할
        X_train, X_test, y_train, y_test = trainer.split_data(X, y)

        # 정규화
        X_train_scaled, X_test_scaled = trainer.normalize_data(X_train, X_test)

        # 학습
        try:
            trainer.train_random_forest(
                X_train_scaled,
                y_train,
                n_estimators=n_est,
                max_depth=max_d,
                min_samples_split=min_split,
                min_samples_leaf=min_leaf
            )

            # 평가
            eval_results = trainer.evaluate_model(
                X_test_scaled,
                y_test,
                X_train_scaled,
                y_train
            )

            # 과적합 차이
            overfit_gap = eval_results['train_accuracy'] - eval_results['test_accuracy']

            # 결과 저장
            result = {
                'n_estimators': n_est,
                'max_depth': max_d,
                'min_samples_split': min_split,
                'min_samples_leaf': min_leaf,
                'train_accuracy': eval_results['train_accuracy'],
                'test_accuracy': eval_results['test_accuracy'],
                'test_precision': eval_results['test_precision'],
                'test_recall': eval_results['test_recall'],
                'test_f1': eval_results['test_f1'],
                'overfit_gap': overfit_gap
            }
            results.append(result)

            print(f"  → Test Acc: {eval_results['test_accuracy']*100:.1f}%, "
                  f"Overfit Gap: {overfit_gap*100:.1f}%p, "
                  f"F1: {eval_results['test_f1']:.3f}")

            # 최고 성능 업데이트 (F1-Score 기준, 과적합 고려)
            # F1-Score를 주요 지표로, 과적합 차이가 30% 미만인 경우만
            if eval_results['test_f1'] > best_score and overfit_gap < 0.3:
                best_score = eval_results['test_f1']
                best_params = result.copy()

        except Exception as e:
            print(f"  → 오류 발생: {e}")

        print()

    # ========================================
    # 4. 결과 분석
    # ========================================
    print("=" * 70)
    print("📊 그리드 서치 결과")
    print("=" * 70)

    results_df = pd.DataFrame(results)

    # 결과 정렬 (Test F1-Score 기준)
    results_df = results_df.sort_values('test_f1', ascending=False)

    print(f"\n🏆 상위 5개 조합 (F1-Score 기준):\n")
    print(results_df.head(5).to_string(index=False))

    # 과적합이 적은 조합
    print(f"\n✅ 과적합이 적은 상위 5개 (Overfit Gap < 30%):\n")
    low_overfit = results_df[results_df['overfit_gap'] < 0.3].head(5)
    if len(low_overfit) > 0:
        print(low_overfit.to_string(index=False))
    else:
        print("  과적합이 적은 조합이 없습니다 (모두 30% 이상)")

    # 최적 파라미터
    if best_params:
        print(f"\n🎯 최적 하이퍼파라미터 (F1-Score 최대 + Overfit < 30%):")
        print(f"  - n_estimators: {best_params['n_estimators']}")
        print(f"  - max_depth: {best_params['max_depth']}")
        print(f"  - min_samples_split: {best_params['min_samples_split']}")
        print(f"  - min_samples_leaf: {best_params['min_samples_leaf']}")
        print(f"\n  성능:")
        print(f"  - Test Accuracy: {best_params['test_accuracy']*100:.2f}%")
        print(f"  - Test F1-Score: {best_params['test_f1']:.3f}")
        print(f"  - Overfit Gap: {best_params['overfit_gap']*100:.1f}%p")
    else:
        print(f"\n⚠️  과적합이 적은(30% 미만) 조합을 찾지 못했습니다")
        print(f"  → 과적합이 가장 적은 조합 선택")
        best_params = results_df.sort_values('overfit_gap').iloc[0].to_dict()
        print(f"\n  선택된 파라미터:")
        print(f"  - n_estimators: {int(best_params['n_estimators'])}")
        print(f"  - max_depth: {int(best_params['max_depth'])}")
        print(f"  - min_samples_split: {int(best_params['min_samples_split'])}")
        print(f"  - min_samples_leaf: {int(best_params['min_samples_leaf'])}")

    # ========================================
    # 5. 결과 저장
    # ========================================
    print(f"\n💾 결과 저장...")

    results_dir = os.path.join(os.path.dirname(__file__), '..', 'results')
    results_path = os.path.join(results_dir, 'hyperparameter_tuning_results.csv')
    results_df.to_csv(results_path, index=False)
    print(f"✅ 저장 완료: {results_path}")

    # 최적 파라미터 저장
    best_params_path = os.path.join(results_dir, 'best_hyperparameters.csv')
    pd.DataFrame([best_params]).to_csv(best_params_path, index=False)
    print(f"✅ 최적 파라미터 저장: {best_params_path}")

    # ========================================
    # 6. 인사이트
    # ========================================
    print(f"\n💡 인사이트:")

    # max_depth 분석
    avg_by_depth = results_df.groupby('max_depth').agg({
        'test_accuracy': 'mean',
        'overfit_gap': 'mean'
    })
    print(f"\n  📊 max_depth별 평균:")
    for depth, row in avg_by_depth.iterrows():
        print(f"    depth={depth}: Test Acc={row['test_accuracy']*100:.1f}%, "
              f"Overfit={row['overfit_gap']*100:.1f}%p")

    print("\n" + "=" * 70)
    print("✨ Step 1 완료!")
    print("=" * 70)


if __name__ == "__main__":
    main()
