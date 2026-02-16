"""
Step 3: 로지스틱 회귀 학습

🎯 목적:
가장 단순한 모델로 과적합 근본 해결

📚 로지스틱 회귀란?
- 가장 기본적인 분류 모델
- 선형 결합 → 시그모이드 함수 → 0 or 1
- 과적합 위험 낮음 (파라미터 적음)
- 데이터 적을 때 효과적

💡 왜 시도하나?
- Random Forest: 복잡함 (트리 100개)
- 로지스틱 회귀: 단순함 (선형)
- 데이터 77개 → 단순한 모델이 나을 수도
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
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

from core.feature_engineer import FeatureEngineer
from core.ml_trainer import MLTrainer


def main():
    print("=" * 70)
    print("🤖 Step 3: 로지스틱 회귀 학습")
    print("=" * 70)

    # ========================================
    # 1. 데이터 준비
    # ========================================
    print("\n📁 데이터 로딩...")

    data_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'tsla_optimized_90days.csv')
    df = pd.read_csv(data_path, index_col=0)

    engineer = FeatureEngineer(label_threshold=1.0)
    X, y = engineer.prepare_ml_data(df)

    print(f"✅ 특징: {X.shape[1]}개, 데이터: {len(X)}개")

    # ========================================
    # 2. Train/Test 분할 및 정규화
    # ========================================
    trainer = MLTrainer(test_size=0.2, random_state=42)

    X_train, X_test, y_train, y_test = trainer.split_data(X, y)
    X_train_scaled, X_test_scaled = trainer.normalize_data(X_train, X_test)

    # ========================================
    # 3. 로지스틱 회귀 학습
    # ========================================
    print("\n📊 [모델 A] 로지스틱 회귀...")

    # 💡 C: 정규화 강도 (작을수록 강함)
    # 💡 class_weight='balanced': 클래스 불균형 자동 조정
    lr_model = LogisticRegression(
        C=1.0,                    # 기본값
        max_iter=1000,            # 충분한 반복
        random_state=42,
        class_weight='balanced'   # 클래스 불균형 조정
    )

    print(f"  하이퍼파라미터:")
    print(f"    - C (정규화): 1.0")
    print(f"    - Class Weight: balanced")
    print(f"\n  학습 중...")

    lr_model.fit(X_train_scaled, y_train)
    print(f"✅ 학습 완료")

    # 예측
    y_pred_train = lr_model.predict(X_train_scaled)
    y_pred_test = lr_model.predict(X_test_scaled)

    # 평가
    lr_results = {
        'train_accuracy': accuracy_score(y_train, y_pred_train),
        'test_accuracy': accuracy_score(y_test, y_pred_test),
        'test_precision': precision_score(y_test, y_pred_test, zero_division=0),
        'test_recall': recall_score(y_test, y_pred_test, zero_division=0),
        'test_f1': f1_score(y_test, y_pred_test, zero_division=0),
        'confusion_matrix': confusion_matrix(y_test, y_pred_test)
    }

    print(f"\n📈 성능:")
    print(f"  - Train Accuracy: {lr_results['train_accuracy']*100:.2f}%")
    print(f"  - Test Accuracy: {lr_results['test_accuracy']*100:.2f}%")
    print(f"  - Test Precision: {lr_results['test_precision']*100:.2f}%")
    print(f"  - Test Recall: {lr_results['test_recall']*100:.2f}%")
    print(f"  - Test F1-Score: {lr_results['test_f1']:.3f}")
    print(f"  - Overfit Gap: {(lr_results['train_accuracy'] - lr_results['test_accuracy'])*100:.1f}%p")

    cm = lr_results['confusion_matrix']
    print(f"\n  📊 Confusion Matrix:")
    print(f"                예측 매도(0)  예측 매수(1)")
    print(f"  실제 매도(0)      {cm[0,0]:3d}         {cm[0,1]:3d}")
    print(f"  실제 매수(1)      {cm[1,0]:3d}         {cm[1,1]:3d}")

    # ========================================
    # 4. Random Forest (최적) 재학습 for 비교
    # ========================================
    print("\n\n📊 [모델 B] Random Forest (최적 파라미터)...")

    trainer_rf = MLTrainer(test_size=0.2, random_state=42)
    X_train, X_test, y_train, y_test = trainer_rf.split_data(X, y)
    X_train_scaled, X_test_scaled = trainer_rf.normalize_data(X_train, X_test)

    trainer_rf.train_random_forest(
        X_train_scaled,
        y_train,
        n_estimators=100,
        max_depth=5,
        min_samples_split=10,
        min_samples_leaf=10
    )

    rf_results = trainer_rf.evaluate_model(
        X_test_scaled,
        y_test,
        X_train_scaled,
        y_train
    )

    # ========================================
    # 5. 결과 비교
    # ========================================
    print("\n\n" + "="*70)
    print("📊 최종 비교: 로지스틱 회귀 vs Random Forest")
    print("="*70)

    comparison = pd.DataFrame({
        '지표': ['Train Accuracy', 'Test Accuracy', 'Test Precision',
                'Test Recall', 'Test F1-Score', 'Overfit Gap'],
        '로지스틱 회귀': [
            f"{lr_results['train_accuracy']*100:.2f}%",
            f"{lr_results['test_accuracy']*100:.2f}%",
            f"{lr_results['test_precision']*100:.2f}%",
            f"{lr_results['test_recall']*100:.2f}%",
            f"{lr_results['test_f1']:.3f}",
            f"{(lr_results['train_accuracy'] - lr_results['test_accuracy'])*100:.1f}%p"
        ],
        'Random Forest': [
            f"{rf_results['train_accuracy']*100:.2f}%",
            f"{rf_results['test_accuracy']*100:.2f}%",
            f"{rf_results['test_precision']*100:.2f}%",
            f"{rf_results['test_recall']*100:.2f}%",
            f"{rf_results['test_f1']:.3f}",
            f"{(rf_results['train_accuracy'] - rf_results['test_accuracy'])*100:.1f}%p"
        ]
    })

    print("\n")
    print(comparison.to_string(index=False))

    # ========================================
    # 6. 인사이트
    # ========================================
    print("\n\n💡 인사이트:")

    # Test Accuracy
    acc_diff = lr_results['test_accuracy'] - rf_results['test_accuracy']
    if acc_diff > 0.05:
        print(f"  ✅ 로지스틱 회귀가 Test Accuracy {acc_diff*100:.1f}%p 우수!")
    elif acc_diff < -0.05:
        print(f"  ⚠️  Random Forest가 Test Accuracy {abs(acc_diff)*100:.1f}%p 우수")
    else:
        print(f"  ➖ Test Accuracy 비슷 (차이 {abs(acc_diff)*100:.1f}%p)")

    # Overfit Gap
    gap_lr = lr_results['train_accuracy'] - lr_results['test_accuracy']
    gap_rf = rf_results['train_accuracy'] - rf_results['test_accuracy']
    gap_diff = gap_rf - gap_lr

    if gap_diff > 0.05:
        print(f"  ✅ 로지스틱 회귀가 과적합 {gap_diff*100:.1f}%p 적음! (더 안정적)")
    elif gap_diff < -0.05:
        print(f"  ⚠️  Random Forest가 과적합 {abs(gap_diff)*100:.1f}%p 적음")
    else:
        print(f"  ➖ 과적합 정도 비슷")

    # F1-Score
    f1_diff = lr_results['test_f1'] - rf_results['test_f1']
    if f1_diff > 0:
        print(f"  ✅ 로지스틱 회귀가 F1-Score {f1_diff:.3f} 우수!")
    else:
        print(f"  ➖ Random Forest가 F1-Score {abs(f1_diff):.3f} 우수")

    # 최종 추천
    print(f"\n🏆 최종 추천:")
    if lr_results['test_f1'] > rf_results['test_f1'] and gap_lr < gap_rf:
        print(f"  로지스틱 회귀 사용 권장 (성능 우수 + 과적합 적음)")
        best_model = 'logistic_regression'
        best_results = lr_results
    elif rf_results['test_f1'] > lr_results['test_f1']:
        print(f"  Random Forest 사용 권장 (F1-Score 우수)")
        best_model = 'random_forest'
        best_results = rf_results
    else:
        print(f"  두 모델 성능 비슷, Random Forest 선택 (더 복잡한 패턴 학습 가능)")
        best_model = 'random_forest'
        best_results = rf_results

    # ========================================
    # 7. 결과 저장
    # ========================================
    print(f"\n💾 결과 저장...")

    results_dir = os.path.join(os.path.dirname(__file__), '..', 'results')

    # 비교 결과 저장
    comparison_path = os.path.join(results_dir, 'lr_vs_rf_comparison.csv')
    comparison.to_csv(comparison_path, index=False)
    print(f"✅ 비교 결과 저장: {comparison_path}")

    # 성능 요약
    summary = pd.DataFrame({
        'model': ['Logistic Regression', 'Random Forest', 'Best'],
        'test_accuracy': [
            lr_results['test_accuracy'],
            rf_results['test_accuracy'],
            best_results['test_accuracy']
        ],
        'test_f1': [
            lr_results['test_f1'],
            rf_results['test_f1'],
            best_results['test_f1']
        ],
        'overfit_gap': [
            gap_lr,
            gap_rf,
            gap_lr if best_model == 'logistic_regression' else gap_rf
        ],
        'recommended': [
            'Yes' if best_model == 'logistic_regression' else 'No',
            'Yes' if best_model == 'random_forest' else 'No',
            'Yes'
        ]
    })

    summary_path = os.path.join(results_dir, 'model_comparison_summary.csv')
    summary.to_csv(summary_path, index=False)
    print(f"✅ 모델 비교 요약 저장: {summary_path}")

    print("\n" + "="*70)
    print("✨ Step 3 완료!")
    print("="*70)


if __name__ == "__main__":
    main()
