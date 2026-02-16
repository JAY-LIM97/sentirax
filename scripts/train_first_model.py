"""
첫 번째 ML 모델 학습: Random Forest

🎯 목적:
1. Feature Engineering으로 생성한 데이터로 첫 ML 모델 학습
2. Random Forest로 매수/매도 신호 예측
3. 성능 평가 및 Feature Importance 분석
4. 결과 시각화 및 저장

📊 전체 흐름:
데이터 로드 → 정규화 → Train/Test 분할 → 학습 → 평가 → 시각화
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
import seaborn as sns
from datetime import datetime

from core.feature_engineer import FeatureEngineer
from core.ml_trainer import MLTrainer


def plot_feature_importance(feature_imp: pd.DataFrame, top_n: int = 15, save_path: str = None):
    """
    Feature Importance 시각화

    Args:
        feature_imp: Feature Importance DataFrame
        top_n: 상위 N개 특징만 표시
        save_path: 저장 경로 (None이면 표시만)
    """

    plt.figure(figsize=(10, 8))

    # 상위 N개만 선택
    top_features = feature_imp.head(top_n)

    # 막대 그래프
    colors = plt.cm.viridis(np.linspace(0, 1, len(top_features)))
    bars = plt.barh(range(len(top_features)), top_features['importance'], color=colors)

    plt.yticks(range(len(top_features)), top_features['feature'])
    plt.xlabel('Importance Score', fontsize=12)
    plt.title(f'Top {top_n} Feature Importance (Random Forest)', fontsize=14, fontweight='bold')
    plt.gca().invert_yaxis()  # 중요도 높은 것이 위로

    # 값 표시
    for i, bar in enumerate(bars):
        width = bar.get_width()
        plt.text(width, bar.get_y() + bar.get_height()/2,
                f'{width:.4f}',
                ha='left', va='center', fontsize=9)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"✅ Feature Importance 차트 저장: {save_path}")

    plt.close()


def plot_confusion_matrix(cm: np.ndarray, save_path: str = None):
    """
    Confusion Matrix 시각화

    Args:
        cm: Confusion Matrix
        save_path: 저장 경로
    """

    plt.figure(figsize=(8, 6))

    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=True,
                xticklabels=['예측 매도(0)', '예측 매수(1)'],
                yticklabels=['실제 매도(0)', '실제 매수(1)'])

    plt.title('Confusion Matrix', fontsize=14, fontweight='bold')
    plt.ylabel('Actual', fontsize=12)
    plt.xlabel('Predicted', fontsize=12)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"✅ Confusion Matrix 차트 저장: {save_path}")

    plt.close()


def main():
    print("=" * 70)
    print("🚀 Day 2: 첫 번째 ML 모델 학습 (Random Forest)")
    print("=" * 70)

    # ========================================
    # 1. 데이터 로드
    # ========================================
    print("\n📁 [Step 1] 데이터 로딩...")

    data_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'tsla_optimized_90days.csv')
    df = pd.read_csv(data_path, index_col=0)

    print(f"✅ 데이터 로드 완료: {len(df)}개 행")

    # ========================================
    # 2. Feature Engineering
    # ========================================
    print("\n🔧 [Step 2] Feature Engineering...")

    engineer = FeatureEngineer(label_threshold=1.0)
    X, y = engineer.prepare_ml_data(df)

    print(f"✅ Feature Engineering 완료")
    print(f"  - 특징 수: {X.shape[1]}개")
    print(f"  - 학습 데이터: {len(X)}개")

    # 특징 이름 저장
    feature_names = X.columns.tolist()

    # ========================================
    # 3. ML Trainer 초기화 및 데이터 분할
    # ========================================
    print("\n⚙️  [Step 3] ML Trainer 초기화...")

    trainer = MLTrainer(test_size=0.2, random_state=42)
    trainer.feature_names = feature_names

    # Train/Test 분할
    X_train, X_test, y_train, y_test = trainer.split_data(X, y)

    # ========================================
    # 4. 데이터 정규화
    # ========================================
    X_train_scaled, X_test_scaled = trainer.normalize_data(X_train, X_test)

    # ========================================
    # 5. Random Forest 학습
    # ========================================
    model = trainer.train_random_forest(
        X_train_scaled,
        y_train,
        n_estimators=100,
        max_depth=10,
        min_samples_split=5,
        min_samples_leaf=2
    )

    # ========================================
    # 6. 성능 평가
    # ========================================
    results = trainer.evaluate_model(
        X_test_scaled,
        y_test,
        X_train_scaled,
        y_train
    )

    # ========================================
    # 7. Feature Importance 분석
    # ========================================
    print("\n🔍 [Step 5] Feature Importance 분석...")

    feature_imp = trainer.get_feature_importance(feature_names)

    print(f"\n📊 상위 10개 중요 특징:")
    for i, row in feature_imp.head(10).iterrows():
        print(f"  {i+1:2d}. {row['feature']:20s}: {row['importance']:.4f}")

    # ========================================
    # 8. 결과 시각화
    # ========================================
    print("\n📈 [Step 6] 결과 시각화...")

    # results 폴더 생성
    results_dir = os.path.join(os.path.dirname(__file__), '..', 'results')
    os.makedirs(results_dir, exist_ok=True)

    # Feature Importance 차트
    fi_path = os.path.join(results_dir, 'feature_importance_rf.png')
    plot_feature_importance(feature_imp, top_n=15, save_path=fi_path)

    # Confusion Matrix 차트
    cm_path = os.path.join(results_dir, 'confusion_matrix_rf.png')
    plot_confusion_matrix(results['confusion_matrix'], save_path=cm_path)

    # ========================================
    # 9. 결과 저장
    # ========================================
    print("\n💾 [Step 7] 결과 저장...")

    # Feature Importance CSV
    fi_csv_path = os.path.join(results_dir, 'feature_importance_rf.csv')
    feature_imp.to_csv(fi_csv_path, index=False)
    print(f"✅ Feature Importance 저장: {fi_csv_path}")

    # 성능 지표 저장
    results_summary = {
        'model': 'Random Forest',
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'train_size': len(X_train),
        'test_size': len(X_test),
        'test_accuracy': results['test_accuracy'],
        'test_precision': results['test_precision'],
        'test_recall': results['test_recall'],
        'test_f1': results['test_f1'],
        'train_accuracy': results.get('train_accuracy', None)
    }

    results_df = pd.DataFrame([results_summary])
    results_csv_path = os.path.join(results_dir, 'model_performance_rf.csv')
    results_df.to_csv(results_csv_path, index=False)
    print(f"✅ 성능 지표 저장: {results_csv_path}")

    # 모델 저장
    models_dir = os.path.join(os.path.dirname(__file__), '..', 'models')
    trainer.save_model(save_dir=models_dir, model_name='random_forest')

    # ========================================
    # 10. 최종 요약
    # ========================================
    print("\n" + "=" * 70)
    print("📊 학습 완료 요약")
    print("=" * 70)

    print(f"\n🎯 모델 성능:")
    print(f"  - Test Accuracy: {results['test_accuracy']*100:.2f}%")
    print(f"  - Test Precision: {results['test_precision']*100:.2f}%")
    print(f"  - Test Recall: {results['test_recall']*100:.2f}%")
    print(f"  - Test F1-Score: {results['test_f1']:.3f}")

    print(f"\n🌲 상위 5개 중요 특징:")
    for i, row in feature_imp.head(5).iterrows():
        print(f"  {i+1}. {row['feature']:20s}: {row['importance']:.4f}")

    print(f"\n💡 인사이트:")

    # 정확도 평가
    if results['test_accuracy'] >= 0.65:
        print(f"  ✅ 우수한 성능 (정확도 65% 이상)")
    elif results['test_accuracy'] >= 0.55:
        print(f"  ⚠️  보통 성능 (정확도 55~65%)")
    else:
        print(f"  ❌ 개선 필요 (정확도 55% 미만)")

    # 과적합 체크
    if 'train_accuracy' in results:
        overfit_gap = results['train_accuracy'] - results['test_accuracy']
        if overfit_gap > 0.15:
            print(f"  ⚠️  과적합 가능성 (Train-Test 차이: {overfit_gap*100:.1f}%p)")
        else:
            print(f"  ✅ 과적합 없음 (Train-Test 차이: {overfit_gap*100:.1f}%p)")

    # Feature Importance 인사이트
    top_feature = feature_imp.iloc[0]['feature']
    top_importance = feature_imp.iloc[0]['importance']
    print(f"  🔍 가장 중요한 특징: {top_feature} ({top_importance:.4f})")

    if top_importance > 0.15:
        print(f"     → 이 특징이 15% 이상 영향 (매우 중요)")
    elif top_importance > 0.10:
        print(f"     → 이 특징이 10% 이상 영향 (중요)")

    print(f"\n📁 생성된 파일:")
    print(f"  - {fi_path}")
    print(f"  - {cm_path}")
    print(f"  - {fi_csv_path}")
    print(f"  - {results_csv_path}")

    print("\n" + "=" * 70)
    print("✨ Day 2 완료!")
    print("=" * 70)

    print(f"\n🚀 다음 단계:")
    print(f"  1. ✅ Random Forest 학습 완료")
    print(f"  2. ⏭️  백테스팅으로 실전 성능 검증")
    print(f"  3. ⏭️  하이퍼파라미터 튜닝")
    print(f"  4. ⏭️  다른 모델 (XGBoost, LSTM) 시도")


if __name__ == "__main__":
    main()
