"""
저장된 모델로 예측하기

🎯 목적:
- 학습된 모델 재사용
- 새로운 데이터로 즉시 예측
- 모델 성능 확인

📊 사용 예시:
python predict_with_saved_model.py --ticker TSLA
"""

import sys
import os
import io
import argparse
import pickle

# Windows 한글/이모지 출력 설정
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

# 프로젝트 루트를 파이썬 경로에 추가
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import numpy as np
from datetime import datetime

from core.feature_engineer import FeatureEngineer


def load_model(ticker: str):
    """
    저장된 모델 로드

    Args:
        ticker: 종목 티커 (예: 'TSLA')

    Returns:
        모델 데이터 딕셔너리
    """

    models_dir = os.path.join(os.path.dirname(__file__), '..', 'models')
    model_path = os.path.join(models_dir, f'{ticker.lower()}_logistic_regression.pkl')

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"모델 파일을 찾을 수 없습니다: {model_path}")

    with open(model_path, 'rb') as f:
        model_data = pickle.load(f)

    return model_data


def predict_signal(ticker: str, date: str = None):
    """
    특정 날짜의 매매 신호 예측

    Args:
        ticker: 종목 티커
        date: 예측 날짜 (None이면 최신 데이터)

    Returns:
        예측 신호 (0: 매도, 1: 매수)
    """

    print(f"=" * 70)
    print(f"🔮 {ticker} 매매 신호 예측")
    print(f"=" * 70)

    # 모델 로드
    print(f"\n📂 모델 로딩...")
    model_data = load_model(ticker)

    model = model_data['model']
    scaler = model_data['scaler']
    feature_names = model_data['feature_names']

    print(f"✅ 모델 로드 완료")
    print(f"  - 학습 날짜: {model_data['train_date']}")
    print(f"  - Test 정확도: {model_data['performance']['test_accuracy']*100:.2f}%")
    print(f"  - 백테스팅 수익률: {model_data['performance']['ml_return']:+.2f}%")

    # 데이터 로드
    print(f"\n📊 데이터 로딩...")
    data_dir = os.path.join(os.path.dirname(__file__), '..', 'data')
    data_file = os.path.join(data_dir, f'{ticker.lower()}_longterm_180days.csv')

    if not os.path.exists(data_file):
        raise FileNotFoundError(f"데이터 파일을 찾을 수 없습니다: {data_file}")

    df = pd.read_csv(data_file, index_col=0)
    print(f"✅ 데이터 로드 완료: {len(df)}개 행")

    # Feature Engineering
    print(f"\n🔧 Feature Engineering...")
    engineer = FeatureEngineer(label_threshold=1.0)
    df_features = engineer.create_features(df)

    X = df_features[feature_names]
    valid_idx = X.notna().all(axis=1)
    X_clean = X[valid_idx]
    df_clean = df_features[valid_idx]

    print(f"✅ Feature 생성 완료")

    # 날짜 선택
    if date is None:
        # 최신 날짜
        date = df_clean.index[-1]
        print(f"  - 최신 날짜 사용: {date}")
    else:
        if date not in df_clean.index:
            raise ValueError(f"날짜를 찾을 수 없습니다: {date}")
        print(f"  - 선택된 날짜: {date}")

    # 해당 날짜 데이터
    idx = df_clean.index.get_loc(date)
    X_date = X_clean.iloc[idx:idx+1]

    # 정규화
    X_scaled = scaler.transform(X_date)

    # 예측
    prediction = model.predict(X_scaled)[0]
    probability = model.predict_proba(X_scaled)[0]

    # 결과
    print(f"\n{'='*70}")
    print(f"🎯 예측 결과")
    print(f"{'='*70}")

    print(f"\n📅 날짜: {date}")
    print(f"💵 종가: ${df_clean['Close'].iloc[idx]:.2f}")

    signal_text = "매수 (BUY)" if prediction == 1 else "매도 (SELL)"
    signal_emoji = "🟢" if prediction == 1 else "🔴"

    print(f"\n{signal_emoji} 신호: {signal_text}")
    print(f"  - 매도 확률: {probability[0]*100:.1f}%")
    print(f"  - 매수 확률: {probability[1]*100:.1f}%")

    # 주요 특징값
    print(f"\n📊 주요 특징값:")
    important_features = ['oil', 'volatility_5d', 'vix_lag1', 'volume_ratio', 'ma_5']
    for feat in important_features:
        if feat in df_clean.columns:
            value = df_clean[feat].iloc[idx]
            print(f"  - {feat:20s}: {value:10.2f}")

    return {
        'ticker': ticker,
        'date': date,
        'close': df_clean['Close'].iloc[idx],
        'signal': int(prediction),
        'signal_text': signal_text,
        'probability': probability.tolist(),
        'features': X_date.iloc[0].to_dict()
    }


def show_available_models():
    """저장된 모델 목록 표시"""

    print("=" * 70)
    print("📂 사용 가능한 모델")
    print("=" * 70)

    models_dir = os.path.join(os.path.dirname(__file__), '..', 'models')

    if not os.path.exists(models_dir):
        print("\n⚠️  models 폴더가 없습니다!")
        return

    model_files = [f for f in os.listdir(models_dir) if f.endswith('.pkl')]

    if not model_files:
        print("\n⚠️  저장된 모델이 없습니다!")
        return

    print(f"\n✅ 총 {len(model_files)}개 모델:")

    for filename in sorted(model_files):
        ticker = filename.replace('_logistic_regression.pkl', '').upper()
        filepath = os.path.join(models_dir, filename)

        # 모델 로드해서 정보 표시
        try:
            with open(filepath, 'rb') as f:
                model_data = pickle.load(f)

            print(f"\n  📊 {ticker}")
            print(f"     - 학습 날짜: {model_data.get('train_date', 'N/A')}")
            print(f"     - Test 정확도: {model_data['performance']['test_accuracy']*100:.2f}%")
            print(f"     - 백테스팅 수익률: {model_data['performance']['ml_return']:+.2f}%")
            print(f"     - 초과 수익: {model_data['performance']['excess_return']:+.2f}%p")

        except Exception as e:
            print(f"\n  ⚠️  {ticker}: 로드 실패 ({e})")


def main():
    parser = argparse.ArgumentParser(description='저장된 모델로 매매 신호 예측')
    parser.add_argument('--ticker', type=str, help='종목 티커 (예: TSLA)')
    parser.add_argument('--date', type=str, help='예측 날짜 (YYYY-MM-DD)')
    parser.add_argument('--list', action='store_true', help='사용 가능한 모델 목록 표시')

    args = parser.parse_args()

    if args.list:
        show_available_models()
        return

    if not args.ticker:
        print("❌ --ticker 인자가 필요합니다!")
        print("예시: python predict_with_saved_model.py --ticker TSLA")
        print("또는: python predict_with_saved_model.py --list")
        return

    try:
        result = predict_signal(args.ticker.upper(), args.date)

        print(f"\n{'='*70}")
        print(f"✨ 예측 완료!")
        print(f"{'='*70}")

    except Exception as e:
        print(f"\n❌ 오류 발생: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
