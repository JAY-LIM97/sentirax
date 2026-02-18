"""
Scalping Model - 1분봉 기반 스캘핑 ML 모델

- 급등주 TOP 20 대상 1분봉 데이터 수집 (최근 5일)
- 손절 3% / 익절 5% 기준 레이블링
- ML 모델로 진입 타이밍 예측
- 백테스팅으로 검증

yfinance 최소 단위가 1분이므로 1분봉 사용 (10초 데이터 미지원)
"""

import sys
import os
import io
import platform

if platform.system() == 'Windows':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import numpy as np
import pickle
import yfinance as yf
from datetime import datetime, timedelta
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score, classification_report

# 공유 Feature Engineering (core/scalping_signals.py)
from core.scalping_signals import create_scalping_features


def collect_intraday_data(ticker: str, period: str = '5d') -> pd.DataFrame:
    """1분봉 인트라데이 데이터 수집"""

    try:
        stock = yf.Ticker(ticker)
        df = stock.history(period=period, interval='1m')

        if df.empty or len(df) < 100:
            return None

        return df
    except Exception as e:
        print(f"  Error collecting {ticker}: {e}")
        return None


def label_scalping(df: pd.DataFrame, take_profit: float = 5.0, stop_loss: float = 3.0,
                   max_hold_minutes: int = 60) -> pd.Series:
    """
    스캘핑 레이블링: 진입 후 익절(5%) vs 손절(3%)

    1 = 익절 먼저 도달 (매수 신호)
    0 = 손절 먼저 도달 또는 타임아웃 (매수 금지)
    """

    labels = pd.Series(index=df.index, dtype=float)
    close = df['Close'].values

    for i in range(len(close) - max_hold_minutes):
        entry_price = close[i]
        tp_price = entry_price * (1 + take_profit / 100)
        sl_price = entry_price * (1 - stop_loss / 100)

        label = 0  # 기본: 매수 금지

        for j in range(i + 1, min(i + max_hold_minutes + 1, len(close))):
            if close[j] >= tp_price:
                label = 1  # 익절 도달
                break
            elif close[j] <= sl_price:
                label = 0  # 손절 도달
                break

        labels.iloc[i] = label

    return labels


def train_scalping_model(ticker: str, df_raw: pd.DataFrame) -> dict:
    """단일 종목 스캘핑 모델 학습"""

    print(f"\n{'='*70}")
    print(f"  {ticker} Scalping Model Training")
    print(f"{'='*70}")

    # 1. Feature Engineering
    print(f"\n  1. Feature Engineering...")
    features = create_scalping_features(df_raw)

    # 2. Labeling - 변동성에 맞게 TP/SL 자동 조정
    # 일평균 변동폭 계산
    avg_range = (df_raw['High'] / df_raw['Low'] - 1).mean() * 100
    # TP = 변동폭의 1.5배, SL = 변동폭의 1배 (최소 TP=1.5%, SL=1.0%)
    tp = max(round(avg_range * 1.5, 1), 1.5)
    sl = max(round(avg_range * 1.0, 1), 1.0)
    # 사용자 요청 상한: TP=5%, SL=3%
    tp = min(tp, 5.0)
    sl = min(sl, 3.0)

    print(f"  2. Labeling (TP={tp}%, SL={sl}%, MaxHold=60min)...")
    labels = label_scalping(df_raw, take_profit=tp, stop_loss=sl, max_hold_minutes=60)

    # Feature 선택
    feature_cols = [c for c in features.columns if c not in ['close', 'volume',
                    'ma_5m', 'ma_10m', 'ma_20m', 'ma_60m']]

    # 결합 및 정리
    combined = features[feature_cols].copy()
    combined['label'] = labels

    # NaN 제거
    combined = combined.dropna()

    if len(combined) < 100:
        print(f"  Insufficient data: {len(combined)} rows")
        return None

    X = combined[feature_cols]
    y = combined['label'].astype(int)

    buy_ratio = y.mean() * 100
    print(f"  Data: {len(X)} rows, Buy signals: {buy_ratio:.1f}%")

    # 3. Train/Test Split (시간순 80/20)
    print(f"  3. Train/Test Split...")
    split_idx = int(len(X) * 0.8)
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

    print(f"     Train: {len(X_train)}, Test: {len(X_test)}")

    # 클래스 최소 2개 필요
    if len(y_train.unique()) < 2:
        print(f"  Only 1 class in training data. Skipping.")
        return None
    if len(y_test.unique()) < 2:
        print(f"  Only 1 class in test data. Skipping.")
        return None

    # 최소 매수 신호 10개 이상
    if y_train.sum() < 10:
        print(f"  Too few buy signals in train ({y_train.sum()}). Skipping.")
        return None

    # 4. 정규화
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # 5. GradientBoosting 학습 (스캘핑에 더 적합)
    print(f"  4. GradientBoosting Training...")
    model = GradientBoostingClassifier(
        n_estimators=150,
        max_depth=4,
        learning_rate=0.1,
        subsample=0.8,
        random_state=42
    )
    model.fit(X_train_scaled, y_train)

    # 6. 평가
    print(f"  5. Evaluation...")
    y_pred = model.predict(X_test_scaled)
    y_prob = model.predict_proba(X_test_scaled)

    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, zero_division=0)

    print(f"     Accuracy: {accuracy*100:.2f}%")
    print(f"     F1 Score: {f1:.3f}")

    # 7. 백테스팅 (테스트 구간)
    print(f"  6. Backtesting...")

    # 매수 신호가 1인 경우만 진입
    test_trades = []
    close_prices = df_raw['Close'].reindex(X_test.index).values

    for i in range(len(y_pred)):
        if y_pred[i] == 1 and i < len(close_prices):
            entry_price = close_prices[i]

            # 이후 60분 내 결과 확인
            trade_result = 0
            for j in range(i + 1, min(i + 61, len(close_prices))):
                pnl = (close_prices[j] / entry_price - 1) * 100
                if pnl >= tp:
                    trade_result = tp
                    break
                elif pnl <= -sl:
                    trade_result = -sl
                    break

            if trade_result == 0 and i + 60 < len(close_prices):
                trade_result = (close_prices[min(i + 60, len(close_prices) - 1)] / entry_price - 1) * 100

            test_trades.append(trade_result)

    num_trades = len(test_trades)
    if num_trades > 0:
        wins = sum(1 for t in test_trades if t > 0)
        win_rate = wins / num_trades * 100
        avg_return = np.mean(test_trades)
        total_return = sum(test_trades)

        print(f"     Trades: {num_trades}")
        print(f"     Win Rate: {win_rate:.1f}%")
        print(f"     Avg Return/Trade: {avg_return:+.2f}%")
        print(f"     Total Return: {total_return:+.2f}%")
    else:
        win_rate = 0
        avg_return = 0
        total_return = 0
        print(f"     No trades generated")

    # 8. 모델 저장 조건: 정확도 >= 50% AND 승률 >= 40%
    saved = False
    if accuracy >= 0.50 and win_rate >= 40:
        models_dir = os.path.join(os.path.dirname(__file__), '..', 'models')
        os.makedirs(models_dir, exist_ok=True)

        model_data = {
            'model': model,
            'scaler': scaler,
            'feature_names': feature_cols,
            'train_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'model_type': 'scalping',
            'params': {
                'take_profit': tp,
                'stop_loss': sl,
                'max_hold_minutes': 60
            },
            'performance': {
                'accuracy': accuracy,
                'f1': f1,
                'num_trades': num_trades,
                'win_rate': win_rate,
                'avg_return': avg_return,
                'total_return': total_return
            }
        }

        model_path = os.path.join(models_dir, f'{ticker.lower()}_scalping.pkl')
        with open(model_path, 'wb') as f:
            pickle.dump(model_data, f)

        print(f"\n  SAVED: {model_path}")
        saved = True
    else:
        print(f"\n  NOT SAVED (accuracy < 50% or win_rate < 40%)")

    return {
        'ticker': ticker,
        'accuracy': accuracy,
        'f1': f1,
        'num_trades': num_trades,
        'win_rate': win_rate,
        'avg_return': avg_return,
        'total_return': total_return,
        'data_points': len(X),
        'saved': saved
    }


def main():
    print("=" * 70)
    print("Scalping Model Training - Surging Stocks TOP 20")
    print("=" * 70)
    print("  TP=5%, SL=3%, MaxHold=60min, Interval=1min")
    print()

    # 급등주 목록 로드 (오프닝 서지 우선, 없으면 기존 파일)
    results_dir = os.path.join(os.path.dirname(__file__), '..', 'results')
    opening_file = os.path.join(results_dir, 'opening_surge_today.csv')
    surging_file = os.path.join(results_dir, 'surging_stocks_today.csv')

    if os.path.exists(opening_file):
        df_surging = pd.read_csv(opening_file)
        tickers = df_surging['ticker'].tolist()
        print(f"  Loaded {len(tickers)} stocks from opening surge scan")
    elif os.path.exists(surging_file):
        df_surging = pd.read_csv(surging_file)
        tickers = df_surging['ticker'].tolist()
        print(f"  Loaded {len(tickers)} surging stocks from file")
    else:
        print("  No surging stocks file found. Running scan...")
        from get_surging_stocks import scan_surging_stocks
        df_surging = scan_surging_stocks(top_n=20)
        tickers = df_surging['ticker'].tolist()

    print(f"  Tickers: {tickers}")
    print()

    # 각 종목별 학습
    all_results = []

    for ticker in tickers:
        print(f"\n  Collecting 1min data for {ticker}...")
        df_raw = collect_intraday_data(ticker, period='5d')

        if df_raw is None or len(df_raw) < 200:
            print(f"  Skipping {ticker}: insufficient data")
            continue

        print(f"  Data: {len(df_raw)} 1-min candles")

        result = train_scalping_model(ticker, df_raw)
        if result:
            all_results.append(result)

    # 최종 요약
    print("\n\n" + "=" * 70)
    print("SCALPING MODEL SUMMARY")
    print("=" * 70)

    if all_results:
        print(f"\n{'Ticker':<8} {'Acc':>7} {'F1':>6} {'Trades':>7} {'WinR':>7} {'AvgRet':>8} {'TotRet':>8} {'Saved'}")
        print("-" * 65)

        for r in all_results:
            saved_mark = "YES" if r['saved'] else "NO"
            print(f"{r['ticker']:<8} {r['accuracy']*100:>6.1f}% {r['f1']:>5.3f} "
                  f"{r['num_trades']:>7} {r['win_rate']:>6.1f}% "
                  f"{r['avg_return']:>+7.2f}% {r['total_return']:>+7.2f}% {saved_mark:>5}")

        saved_models = [r for r in all_results if r['saved']]
        print(f"\n  Total: {len(all_results)} trained, {len(saved_models)} saved")

        if saved_models:
            avg_win = np.mean([r['win_rate'] for r in saved_models])
            avg_ret = np.mean([r['avg_return'] for r in saved_models])
            print(f"  Saved models avg win rate: {avg_win:.1f}%")
            print(f"  Saved models avg return/trade: {avg_ret:+.2f}%")

        # 결과 저장
        df_results = pd.DataFrame(all_results)
        save_path = os.path.join(results_dir, 'scalping_model_results.csv')
        df_results.to_csv(save_path, index=False)
        print(f"\n  Results saved: {save_path}")

    print("\n" + "=" * 70)
    print("Training Complete!")
    print("=" * 70)


if __name__ == "__main__":
    main()
