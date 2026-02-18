"""
국내주식 스캘핑 모델 학습

- 국내 급등주 TOP 20 대상 1분봉 데이터 학습
- 해외 scalping_model과 동일한 feature engineering
- 저장: models/{api_ticker}_kr_scalping.pkl
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
from datetime import datetime
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score

# 공유 Feature Engineering (core/scalping_signals.py)
from core.scalping_signals import create_scalping_features


PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def collect_kr_intraday(yf_ticker: str, period: str = '5d') -> pd.DataFrame:
    """1분봉 데이터 수집 (KS/KQ 종목)"""
    try:
        stock = yf.Ticker(yf_ticker)
        df = stock.history(period=period, interval='1m')
        if df.empty or len(df) < 100:
            return None
        return df
    except Exception as e:
        print(f"  데이터 수집 오류 {yf_ticker}: {e}")
        return None


def label_scalping(df: pd.DataFrame, take_profit: float, stop_loss: float,
                   max_hold_minutes: int = 60) -> pd.Series:
    """스캘핑 레이블링 (1=익절, 0=손절/타임아웃)"""
    labels = pd.Series(index=df.index, dtype=float)
    close = df['Close'].values

    for i in range(len(close) - max_hold_minutes):
        entry_price = close[i]
        tp_price = entry_price * (1 + take_profit / 100)
        sl_price = entry_price * (1 - stop_loss / 100)
        label = 0

        for j in range(i + 1, min(i + max_hold_minutes + 1, len(close))):
            if close[j] >= tp_price:
                label = 1
                break
            elif close[j] <= sl_price:
                break

        labels.iloc[i] = label

    return labels


def train_kr_scalping_model(api_ticker: str, yf_ticker: str, name: str,
                             df_raw: pd.DataFrame) -> dict:
    """단일 종목 국내 스캘핑 모델 학습"""
    print(f"\n{'='*70}")
    print(f"  {name} ({api_ticker}) 국내 스캘핑 모델 학습")
    print(f"{'='*70}")

    features = create_scalping_features(df_raw)

    # TP/SL 자동 설정
    avg_range = (df_raw['High'] / df_raw['Low'] - 1).mean() * 100
    tp = min(max(round(avg_range * 1.5, 1), 1.5), 5.0)
    sl = min(max(round(avg_range * 1.0, 1), 1.0), 3.0)

    print(f"  Labeling (TP={tp}%, SL={sl}%, MaxHold=60min)...")
    labels = label_scalping(df_raw, take_profit=tp, stop_loss=sl, max_hold_minutes=60)

    feature_cols = [c for c in features.columns if c not in
                    ['close', 'volume', 'ma_5m', 'ma_10m', 'ma_20m', 'ma_60m']]

    combined = features[feature_cols].copy()
    combined['label'] = labels
    combined = combined.dropna()

    if len(combined) < 100:
        print(f"  데이터 부족: {len(combined)}행")
        return None

    X = combined[feature_cols]
    y = combined['label'].astype(int)

    buy_ratio = y.mean() * 100
    print(f"  데이터: {len(X)}행, 매수신호: {buy_ratio:.1f}%")

    split_idx = int(len(X) * 0.8)
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

    if len(y_train.unique()) < 2 or len(y_test.unique()) < 2 or y_train.sum() < 10:
        print(f"  학습 조건 미충족 (클래스 or 신호 부족). 건너뜀.")
        return None

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    model = GradientBoostingClassifier(
        n_estimators=150, max_depth=4,
        learning_rate=0.1, subsample=0.8, random_state=42
    )
    model.fit(X_train_scaled, y_train)

    y_pred = model.predict(X_test_scaled)
    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, zero_division=0)

    # 백테스팅
    close_prices = df_raw['Close'].reindex(X_test.index).values
    test_trades = []
    for i in range(len(y_pred)):
        if y_pred[i] == 1 and i < len(close_prices):
            entry_price = close_prices[i]
            trade_result = 0
            for j in range(i + 1, min(i + 61, len(close_prices))):
                pnl = (close_prices[j] / entry_price - 1) * 100
                if pnl >= tp:
                    trade_result = tp
                    break
                elif pnl <= -sl:
                    trade_result = -sl
                    break
            test_trades.append(trade_result)

    num_trades = len(test_trades)
    win_rate = sum(1 for t in test_trades if t > 0) / num_trades * 100 if num_trades > 0 else 0
    avg_return = np.mean(test_trades) if test_trades else 0
    total_return = sum(test_trades) if test_trades else 0

    print(f"  Accuracy: {accuracy*100:.1f}%  F1: {f1:.3f}")
    print(f"  Trades: {num_trades}  WinRate: {win_rate:.1f}%  AvgRet: {avg_return:+.2f}%")

    saved = False
    if accuracy >= 0.50 and win_rate >= 40:
        models_dir = os.path.join(PROJECT_ROOT, 'models')
        os.makedirs(models_dir, exist_ok=True)

        model_data = {
            'model': model,
            'scaler': scaler,
            'feature_names': feature_cols,
            'train_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'model_type': 'kr_scalping',
            'api_ticker': api_ticker,
            'yf_ticker': yf_ticker,
            'name': name,
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

        model_path = os.path.join(models_dir, f'{api_ticker}_kr_scalping.pkl')
        with open(model_path, 'wb') as fp:
            pickle.dump(model_data, fp)

        print(f"  SAVED: {model_path}")
        saved = True
    else:
        print(f"  NOT SAVED (accuracy < 50% or win_rate < 40%)")

    return {
        'api_ticker': api_ticker,
        'name': name,
        'accuracy': accuracy,
        'f1': f1,
        'num_trades': num_trades,
        'win_rate': win_rate,
        'avg_return': avg_return,
        'total_return': total_return,
        'saved': saved
    }


def main():
    print("=" * 70)
    print("국내주식 스캘핑 모델 학습 - 급등주 TOP 20")
    print("=" * 70)

    results_dir = os.path.join(PROJECT_ROOT, 'results')
    surge_file = os.path.join(results_dir, 'domestic_surge_today.csv')

    if os.path.exists(surge_file):
        df_surge = pd.read_csv(surge_file)
        stocks = list(zip(df_surge['api_ticker'].astype(str).str.zfill(6),
                          df_surge['yf_ticker'],
                          df_surge['name']))
        print(f"  급등주 {len(stocks)}개 로드 완료")
    else:
        print("  domestic_surge_today.csv 없음. 스캔 실행...")
        from get_domestic_surging_stocks import scan_domestic_surging_stocks
        df_surge = scan_domestic_surging_stocks(top_n=20, opening_surge=False)
        if df_surge.empty:
            print("  급등주 없음. 종료.")
            return
        stocks = list(zip(df_surge['api_ticker'].astype(str).str.zfill(6),
                          df_surge['yf_ticker'],
                          df_surge['name']))

    all_results = []

    for api_ticker, yf_ticker, name in stocks:
        print(f"\n  {name}({yf_ticker}) 1분봉 데이터 수집 중...")
        df_raw = collect_kr_intraday(yf_ticker, period='5d')

        if df_raw is None or len(df_raw) < 200:
            print(f"  건너뜀: 데이터 부족")
            continue

        print(f"  데이터: {len(df_raw)}개 1분봉")
        result = train_kr_scalping_model(api_ticker, yf_ticker, name, df_raw)
        if result:
            all_results.append(result)

    print("\n\n" + "=" * 70)
    print("국내 스캘핑 모델 학습 요약")
    print("=" * 70)

    if all_results:
        saved = [r for r in all_results if r['saved']]
        print(f"\n  총 {len(all_results)}개 학습, {len(saved)}개 저장")
        print(f"\n  {'종목':12s} {'Acc':>7} {'WinR':>7} {'AvgRet':>8} {'Saved'}")
        print("  " + "-" * 45)
        for r in all_results:
            mark = "YES" if r['saved'] else "NO"
            print(f"  {r['name']:12s} {r['accuracy']*100:>6.1f}% {r['win_rate']:>6.1f}% "
                  f"{r['avg_return']:>+7.2f}% {mark:>5}")
    else:
        print("  저장된 모델 없음")


if __name__ == '__main__':
    main()
