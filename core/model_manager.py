"""
Sentirax Model Manager - 모델 일일 재학습 + 성과 추적 + 자동 교체

기능:
1. 스윙 모델 일일 재학습 (최신 데이터 반영)
2. 거래 성과 추적 (일별 수익률, 승률, 누적 수익)
3. 성과 기반 모델 자동 교체 (성과 나쁜 모델 비활성화)
"""

import os
import sys
import json
import pickle
import shutil
from datetime import datetime, timedelta
from typing import Optional

import pandas as pd
import numpy as np
import yfinance as yf
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from core.feature_engineer import FeatureEngineer


class PerformanceTracker:
    """거래 성과 추적"""

    def __init__(self, base_dir: str = None):
        if base_dir is None:
            base_dir = os.path.join(os.path.dirname(__file__), '..')
        self.base_dir = base_dir
        self.tracker_dir = os.path.join(base_dir, 'results', 'performance')
        os.makedirs(self.tracker_dir, exist_ok=True)

        self.trades_file = os.path.join(self.tracker_dir, 'trade_history.csv')
        self.daily_file = os.path.join(self.tracker_dir, 'daily_summary.csv')
        self.model_scores_file = os.path.join(self.tracker_dir, 'model_scores.json')

    def log_trade(self, ticker: str, signal: str, price: float,
                  model_type: str, probability: float, timestamp: str = None):
        """거래 기록 저장"""
        if timestamp is None:
            timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

        trade = {
            'timestamp': timestamp,
            'ticker': ticker,
            'signal': signal,  # 'BUY' or 'SELL'
            'price': price,
            'model_type': model_type,  # 'swing' or 'scalping'
            'probability': probability,
            'date': timestamp[:10]
        }

        if os.path.exists(self.trades_file):
            df = pd.read_csv(self.trades_file)
            df = pd.concat([df, pd.DataFrame([trade])], ignore_index=True)
        else:
            df = pd.DataFrame([trade])

        df.to_csv(self.trades_file, index=False)

    def calculate_daily_performance(self, date: str = None):
        """일별 성과 계산 (전일 매수 → 오늘 종가 기준)"""
        if date is None:
            date = datetime.now().strftime('%Y-%m-%d')

        if not os.path.exists(self.trades_file):
            return None

        df = pd.read_csv(self.trades_file)
        # 해당 날짜 이전 매수 기록
        buy_trades = df[(df['signal'] == 'BUY') & (df['date'] <= date)]

        if buy_trades.empty:
            return None

        results = []
        for _, trade in buy_trades.iterrows():
            try:
                stock = yf.Ticker(trade['ticker'])
                hist = stock.history(period='5d')
                if hist.empty:
                    continue

                current_price = hist['Close'].iloc[-1]
                entry_price = trade['price']
                pnl_pct = (current_price / entry_price - 1) * 100

                results.append({
                    'date': date,
                    'ticker': trade['ticker'],
                    'model_type': trade['model_type'],
                    'entry_price': entry_price,
                    'current_price': current_price,
                    'pnl_pct': pnl_pct,
                    'entry_date': trade['date']
                })
            except Exception:
                continue

        if not results:
            return None

        df_results = pd.DataFrame(results)

        # 일별 요약 저장
        summary = {
            'date': date,
            'total_positions': len(df_results),
            'avg_pnl': df_results['pnl_pct'].mean(),
            'total_pnl': df_results['pnl_pct'].sum(),
            'winners': (df_results['pnl_pct'] > 0).sum(),
            'losers': (df_results['pnl_pct'] <= 0).sum(),
            'win_rate': (df_results['pnl_pct'] > 0).mean() * 100,
            'best_ticker': df_results.loc[df_results['pnl_pct'].idxmax(), 'ticker'],
            'best_pnl': df_results['pnl_pct'].max(),
            'worst_ticker': df_results.loc[df_results['pnl_pct'].idxmin(), 'ticker'],
            'worst_pnl': df_results['pnl_pct'].min()
        }

        # 기존 일별 요약에 추가
        if os.path.exists(self.daily_file):
            df_daily = pd.read_csv(self.daily_file)
            df_daily = df_daily[df_daily['date'] != date]  # 중복 제거
            df_daily = pd.concat([df_daily, pd.DataFrame([summary])], ignore_index=True)
        else:
            df_daily = pd.DataFrame([summary])

        df_daily.to_csv(self.daily_file, index=False)

        return summary

    def get_model_score(self, ticker: str, model_type: str, lookback_days: int = 7) -> dict:
        """특정 모델의 최근 성과 점수"""
        if not os.path.exists(self.trades_file):
            return {'score': 50, 'trades': 0, 'reason': 'no_data'}

        df = pd.read_csv(self.trades_file)
        cutoff = (datetime.now() - timedelta(days=lookback_days)).strftime('%Y-%m-%d')

        model_trades = df[
            (df['ticker'] == ticker) &
            (df['model_type'] == model_type) &
            (df['date'] >= cutoff) &
            (df['signal'] == 'BUY')
        ]

        if len(model_trades) < 2:
            return {'score': 50, 'trades': len(model_trades), 'reason': 'insufficient_data'}

        # 각 거래의 결과 확인
        pnl_list = []
        for _, trade in model_trades.iterrows():
            try:
                stock = yf.Ticker(trade['ticker'])
                hist = stock.history(period='5d')
                if not hist.empty:
                    current = hist['Close'].iloc[-1]
                    pnl = (current / trade['price'] - 1) * 100
                    pnl_list.append(pnl)
            except Exception:
                continue

        if not pnl_list:
            return {'score': 50, 'trades': 0, 'reason': 'price_fetch_failed'}

        avg_pnl = np.mean(pnl_list)
        win_rate = sum(1 for p in pnl_list if p > 0) / len(pnl_list) * 100

        # 점수 계산 (0~100)
        # 승률 50% 이상이면 가점, 평균 수익이 양수면 가점
        score = 50
        score += min(avg_pnl * 5, 25)    # 평균 수익 1% = +5점 (최대 +25)
        score += (win_rate - 50) * 0.5     # 승률 60% = +5점
        score = max(0, min(100, score))

        return {
            'score': round(score, 1),
            'trades': len(pnl_list),
            'avg_pnl': round(avg_pnl, 2),
            'win_rate': round(win_rate, 1),
            'reason': 'calculated'
        }

    def update_all_scores(self) -> dict:
        """모든 모델의 점수 업데이트"""
        models_dir = os.path.join(self.base_dir, 'models')
        scores = {}

        for f in os.listdir(models_dir):
            if f.endswith('.pkl'):
                ticker = f.split('_')[0].upper()
                if 'scalping' in f:
                    model_type = 'scalping'
                elif 'top20_500d' in f:
                    model_type = 'swing'
                else:
                    continue

                score_data = self.get_model_score(ticker, model_type)
                key = f"{ticker}_{model_type}"
                scores[key] = score_data

        # 저장
        with open(self.model_scores_file, 'w') as fp:
            json.dump(scores, fp, indent=2)

        return scores

    def get_dashboard(self) -> str:
        """성과 대시보드 텍스트 생성"""
        lines = []
        lines.append("=" * 70)
        lines.append("  SENTIRAX PERFORMANCE DASHBOARD")
        lines.append("=" * 70)

        # 일별 요약
        if os.path.exists(self.daily_file):
            df = pd.read_csv(self.daily_file)
            if not df.empty:
                latest = df.iloc[-1]
                lines.append(f"\n  Latest ({latest['date']}):")
                lines.append(f"    Positions: {int(latest['total_positions'])}")
                lines.append(f"    Avg P&L:   {latest['avg_pnl']:+.2f}%")
                lines.append(f"    Win Rate:  {latest['win_rate']:.1f}%")
                lines.append(f"    Best:      {latest['best_ticker']} ({latest['best_pnl']:+.2f}%)")
                lines.append(f"    Worst:     {latest['worst_ticker']} ({latest['worst_pnl']:+.2f}%)")

                if len(df) > 1:
                    lines.append(f"\n  Last {len(df)} days:")
                    lines.append(f"    Cumulative P&L: {df['total_pnl'].sum():+.2f}%")
                    lines.append(f"    Avg Daily Win Rate: {df['win_rate'].mean():.1f}%")

        # 모델 점수
        if os.path.exists(self.model_scores_file):
            with open(self.model_scores_file) as fp:
                scores = json.load(fp)

            if scores:
                lines.append(f"\n  Model Scores (7-day):")
                lines.append(f"  {'Model':<20} {'Score':>6} {'Trades':>7} {'AvgPnL':>8} {'WinR':>7}")
                lines.append(f"  {'-'*50}")

                sorted_scores = sorted(scores.items(), key=lambda x: x[1]['score'], reverse=True)
                for key, data in sorted_scores:
                    if data['trades'] > 0:
                        lines.append(f"  {key:<20} {data['score']:>5.1f} {data['trades']:>7} "
                                     f"{data.get('avg_pnl', 0):>+7.2f}% {data.get('win_rate', 0):>6.1f}%")

        lines.append("\n" + "=" * 70)
        return "\n".join(lines)


class SwingModelRetrainer:
    """스윙 모델 일일 재학습"""

    def __init__(self, base_dir: str = None):
        if base_dir is None:
            base_dir = os.path.join(os.path.dirname(__file__), '..')
        self.base_dir = base_dir
        self.models_dir = os.path.join(base_dir, 'models')
        self.data_dir = os.path.join(base_dir, 'data')
        self.engineer = FeatureEngineer(label_threshold=1.0)

        self.TOP20_TICKERS = [
            'NVDA', 'AAPL', 'GOOGL', 'MSFT', 'AMZN', 'META', 'TSLA', 'AVGO',
            'WMT', 'LLY', 'JPM', 'XOM', 'JNJ', 'ORCL', 'COST', 'ABBV',
            'HD', 'BAC', 'PG', 'CVX'
        ]

    def collect_fresh_data(self, ticker: str, days: int = 500) -> pd.DataFrame:
        """최신 데이터 수집"""
        try:
            end_date = datetime.now()
            start_date = end_date - timedelta(days=int(days * 1.5))

            stock = yf.Ticker(ticker)
            df_stock = stock.history(start=start_date, end=end_date)

            if df_stock.empty:
                return None

            df = pd.DataFrame({
                'Close': df_stock['Close'],
                'Volume': df_stock['Volume']
            })

            # 매크로 지표
            macros = {
                'vix': '^VIX', 'treasury_10y': '^TNX',
                'oil': 'CL=F', 'nasdaq': '^IXIC', 'sp500': '^GSPC'
            }

            for name, symbol in macros.items():
                try:
                    macro = yf.Ticker(symbol)
                    df_macro = macro.history(start=start_date, end=end_date)
                    df[name] = df_macro['Close'].reindex(df.index, method='ffill')
                except Exception:
                    df[name] = np.nan

            # 기술 지표
            delta = df['Close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            rs = gain / loss
            df['rsi'] = 100 - (100 / (1 + rs))

            for period in [5, 20, 50]:
                df[f'ma_{period}'] = df['Close'].rolling(window=period).mean()

            df['volume_ratio'] = df['Volume'] / df['Volume'].rolling(window=20).mean()
            df['volume_ma_5'] = df['Volume'].rolling(window=5).mean()
            df['volume_ma_20'] = df['Volume'].rolling(window=20).mean()
            df['volume_trend'] = (df['volume_ma_5'] / df['volume_ma_20']).fillna(1)
            df['next_day_return'] = df['Close'].pct_change().shift(-1) * 100

            df = df.dropna(subset=['ma_50'])

            return df

        except Exception as e:
            print(f"  Data collection error for {ticker}: {e}")
            return None

    def retrain_single(self, ticker: str, tracker: PerformanceTracker = None) -> dict:
        """단일 종목 모델 재학습"""

        print(f"\n  [{ticker}] Retraining...")

        # 1. 최신 데이터 수집
        df = self.collect_fresh_data(ticker)
        if df is None or len(df) < 100:
            print(f"  [{ticker}] Insufficient data")
            return {'ticker': ticker, 'success': False, 'reason': 'no_data'}

        df.index = pd.to_datetime(df.index, utc=True).tz_localize(None)

        # 2. Feature Engineering
        X_clean, y_clean = self.engineer.prepare_ml_data(df)
        df_clean = df.loc[X_clean.index]
        feature_cols = X_clean.columns.tolist()

        if len(X_clean) < 50:
            print(f"  [{ticker}] Too few samples: {len(X_clean)}")
            return {'ticker': ticker, 'success': False, 'reason': 'few_samples'}

        # 3. Train/Test Split (80/20)
        split_idx = int(len(X_clean) * 0.8)
        X_train, X_test = X_clean.iloc[:split_idx], X_clean.iloc[split_idx:]
        y_train, y_test = y_clean.iloc[:split_idx], y_clean.iloc[split_idx:]
        df_test = df_clean.iloc[split_idx:]

        # 4. 정규화 + 학습
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        model = LogisticRegression(random_state=42, max_iter=1000)
        model.fit(X_train_scaled, y_train)

        # 5. 평가
        y_pred = model.predict(X_test_scaled)
        accuracy = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, zero_division=0)

        # 백테스팅
        df_test_copy = df_test.copy()
        df_test_copy['prediction'] = y_pred
        df_test_copy['actual_return'] = df_test_copy['Close'].pct_change().shift(-1) * 100
        df_test_copy['strategy_return'] = 0.0
        df_test_copy.loc[df_test_copy['prediction'] == 1, 'strategy_return'] = df_test_copy['actual_return']

        ml_return = (1 + df_test_copy['strategy_return'] / 100).prod() - 1
        ml_return_pct = ml_return * 100
        bh_return_pct = (df_test['Close'].iloc[-1] / df_test['Close'].iloc[0] - 1) * 100

        # 6. 기존 모델과 비교
        old_model_path = os.path.join(self.models_dir, f'{ticker.lower()}_top20_500d.pkl')
        old_accuracy = 0
        old_ml_return = -999

        if os.path.exists(old_model_path):
            with open(old_model_path, 'rb') as f:
                old_data = pickle.load(f)
            old_perf = old_data.get('performance', {})
            old_accuracy = old_perf.get('test_accuracy', 0)
            old_ml_return = old_perf.get('ml_return', -999)

        # 7. 성과 점수 반영
        model_score = 50
        if tracker:
            score_data = tracker.get_model_score(ticker, 'swing')
            model_score = score_data['score']

        # 8. 교체 결정
        # 새 모델이 기존보다 나으면 교체, 아니면 유지
        new_is_better = (
            (accuracy >= 0.45 and ml_return_pct > -15) and  # 최소 기준
            (accuracy >= old_accuracy - 0.02 or ml_return_pct > old_ml_return)  # 기존보다 비슷하거나 나음
        )

        # 실전 성과가 나쁘면 (score < 30) 강제 재학습
        force_retrain = model_score < 30

        should_save = new_is_better or force_retrain

        result = {
            'ticker': ticker,
            'success': True,
            'new_accuracy': accuracy,
            'new_ml_return': ml_return_pct,
            'old_accuracy': old_accuracy,
            'old_ml_return': old_ml_return,
            'model_score': model_score,
            'replaced': should_save,
            'reason': 'better' if new_is_better else ('forced' if force_retrain else 'kept_old')
        }

        if should_save:
            # 기존 모델 백업
            if os.path.exists(old_model_path):
                backup_dir = os.path.join(self.models_dir, 'backup')
                os.makedirs(backup_dir, exist_ok=True)
                date_str = datetime.now().strftime('%Y%m%d')
                backup_path = os.path.join(backup_dir, f'{ticker.lower()}_top20_500d_{date_str}.pkl')
                shutil.copy2(old_model_path, backup_path)

            # 새 모델 저장
            model_data = {
                'model': model,
                'scaler': scaler,
                'feature_names': feature_cols,
                'train_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'performance': {
                    'train_samples': len(X_train),
                    'test_samples': len(X_test),
                    'test_accuracy': accuracy,
                    'test_f1': f1,
                    'ml_return': ml_return_pct,
                    'bh_return': bh_return_pct,
                    'excess_return': ml_return_pct - bh_return_pct,
                },
                'retrain_info': {
                    'replaced': True,
                    'reason': result['reason'],
                    'old_accuracy': old_accuracy,
                    'old_ml_return': old_ml_return,
                    'model_score': model_score
                }
            }

            with open(old_model_path, 'wb') as f:
                pickle.dump(model_data, f)

            action = "REPLACED" if new_is_better else "FORCE-REPLACED"
            print(f"  [{ticker}] {action}: acc {old_accuracy*100:.1f}% -> {accuracy*100:.1f}%, "
                  f"ret {old_ml_return:+.1f}% -> {ml_return_pct:+.1f}%")
        else:
            print(f"  [{ticker}] KEPT: old acc={old_accuracy*100:.1f}% >= new={accuracy*100:.1f}%")

        return result

    def retrain_all(self, tracker: PerformanceTracker = None) -> list:
        """전체 종목 재학습"""
        print("=" * 70)
        print("  SWING MODEL DAILY RETRAIN")
        print(f"  Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"  Tickers: {len(self.TOP20_TICKERS)}")
        print("=" * 70)

        results = []
        for ticker in self.TOP20_TICKERS:
            result = self.retrain_single(ticker, tracker)
            results.append(result)

        # 요약
        success = [r for r in results if r['success']]
        replaced = [r for r in success if r.get('replaced', False)]
        kept = [r for r in success if not r.get('replaced', False)]

        print(f"\n{'='*70}")
        print(f"  RETRAIN SUMMARY")
        print(f"{'='*70}")
        print(f"  Total: {len(self.TOP20_TICKERS)}")
        print(f"  Success: {len(success)}")
        print(f"  Replaced: {len(replaced)} models")
        print(f"  Kept old: {len(kept)} models")

        if replaced:
            print(f"\n  Replaced models:")
            for r in replaced:
                print(f"    {r['ticker']}: acc {r['old_accuracy']*100:.1f}%->{r['new_accuracy']*100:.1f}%, "
                      f"ret {r['old_ml_return']:+.1f}%->{r['new_ml_return']:+.1f}% ({r['reason']})")

        # 결과 저장
        results_dir = os.path.join(self.base_dir, 'results', 'performance')
        os.makedirs(results_dir, exist_ok=True)
        df_results = pd.DataFrame(results)
        date_str = datetime.now().strftime('%Y%m%d')
        df_results.to_csv(os.path.join(results_dir, f'retrain_{date_str}.csv'), index=False)

        return results


class ModelAutoSwitcher:
    """성과 기반 모델 자동 교체/비활성화"""

    def __init__(self, base_dir: str = None):
        if base_dir is None:
            base_dir = os.path.join(os.path.dirname(__file__), '..')
        self.base_dir = base_dir
        self.models_dir = os.path.join(base_dir, 'models')
        self.config_file = os.path.join(self.models_dir, 'active_models.json')

    def load_config(self) -> dict:
        """활성 모델 설정 로드"""
        if os.path.exists(self.config_file):
            with open(self.config_file) as f:
                return json.load(f)
        return {}

    def save_config(self, config: dict):
        """활성 모델 설정 저장"""
        with open(self.config_file, 'w') as f:
            json.dump(config, f, indent=2)

    def evaluate_and_switch(self, tracker: PerformanceTracker) -> dict:
        """모든 모델 평가 후 활성/비활성 결정"""

        print("\n" + "=" * 70)
        print("  MODEL AUTO-SWITCH EVALUATION")
        print("=" * 70)

        scores = tracker.update_all_scores()
        config = self.load_config()
        changes = []

        for key, score_data in scores.items():
            ticker, model_type = key.rsplit('_', 1)
            score = score_data['score']
            trades = score_data['trades']

            prev_status = config.get(key, {}).get('active', True)

            # 판단 로직
            if trades < 2:
                # 데이터 부족 - 유지
                active = prev_status
                reason = 'insufficient_data'
            elif score < 25:
                # 매우 나쁨 - 비활성화
                active = False
                reason = f'low_score ({score:.1f})'
            elif score < 40 and prev_status:
                # 나쁨 - 비활성화
                active = False
                reason = f'underperforming ({score:.1f})'
            elif score >= 50 and not prev_status:
                # 좋아짐 - 재활성화
                active = True
                reason = f'recovered ({score:.1f})'
            else:
                active = prev_status
                reason = 'no_change'

            config[key] = {
                'active': active,
                'score': score,
                'trades': trades,
                'last_updated': datetime.now().strftime('%Y-%m-%d'),
                'reason': reason
            }

            if active != prev_status:
                status = "ACTIVATED" if active else "DEACTIVATED"
                changes.append(f"  {status}: {key} (score={score:.1f})")
                print(f"  {status}: {key} (score={score:.1f}, reason={reason})")

        self.save_config(config)

        if not changes:
            print("  No changes needed.")

        # 활성/비활성 요약
        active_count = sum(1 for v in config.values() if v.get('active', True))
        inactive_count = len(config) - active_count

        print(f"\n  Active models: {active_count}")
        print(f"  Inactive models: {inactive_count}")

        return config

    def get_active_tickers(self, model_type: str) -> list:
        """활성 상태인 모델의 티커 목록"""
        config = self.load_config()

        if not config:
            # 설정 없으면 모든 모델 활성
            return None

        active = []
        for key, data in config.items():
            if key.endswith(f'_{model_type}') and data.get('active', True):
                ticker = key.replace(f'_{model_type}', '')
                active.append(ticker)

        return active if active else None
