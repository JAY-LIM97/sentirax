import pandas as pd
import numpy as np
from typing import Dict, List, Tuple

class BacktestEvaluator:
    """백테스팅 평가 시스템 - 개선 버전"""
    
    def __init__(self, data: pd.DataFrame):
        """
        Args:
            data: DataFrame with columns including:
                - sentiment_score (optional)
                - next_day_return
                - Close or close_price
                - volume
                - news_count
        """
        self.data = data.copy()
        self.data = self.data.dropna(subset=['next_day_return'])
        
        # 컬럼명 표준화
        if 'close_price' in self.data.columns and 'Close' not in self.data.columns:
            self.data['Close'] = self.data['close_price']
        
        if 'Close' not in self.data.columns:
            raise ValueError("❌ 주가 데이터 (Close 또는 close_price) 컬럼이 필요합니다!")
        
        # 거래량 지표 계산
        self._calculate_volume_indicators()
        
        # RSI 계산 (없으면)
        if 'rsi' not in self.data.columns:
            self._calculate_rsi()
    
    def _calculate_volume_indicators(self):
        """거래량 기반 지표 계산"""
        if 'volume' in self.data.columns:
            # 거래량 이동평균
            self.data['volume_ma_5'] = self.data['volume'].rolling(window=5, min_periods=1).mean()
            self.data['volume_ma_20'] = self.data['volume'].rolling(window=20, min_periods=1).mean()
            
            # 거래량 비율
            self.data['volume_ratio'] = self.data['volume'] / self.data['volume_ma_5']
            
            # 거래량 추세
            self.data['volume_trend'] = (self.data['volume_ma_5'] / self.data['volume_ma_20'] - 1) * 100
    
    def _calculate_rsi(self, period: int = 14):
        """RSI 계산"""
        if 'Close' in self.data.columns:
            delta = self.data['Close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=period, min_periods=1).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=period, min_periods=1).mean()
            
            rs = gain / loss
            self.data['rsi'] = 100 - (100 / (1 + rs))
            self.data['rsi'] = self.data['rsi'].fillna(50)
    
    def generate_composite_signal(self, row: pd.Series) -> float:
        """
        복합 신호 생성 (거래량 + 감성 + 뉴스)
        
        Returns:
            -1.0 ~ +1.0 사이의 신호 강도
        """
        score = 0.0
        
        # 1. 거래량 (가중치 50%)
        volume_ratio = row.get('volume_ratio', 1.0)
        if volume_ratio > 1.3:  # 30% 이상 증가
            score += 0.5
        elif volume_ratio > 1.1:
            score += 0.3
        elif volume_ratio < 0.8:  # 20% 이상 감소
            score -= 0.3
        
        # 2. 뉴스 개수 (가중치 20%)
        news_count = row.get('news_count', 0)
        if news_count >= 5:
            score += 0.2
        elif news_count >= 3:
            score += 0.1
        
        # 3. 감성 점수 (가중치 20%)
        sentiment = row.get('sentiment_score', 0)
        score += sentiment * 0.2
        
        # 4. RSI (가중치 10%)
        rsi = row.get('rsi', 50)
        if rsi < 30:  # 과매도
            score += 0.1
        elif rsi > 70:  # 과매수
            score -= 0.1
        
        return np.clip(score, -1.0, 1.0)
    
    def evaluate_sentiment_strategy(self, 
                                   buy_threshold: float = 0.3,
                                   sell_threshold: float = -0.3) -> Dict:
        """감성 점수 기반 전략 평가"""
        
        if 'sentiment_score' not in self.data.columns:
            print("⚠️ 감성 점수 없음 - 평가 불가")
            return {}
        
        print("📊 감성 기반 전략 평가")
        print("-"*60)
        
        predictions = []
        actuals = []
        
        for idx, row in self.data.iterrows():
            sentiment = row['sentiment_score']
            actual_return = row['next_day_return']
            
            # 예측
            if sentiment >= buy_threshold:
                predicted = 1
            elif sentiment <= sell_threshold:
                predicted = -1
            else:
                predicted = 0
            
            # 실제
            if actual_return > 0:
                actual = 1
            elif actual_return < 0:
                actual = -1
            else:
                actual = 0
            
            predictions.append(predicted)
            actuals.append(actual)
        
        # 메트릭 계산
        correct = sum(1 for p, a in zip(predictions, actuals) 
                     if (p == a) or (p == 0))
        accuracy = correct / len(predictions) * 100 if len(predictions) > 0 else 0
        
        tp = sum(1 for p, a in zip(predictions, actuals) if p == 1 and a == 1)
        fp = sum(1 for p, a in zip(predictions, actuals) if p == 1 and a != 1)
        fn = sum(1 for p, a in zip(predictions, actuals) if p != 1 and a == 1)
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        results = {
            'accuracy': accuracy,
            'precision': precision * 100,
            'recall': recall * 100,
            'f1_score': f1 * 100,
            'buy_signals': sum(1 for p in predictions if p == 1),
            'sell_signals': sum(1 for p in predictions if p == -1),
            'hold_signals': sum(1 for p in predictions if p == 0)
        }
        
        print(f"✅ 정확도: {results['accuracy']:.1f}%")
        print(f"✅ 정밀도: {results['precision']:.1f}%")
        print(f"✅ 재현율: {results['recall']:.1f}%")
        print(f"✅ F1 Score: {results['f1_score']:.1f}%")
        print(f"\n📊 신호 분포:")
        print(f"   매수: {results['buy_signals']}회")
        print(f"   매도: {results['sell_signals']}회")
        print(f"   관망: {results['hold_signals']}회")
        
        return results
    
    def simulate_sentiment_strategy(self, 
                                   initial_capital: float = 10000,
                                   buy_threshold: float = 0.05,
                                   sell_threshold: float = -0.05,
                                   stop_loss: float = 0.10,
                                   take_profit: float = 0.20) -> Dict:
        """감성 기반 트레이딩 시뮬레이션 (손절/익절 포함)"""
        
        print("\n💰 감성 기반 트레이딩 시뮬레이션")
        print("-"*60)
        print(f"초기 자본: ${initial_capital:,.2f}")
        print(f"매수 기준: 감성 ≥ {buy_threshold:+.2f}")
        print(f"매도 기준: 감성 ≤ {sell_threshold:+.2f}")
        print(f"손절: -{stop_loss*100:.0f}% | 익절: +{take_profit*100:.0f}%\n")
        
        capital = initial_capital
        position = 0
        buy_price = 0
        trades = []
        
        for idx, row in self.data.iterrows():
            sentiment = row.get('sentiment_score', 0)
            price = row['Close']
            
            # 매수
            if sentiment >= buy_threshold and position == 0:
                position = 1
                buy_price = price
                print(f"📈 {idx}: 매수 @ ${price:.2f} (감성: {sentiment:+.2f})")
            
            # 손절/익절/매도 신호
            elif position == 1:
                current_return = (price - buy_price) / buy_price
                
                # 손절
                if current_return <= -stop_loss:
                    profit = current_return
                    capital *= (1 + profit)
                    trades.append({'return': profit * 100, 'win': False, 'type': '손절'})
                    print(f"🛑 {idx}: 손절 @ ${price:.2f} ({profit*100:+.2f}%)")
                    position = 0
                
                # 익절
                elif current_return >= take_profit:
                    profit = current_return
                    capital *= (1 + profit)
                    trades.append({'return': profit * 100, 'win': True, 'type': '익절'})
                    print(f"💰 {idx}: 익절 @ ${price:.2f} ({profit*100:+.2f}%)")
                    position = 0
                
                # 매도 신호
                elif sentiment <= sell_threshold:
                    profit = current_return
                    capital *= (1 + profit)
                    trades.append({'return': profit * 100, 'win': profit > 0, 'type': '신호'})
                    print(f"📉 {idx}: 매도 @ ${price:.2f} ({profit*100:+.2f}%) (감성: {sentiment:+.2f})")
                    position = 0
        
        # 청산
        if position == 1:
            sell_price = self.data.iloc[-1]['Close']
            profit = (sell_price - buy_price) / buy_price
            capital *= (1 + profit)
            trades.append({'return': profit * 100, 'win': profit > 0, 'type': '청산'})
            print(f"📉 {self.data.index[-1]}: 청산 @ ${sell_price:.2f} ({profit*100:+.2f}%)")
        
        return self._calculate_performance(capital, initial_capital, trades)
    
    def simulate_volume_strategy(self,
                                initial_capital: float = 10000,
                                volume_threshold: float = 1.3) -> Dict:
        """거래량 기반 트레이딩 시뮬레이션"""
        
        print("\n💰 거래량 기반 트레이딩 시뮬레이션")
        print("-"*60)
        print(f"초기 자본: ${initial_capital:,.2f}")
        print(f"거래량 급증 기준: {volume_threshold:.1f}x 이상\n")
        
        if 'volume_ratio' not in self.data.columns:
            print("❌ 거래량 데이터 없음")
            return {}
        
        capital = initial_capital
        position = 0
        buy_price = 0
        trades = []
        
        for idx, row in self.data.iterrows():
            price = row['Close']
            vol_ratio = row['volume_ratio']
            
            # 거래량 급증 → 매수
            if vol_ratio >= volume_threshold and position == 0:
                position = 1
                buy_price = price
                print(f"📈 {idx}: 매수 @ ${price:.2f} (거래량: {vol_ratio:.2f}x)")
            
            # 거래량 감소 → 매도
            elif vol_ratio < 0.9 and position == 1:
                profit = (price - buy_price) / buy_price
                capital *= (1 + profit)
                trades.append({'return': profit * 100, 'win': profit > 0, 'type': '신호'})
                print(f"📉 {idx}: 매도 @ ${price:.2f} ({profit*100:+.2f}%) (거래량: {vol_ratio:.2f}x)")
                position = 0
        
        # 청산
        if position == 1:
            sell_price = self.data.iloc[-1]['Close']
            profit = (sell_price - buy_price) / buy_price
            capital *= (1 + profit)
            trades.append({'return': profit * 100, 'win': profit > 0, 'type': '청산'})
        
        return self._calculate_performance(capital, initial_capital, trades)
    
    def simulate_composite_strategy(self,
                                   initial_capital: float = 10000,
                                   buy_threshold: float = 0.4,
                                   sell_threshold: float = -0.3) -> Dict:
        """복합 신호 기반 트레이딩 시뮬레이션"""
        
        print("\n💰 복합 신호 트레이딩 시뮬레이션")
        print("-"*60)
        print(f"초기 자본: ${initial_capital:,.2f}")
        print(f"복합 신호 (거래량 50% + 뉴스 20% + 감성 20% + RSI 10%)")
        print(f"매수 기준: 신호 ≥ {buy_threshold:+.2f}")
        print(f"매도 기준: 신호 ≤ {sell_threshold:+.2f}\n")
        
        capital = initial_capital
        position = 0
        buy_price = 0
        trades = []
        
        for idx, row in self.data.iterrows():
            price = row['Close']
            signal = self.generate_composite_signal(row)
            
            # 매수
            if signal >= buy_threshold and position == 0:
                position = 1
                buy_price = price
                print(f"📈 {idx}: 매수 @ ${price:.2f} (신호: {signal:+.2f})")
            
            # 매도
            elif signal <= sell_threshold and position == 1:
                profit = (price - buy_price) / buy_price
                capital *= (1 + profit)
                trades.append({'return': profit * 100, 'win': profit > 0, 'type': '신호'})
                print(f"📉 {idx}: 매도 @ ${price:.2f} ({profit*100:+.2f}%) (신호: {signal:+.2f})")
                position = 0
        
        # 청산
        if position == 1:
            sell_price = self.data.iloc[-1]['Close']
            profit = (sell_price - buy_price) / buy_price
            capital *= (1 + profit)
            trades.append({'return': profit * 100, 'win': profit > 0, 'type': '청산'})
        
        return self._calculate_performance(capital, initial_capital, trades)
    
    def _calculate_performance(self, final_capital: float, initial_capital: float,
                              trades: List[Dict]) -> Dict:
        """성과 지표 계산"""
        
        total_return = (final_capital - initial_capital) / initial_capital * 100
        buy_hold_return = (self.data.iloc[-1]['Close'] - self.data.iloc[0]['Close']) / self.data.iloc[0]['Close'] * 100
        
        num_trades = len(trades)
        wins = sum(1 for t in trades if t['win'])
        losses = num_trades - wins
        win_rate = (wins / num_trades * 100) if num_trades > 0 else 0
        
        # 평균 수익/손실
        if trades:
            winning_trades = [t['return'] for t in trades if t['win']]
            losing_trades = [t['return'] for t in trades if not t['win']]
            avg_win = np.mean(winning_trades) if winning_trades else 0
            avg_loss = np.mean(losing_trades) if losing_trades else 0
        else:
            avg_win = 0
            avg_loss = 0
        
        results = {
            'initial_capital': initial_capital,
            'final_capital': final_capital,
            'total_return': total_return,
            'buy_hold_return': buy_hold_return,
            'excess_return': total_return - buy_hold_return,
            'num_trades': num_trades,
            'wins': wins,
            'losses': losses,
            'win_rate': win_rate,
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'profit_factor': abs(avg_win / avg_loss) if avg_loss != 0 else 0,
            'trades': trades
        }
        
        print(f"\n{'='*60}")
        print(f"💰 최종 자본: ${final_capital:,.2f}")
        print(f"📈 총 수익률: {total_return:+.2f}%")
        print(f"📊 Buy & Hold: {buy_hold_return:+.2f}%")
        print(f"🎯 초과 수익: {results['excess_return']:+.2f}%")
        print(f"🔄 거래 횟수: {num_trades}회")
        if num_trades > 0:
            print(f"✅ 승: {wins}회 | ❌ 패: {losses}회")
            print(f"📊 승률: {win_rate:.1f}%")
            print(f"💹 평균 수익: {avg_win:+.2f}% | 평균 손실: {avg_loss:+.2f}%")
            if results['profit_factor'] > 0:
                print(f"⚖️ Profit Factor: {results['profit_factor']:.2f}")
        print(f"{'='*60}")
        
        return results
    
    def calculate_correlation(self) -> pd.Series:
        """특징 간 상관관계 계산"""
        
        print("\n📊 상관관계 분석")
        print("-"*60)
        
        numeric_cols = self.data.select_dtypes(include=[np.number]).columns
        correlations = self.data[numeric_cols].corr()['next_day_return'].sort_values(ascending=False)
        
        print("\n🔗 next_day_return과의 상관관계:")
        for col, corr in correlations.items():
            if col != 'next_day_return':
                emoji = "🟢" if corr > 0.2 else "🔴" if corr < -0.2 else "⚪"
                print(f"   {emoji} {col:25s}: {corr:+.3f}")
        
        return correlations
    
    def compare_all_strategies(self, initial_capital: float = 10000) -> pd.DataFrame:
        """모든 전략 비교"""
        
        print("\n" + "="*60)
        print("🔬 전략 비교 분석")
        print("="*60 + "\n")
        
        results = []
        
        # 1. Buy & Hold
        buy_hold_return = (self.data.iloc[-1]['Close'] - self.data.iloc[0]['Close']) / self.data.iloc[0]['Close'] * 100
        results.append({
            'strategy': 'Buy & Hold',
            'return': buy_hold_return,
            'trades': 1,
            'win_rate': 100 if buy_hold_return > 0 else 0
        })
        
        # 2. 감성 전략
        if 'sentiment_score' in self.data.columns:
            sentiment_result = self.simulate_sentiment_strategy(initial_capital, buy_threshold=0.05, sell_threshold=-0.05)
            results.append({
                'strategy': 'Sentiment',
                'return': sentiment_result['total_return'],
                'trades': sentiment_result['num_trades'],
                'win_rate': sentiment_result['win_rate']
            })
        
        # 3. 거래량 전략
        if 'volume_ratio' in self.data.columns:
            volume_result = self.simulate_volume_strategy(initial_capital)
            results.append({
                'strategy': 'Volume',
                'return': volume_result['total_return'],
                'trades': volume_result['num_trades'],
                'win_rate': volume_result['win_rate']
            })
        
        # 4. 복합 전략
        composite_result = self.simulate_composite_strategy(initial_capital)
        results.append({
            'strategy': 'Composite',
            'return': composite_result['total_return'],
            'trades': composite_result['num_trades'],
            'win_rate': composite_result['win_rate']
        })
        
        # DataFrame 생성
        df = pd.DataFrame(results)
        df = df.sort_values('return', ascending=False)
        
        print("\n📊 전략 비교 결과:")
        print("="*60)
        print(df.to_string(index=False))
        print("="*60)
        
        return df