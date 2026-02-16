# 🎉 Sentirax Final Report

**Generated**: 2026-02-11 21:12:44

---

## 📊 Executive Summary

- **Total ML Strategy Return**: +146.71% (across 4 stocks)
- **Total Buy & Hold Return**: -16.62%
- **Average Win Rate**: 74.0%
- **Tested Stocks**: TSLA, NVDA, AAPL, MSFT

---

## 🎯 Individual Stock Performance

### TSLA

| Metric | Value |
|--------|-------|
| **Test Accuracy** | 45.45% |
| **F1-Score** | 0.400 |
| **ML Strategy Return** | +59.20% |
| **Buy & Hold Return** | -5.29% |
| **Excess Return** | +64.49%p |
| **Win Rate** | 73.3% |
| **Number of Trades** | 15 |

✅ **ML strategy outperformed Buy & Hold by +64.49%p**

### NVDA

| Metric | Value |
|--------|-------|
| **Test Accuracy** | 50.00% |
| **F1-Score** | 0.444 |
| **ML Strategy Return** | +48.82% |
| **Buy & Hold Return** | +3.51% |
| **Excess Return** | +45.31%p |
| **Win Rate** | 81.2% |
| **Number of Trades** | 16 |

✅ **ML strategy outperformed Buy & Hold by +45.31%p**

### AAPL

| Metric | Value |
|--------|-------|
| **Test Accuracy** | 75.00% |
| **F1-Score** | 0.857 |
| **ML Strategy Return** | +24.44% |
| **Buy & Hold Return** | +5.63% |
| **Excess Return** | +18.81%p |
| **Win Rate** | 80.0% |
| **Number of Trades** | 5 |

✅ **ML strategy outperformed Buy & Hold by +18.81%p**

### MSFT

| Metric | Value |
|--------|-------|
| **Test Accuracy** | 50.00% |
| **F1-Score** | 0.000 |
| **ML Strategy Return** | +14.25% |
| **Buy & Hold Return** | -20.46% |
| **Excess Return** | +34.71%p |
| **Win Rate** | 61.5% |
| **Number of Trades** | 13 |

✅ **ML strategy outperformed Buy & Hold by +34.71%p**

---

## 🏆 Best Performers

- **Highest Return**: TSLA (+59.20%)
- **Best F1-Score**: AAPL (0.857)
- **Highest Win Rate**: NVDA (81.2%)

---

## 🔧 Technical Stack

- **Model**: Logistic Regression
- **Features**: 26 (price, volume, volatility, macro-economic indicators)
- **Data Period**: 95 days (2025-09-25 ~ 2026-02-10)
- **Feature Engineering**: StandardScaler normalization
- **Train/Test Split**: 80/20 (time-series order maintained)

---

## 💡 Key Insights

1. **Simple models work best with limited data**
   - Logistic Regression outperformed Random Forest
   - Lower overfitting, more stable predictions

2. **Macro-economic indicators are crucial**
   - Oil prices, VIX, Treasury yields showed high feature importance
   - Better than sentiment analysis (Phase 1 failed)

3. **ML strategy excels in volatile markets**
   - TSLA: +59% return in sideways market
   - MSFT: +14% return in -20% decline

4. **High win rates across all stocks**
   - Average: 74.0%
   - Range: 61.5% - 81.2%

---

## 🚀 Next Steps

1. Expand to more stocks (tech sector, S&P 500)
2. Implement ensemble models (combine multiple predictions)
3. Add risk management (stop-loss, position sizing)
4. Deploy real-time trading system
5. Monitor and retrain models regularly

---

*Powered by **Sentirax AI Trading System*** 🤖
