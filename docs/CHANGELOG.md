# Changelog

All notable changes to Sentirax will be documented in this file.

## [Unreleased]
### Planned
- Backtesting system with 90-day historical data
- Trading simulation and ROI calculation
- Data visualization (charts)
- Database integration (PostgreSQL)
- Web dashboard (Streamlit)

## [0.1.0] - 2026-02-08
### Added
- Multi-LLM sentiment analysis (Gemini, Claude, Groq)
- NewsAPI integration for real-time news collection
- Batch processing for 20+ news articles
- Investment recommendation engine (BUY/SELL/HOLD)
- Configurable thresholds and analysis periods
- yfinance integration for stock prices

### Fixed
- Gemini API model 404 errors via auto-detection
- JSON parsing failures with manual fallback
- Token limit issues by increasing to 8000

### Security
- API key protection using .env
- Added .gitignore for sensitive data
- Set repository to private

### Technical Details
- **Language**: Python 3.14
- **Lines of Code**: ~500
- **Commits**: 5
- **Development Time**: 8 hours
- **Test Coverage**: Manual testing

---

**Full Changelog**: https://github.com/JAY-LIM97/sentirax/commits/v0.1.0