from optimized_collector import OptimizedCollector

print("""
╔══════════════════════════════════════════════════════════╗
║                                                          ║
║          🚀 Sentirax Data Collection v2.0 🚀            ║
║                                                          ║
║   최적화된 데이터 수집 시스템                              ║
║   • 뉴스: 최근 2주 (API 절약)                            ║
║   • 거시경제: 90일 (풍부한 데이터)                        ║
║   • 기술지표: 90일 (고급 분석)                           ║
║                                                          ║
╚══════════════════════════════════════════════════════════╝
""")

# 수집기 생성
collector = OptimizedCollector()

# 데이터 수집
df = collector.collect_optimized_data(
    symbol="TSLA",
    company_name="Tesla",
    full_days=90,      # 전체 분석 90일
    news_days=14       # 뉴스는 최근 2주만
)

# 결과 확인
print("\n" + "="*60)
print("📊 수집 결과 요약")
print("="*60)

print(f"\n📅 기간: {df.index[0]} ~ {df.index[-1]} ({len(df)}일)")
print(f"\n📋 특징 개수: {len(df.columns)}개")

print("\n📈 주요 통계:")
stats_cols = ['Close', 'sentiment_score', 'vix', 'rsi', 'next_day_return']
available_stats = [col for col in stats_cols if col in df.columns]
print(df[available_stats].describe().round(2))

print("\n🔍 데이터 샘플 (최근 5일):")
print(df.tail())

# 저장
collector.save_data(df, 'data/tsla_optimized_90days.csv')

print("\n✅ 모든 작업 완료!")
print("   다음 단계: 백테스팅 & 예측 모델 훈련")