from typing import Dict, List
from config import Config
import json
import re

# LLM Provider에 따라 import
if Config.LLM_PROVIDER == 'claude':
    from anthropic import Anthropic
elif Config.LLM_PROVIDER == 'gemini':
    import google.generativeai as genai
elif Config.LLM_PROVIDER == 'groq':
    from groq import Groq

class SentimentAnalyzer:
    """AI를 활용한 뉴스 감성 분석"""
    
    def __init__(self):
        self.provider = Config.LLM_PROVIDER
        
        if self.provider == 'claude':
            api_key = Config.ANTHROPIC_API_KEY
            if not api_key:
                raise ValueError("❌ ANTHROPIC_API_KEY가 설정되지 않았습니다.")
            self.client = Anthropic(api_key=api_key)
            self.model = "claude-sonnet-4-20250514"
            
        elif self.provider == 'gemini':
            api_key = Config.GEMINI_API_KEY
            if not api_key:
                raise ValueError("❌ GEMINI_API_KEY가 설정되지 않았습니다.")
            genai.configure(api_key=api_key)
            
            # 사용 가능한 모델 자동 탐지
            available_models = []
            for model in genai.list_models():
                if 'generateContent' in model.supported_generation_methods:
                    available_models.append(model.name)
            
            if not available_models:
                raise ValueError("❌ 사용 가능한 Gemini 모델이 없습니다.")
            
            # 첫 번째 사용 가능한 모델 사용
            model_name = available_models[0].replace('models/', '')
            print(f"📌 사용 모델: {model_name}")
            self.model = genai.GenerativeModel(model_name)
            
        elif self.provider == 'groq':
            api_key = Config.GROQ_API_KEY
            if not api_key:
                raise ValueError("❌ GROQ_API_KEY가 설정되지 않았습니다.")
            self.client = Groq(api_key=api_key)
            self.model = "llama-3.3-70b-versatile"
        
        else:
            raise ValueError(f"❌ 지원하지 않는 LLM Provider: {self.provider}")
    
    def analyze_news_sentiment_batch(self, news_text: str, symbol: str, 
                                     news_count: int) -> Dict:
        """
        대량의 뉴스를 배치로 나눠서 분석
        
        Args:
            news_text: 전체 뉴스 텍스트
            symbol: 주식 심볼
            news_count: 총 뉴스 개수
        """
        BATCH_SIZE = 5  # 한 번에 5개씩 분석
        
        if news_count <= BATCH_SIZE:
            # 뉴스가 5개 이하면 한 번에 분석
            return self.analyze_news_sentiment(news_text, symbol)
        
        # 뉴스 분할
        news_list = self._split_news_text(news_text, news_count)
        
        print(f"📦 {news_count}개 뉴스를 {BATCH_SIZE}개씩 배치 분석 시작...\n")
        
        all_scores = []
        batch_num = 0
        
        for i in range(0, news_count, BATCH_SIZE):
            batch_num += 1
            batch = news_list[i:i+BATCH_SIZE]
            batch_text = "\n\n".join(batch)
            
            print(f"🔄 배치 {batch_num}/{(news_count + BATCH_SIZE - 1) // BATCH_SIZE} 분석 중...")
            
            # 배치 분석
            result = self.analyze_news_sentiment(batch_text, symbol)
            
            if result['recommendation'] != 'ERROR':
                # 뉴스 번호 오프셋 조정
                for score_item in result.get('individual_scores', []):
                    score_item['news_number'] += i
                
                all_scores.extend(result.get('individual_scores', []))
        
        # 전체 평균 계산
        if all_scores:
            overall_score = sum(item['score'] for item in all_scores) / len(all_scores)
        else:
            overall_score = 0.0
        
        # 최종 결과
        final_result = {
            'individual_scores': all_scores,
            'overall_score': overall_score,
            'reasoning': f'{news_count}개 뉴스를 {batch_num}개 배치로 분석 완료. 평균 감성 점수: {overall_score:.2f}'
        }
        
        # 추천 결정
        if overall_score >= Config.BULLISH_THRESHOLD:
            final_result['recommendation'] = 'BUY'
        elif overall_score <= Config.BEARISH_THRESHOLD:
            final_result['recommendation'] = 'SELL'
        else:
            final_result['recommendation'] = 'HOLD'
        
        print(f"✅ 전체 분석 완료!\n")
        
        return final_result
    
    def _split_news_text(self, news_text: str, news_count: int) -> List[str]:
        """뉴스 텍스트를 개별 뉴스로 분할"""
        lines = news_text.split('\n')
        news_list = []
        current_news = []
        
        for line in lines:
            # 뉴스 시작 패턴 감지 (숫자. 로 시작)
            if re.match(r'^\d+\.', line.strip()):
                if current_news:
                    news_list.append('\n'.join(current_news))
                current_news = [line]
            else:
                current_news.append(line)
        
        if current_news:
            news_list.append('\n'.join(current_news))
        
        return news_list
    
    def analyze_news_sentiment(self, news_text: str, symbol: str) -> Dict:
        """뉴스 감성 분석 실행 (단일 배치)"""
        
        prompt = f"""당신은 전문 금융 애널리스트입니다. 다음 {symbol} 주식 관련 뉴스들을 분석하고, 각 뉴스가 주가에 미칠 영향을 점수화해주세요.

{news_text}

**분석 요구사항:**
1. 각 뉴스에 대해 -1.0(매우 부정) ~ +1.0(매우 긍정) 사이의 점수를 부여
2. 전체 평균 점수 계산
3. 핵심 근거 요약 (짧게, 한 문장)

**중요: 반드시 아래 JSON 형식으로만 응답하세요. reason은 20단어 이내로.**

{{
    "individual_scores": [
        {{"news_number": 1, "score": 0.5, "reason": "짧은 이유"}},
        {{"news_number": 2, "score": -0.3, "reason": "짧은 이유"}}
    ],
    "overall_score": 0.1,
    "reasoning": "종합 분석 근거 한 문장"
}}"""

        try:
            if self.provider == 'claude':
                response_text = self._analyze_with_claude(prompt)
            elif self.provider == 'gemini':
                response_text = self._analyze_with_gemini(prompt)
            elif self.provider == 'groq':
                response_text = self._analyze_with_groq(prompt)
            
            # JSON 파싱
            result = self._parse_json_response(response_text)
            
            # 추천 결정
            score = result['overall_score']
            if score >= Config.BULLISH_THRESHOLD:
                result['recommendation'] = 'BUY'
            elif score <= Config.BEARISH_THRESHOLD:
                result['recommendation'] = 'SELL'
            else:
                result['recommendation'] = 'HOLD'
            
            return result
            
        except Exception as e:
            print(f"❌ 분석 실패: {e}")
            return {
                'overall_score': 0.0,
                'recommendation': 'ERROR',
                'individual_scores': [],
                'reasoning': f'분석 중 오류 발생: {str(e)}'
            }
    
    def _analyze_with_claude(self, prompt: str) -> str:
        """Claude API 호출"""
        message = self.client.messages.create(
            model=self.model,
            max_tokens=4000,
            messages=[{"role": "user", "content": prompt}]
        )
        return message.content[0].text
    
    def _analyze_with_gemini(self, prompt: str) -> str:
        """Gemini API 호출"""
        response = self.model.generate_content(
            prompt,
            generation_config={
                'temperature': 0.5,
                'max_output_tokens': 8000,  # 증가!
            }
        )
        return response.text
    
    def _analyze_with_groq(self, prompt: str) -> str:
        """Groq API 호출"""
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": "당신은 전문 금융 애널리스트입니다. JSON 형식으로만 응답하세요."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.5,
            max_tokens=4000
        )
        return response.choices[0].message.content
    
    def _parse_json_response(self, response_text: str) -> Dict:
        """JSON 응답 파싱 (강력한 에러 핸들링)"""
        
        # 디버깅 출력 제거 (배치 분석시 너무 많음)
        # print(f"\n🔍 AI 응답 미리보기:\n{'-'*60}\n{response_text[:300]}...\n{'-'*60}\n")
        
        try:
            # 1단계: JSON 블록 추출
            if "```json" in response_text:
                response_text = response_text.split("```json")[1].split("```")[0]
            elif "```" in response_text:
                response_text = response_text.split("```")[1].split("```")[0]
            
            # 2단계: 문자열 정리
            response_text = response_text.strip()
            
            # 3단계: JSON 파싱 시도
            result = json.loads(response_text)
            
        except json.JSONDecodeError as e:
            # 4단계: 수동 파싱 시도
            try:
                result = self._manual_parse(response_text)
            except Exception as manual_error:
                raise e
        
        # 필수 필드 확인 및 기본값 설정
        if 'overall_score' not in result:
            result['overall_score'] = 0.0
        if 'individual_scores' not in result:
            result['individual_scores'] = []
        if 'reasoning' not in result:
            result['reasoning'] = '분석 완료'
        
        return result
    
    def _manual_parse(self, text: str) -> Dict:
        """수동 JSON 파싱 (비상용)"""
        result = {
            'individual_scores': [],
            'overall_score': 0.0,
            'reasoning': ''
        }
        
        # overall_score 추출
        score_match = re.search(r'"overall_score":\s*([-\d.]+)', text)
        if score_match:
            result['overall_score'] = float(score_match.group(1))
        
        # reasoning 추출
        reasoning_match = re.search(r'"reasoning":\s*"([^"\\]*(\\.[^"\\]*)*)"', text)
        if reasoning_match:
            result['reasoning'] = reasoning_match.group(1)
        
        # individual_scores 추출
        scores_section = re.search(r'"individual_scores":\s*\[(.*?)\]', text, re.DOTALL)
        if scores_section:
            items = re.findall(
                r'\{[^}]*"news_number":\s*(\d+)[^}]*"score":\s*([-\d.]+)[^}]*"reason":\s*"([^"]+)"[^}]*\}',
                scores_section.group(1),
                re.DOTALL
            )
            for num, score, reason in items:
                result['individual_scores'].append({
                    'news_number': int(num),
                    'score': float(score),
                    'reason': reason
                })
        
        return result
    
    def format_analysis_report(self, analysis: Dict, symbol: str) -> str:
        """분석 결과 포맷팅"""
        
        report = f"\n{'='*60}\n"
        report += f"🎯 Sentirax 분석 리포트: {symbol}\n"
        report += f"🤖 분석 엔진: {self.provider.upper()}\n"
        report += f"{'='*60}\n\n"
        
        score = analysis['overall_score']
        recommendation = analysis['recommendation']
        
        # 이모지 선택
        if recommendation == 'BUY':
            emoji = "📈"
            action_kr = "매수"
        elif recommendation == 'SELL':
            emoji = "📉"
            action_kr = "매도"
        elif recommendation == 'ERROR':
            emoji = "❌"
            action_kr = "오류"
        else:
            emoji = "⏸️"
            action_kr = "관망"
        
        report += f"{emoji} 종합 감성 점수: {score:.2f} / 1.0\n"
        report += f"💡 투자 추천: {action_kr} ({recommendation})\n"
        report += f"📊 분석된 뉴스: {len(analysis.get('individual_scores', []))}개\n\n"
        
        # 개별 뉴스 점수 (상위 10개만 표시)
        if analysis.get('individual_scores'):
            report += "📰 주요 뉴스 분석 (상위 10개):\n"
            report += "-" * 60 + "\n"
            
            # 점수 절댓값 기준으로 정렬
            sorted_scores = sorted(
                analysis['individual_scores'],
                key=lambda x: abs(x['score']),
                reverse=True
            )[:10]
            
            for item in sorted_scores:
                news_num = item['news_number']
                news_score = item['score']
                reason = item['reason']
                
                sentiment_icon = "🟢" if news_score > 0 else "🔴" if news_score < 0 else "⚪"
                
                report += f"{sentiment_icon} 뉴스 {news_num}: {news_score:+.2f}\n"
                report += f"   → {reason}\n\n"
        
        # 종합 근거
        report += "🧠 분석 근거:\n"
        report += "-" * 60 + "\n"
        report += f"{analysis.get('reasoning', 'N/A')}\n"
        
        report += f"\n{'='*60}\n"
        
        return report