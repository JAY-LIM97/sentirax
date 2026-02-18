"""
한국투자증권 API 래퍼 (Sentirax용)

🎯 목적:
- ML 예측 결과를 실제 매매로 연결
- 모의투자로 안전하게 테스트
- 해외주식 전용 (미국 주식)

📊 주요 기능:
1. 인증 (토큰 발급)
2. 잔고 조회
3. 매수/매도 주문
4. 주문 체결 확인
"""

import os
import json
import requests
from datetime import datetime
from typing import Optional, Dict, Any
import pickle


class KISAPIError(Exception):
    """KIS API 호출 에러"""
    pass


class KISTradingAPI:
    """
    한국투자증권 해외주식 거래 API 래퍼

    환경 변수에서 API 키를 읽어 사용
    """

    def __init__(self, paper_trading: bool = True):
        """
        Args:
            paper_trading: True면 모의투자, False면 실전투자
        """
        self.paper_trading = paper_trading

        # API 키 로드
        self.app_key, self.app_secret = self._load_api_keys()

        # 도메인 설정
        if paper_trading:
            self.base_url = "https://openapivts.koreainvestment.com:29443"
        else:
            self.base_url = "https://openapi.koreainvestment.com:9443"

        # 계좌번호 (나중에 설정 필요)
        self.account_no = None
        self.account_code = "01"  # 종합계좌

        # 토큰
        self.access_token = None
        self.token_expired = None

        print(f"✅ KIS API 초기화 완료")
        print(f"  - 모드: {'모의투자' if paper_trading else '실전투자'}")
        print(f"  - URL: {self.base_url}")

    def _load_api_keys(self) -> tuple[str, str]:
        """
        .env 파일에서 API 키 로드

        모의투자/실전투자에 따라 다른 키 사용:
        - 모의투자: HT_API_FK_KEY, HT_API_FK_SECRET_KEY
        - 실전투자: HT_API_KEY, HT_API_SECRET_KEY

        Returns:
            (app_key, app_secret)
        """
        env_path = os.path.join(os.path.dirname(__file__), '..', '.env')

        if not os.path.exists(env_path):
            raise FileNotFoundError(f".env 파일을 찾을 수 없습니다: {env_path}")

        # 모의투자/실전투자에 따라 다른 키 이름 사용
        if self.paper_trading:
            key_name = 'HT_API_FK_KEY'
            secret_name = 'HT_API_FK_SECRET_KEY'
        else:
            key_name = 'HT_API_KEY'
            secret_name = 'HT_API_SECRET_KEY'

        app_key = None
        app_secret = None

        with open(env_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line.startswith(f'{key_name}='):
                    app_key = line.split('=', 1)[1].strip()
                elif line.startswith(f'{secret_name}='):
                    app_secret = line.split('=', 1)[1].strip()

        if not app_key or not app_secret:
            raise ValueError(f"API 키를 찾을 수 없습니다. .env 파일에서 {key_name}, {secret_name}를 확인하세요.")

        print(f"  - 사용 키: {key_name[:15]}...")  # 보안을 위해 일부만 표시

        return app_key, app_secret

    def authenticate(self) -> bool:
        """
        접근 토큰 발급

        Returns:
            성공 여부
        """
        print("\n🔐 토큰 발급 중...")

        url = f"{self.base_url}/oauth2/tokenP"

        headers = {
            "Content-Type": "application/json",
            "Accept": "text/plain",
            "charset": "UTF-8"
        }

        body = {
            "grant_type": "client_credentials",
            "appkey": self.app_key,
            "appsecret": self.app_secret
        }

        try:
            response = requests.post(url, headers=headers, data=json.dumps(body))

            if response.status_code == 200:
                data = response.json()
                self.access_token = data['access_token']
                self.token_expired = data['access_token_token_expired']

                print(f"✅ 토큰 발급 성공")
                print(f"  - 만료: {self.token_expired}")

                return True
            else:
                print(f"❌ 토큰 발급 실패: {response.status_code}")
                print(f"  - {response.text}")
                return False

        except Exception as e:
            print(f"❌ 토큰 발급 오류: {e}")
            return False

    # 거래소별 NYSE 종목 (기본값 NASD, 아래 목록은 NYSE)
    _NYSE_TICKERS = {
        'JPM', 'XOM', 'JNJ', 'WMT', 'PG', 'HD', 'BAC', 'CVX', 'ABBV', 'LLY',
        'MRK', 'KO', 'TMO', 'ABT', 'CRM', 'MCD', 'ACN', 'DHR', 'NEE', 'PM',
        'UNP', 'RTX', 'LOW', 'SLB', 'OXY', 'MPC', 'PSX', 'VLO', 'GS', 'MS',
        'C', 'WFC', 'AXP', 'NKE', 'TGT', 'DG', 'CMG', 'YUM', 'PINS', 'S',
        'CEG', 'BILL', 'PEP',
    }

    def _get_excg_cd(self, ticker: str) -> str:
        """종목 티커로 KIS 해외거래소 코드 반환 (NASD / NYSE)"""
        return 'NYSE' if ticker.upper() in self._NYSE_TICKERS else 'NASD'

    def set_account(self, account_no: str, account_code: str = "01"):
        """
        거래 계좌 설정

        Args:
            account_no: 계좌번호 앞 8자리
            account_code: 계좌 코드 (01: 종합계좌)
        """
        self.account_no = account_no
        self.account_code = account_code

        print(f"✅ 계좌 설정: {account_no}-{account_code}")

    def _get_headers(self, tr_id: str, additional: Optional[Dict[str, str]] = None) -> Dict[str, str]:
        """
        API 호출용 헤더 생성

        Args:
            tr_id: 거래 ID
            additional: 추가 헤더

        Returns:
            헤더 딕셔너리
        """
        if not self.access_token:
            raise KISAPIError("토큰이 없습니다. authenticate()를 먼저 호출하세요.")

        # 모의투자일 경우 TR ID 변환 (T/J/C -> V)
        if self.paper_trading and tr_id[0] in ('T', 'J', 'C'):
            tr_id = 'V' + tr_id[1:]

        headers = {
            "Content-Type": "application/json",
            "authorization": f"Bearer {self.access_token}",
            "appkey": self.app_key,
            "appsecret": self.app_secret,
            "tr_id": tr_id,
            "custtype": "P"  # 개인
        }

        if additional:
            headers.update(additional)

        return headers

    def get_balance(self) -> Optional[Dict[str, Any]]:
        """
        해외주식 잔고 조회

        Returns:
            잔고 정보 딕셔너리
        """
        if not self.account_no:
            raise KISAPIError("계좌번호가 설정되지 않았습니다.")

        print("\n💰 잔고 조회 중...")

        url = f"{self.base_url}/uapi/overseas-stock/v1/trading/inquire-balance"

        # 미국 주식 잔고 조회
        tr_id = "TTTS3012R"  # 실전: TTTS3012R, 모의: VTTS3012R

        headers = self._get_headers(tr_id)

        params = {
            "CANO": self.account_no,
            "ACNT_PRDT_CD": self.account_code,
            "OVRS_EXCG_CD": "NASD",  # 나스닥
            "TR_CRCY_CD": "USD",  # 달러
            "CTX_AREA_FK200": "",
            "CTX_AREA_NK200": ""
        }

        try:
            response = requests.get(url, headers=headers, params=params)

            if response.status_code == 200:
                data = response.json()

                if data.get('rt_cd') == '0':
                    print(f"✅ 잔고 조회 성공")
                    return data
                else:
                    print(f"❌ 잔고 조회 실패: {data.get('msg1')}")
                    return None
            else:
                print(f"❌ API 호출 실패: {response.status_code}")
                return None

        except Exception as e:
            print(f"❌ 잔고 조회 오류: {e}")
            return None

    def get_account_summary(self) -> Optional[Dict[str, float]]:
        """
        해외주식 계좌 요약 조회 (USD 기준)

        Returns:
            {'total_usd': 전체자산(USD), 'available_usd': 인출가능금액(USD)}
            실패 시 None
        """
        data = self.get_balance()
        if not data:
            return None

        output2 = data.get('output2', [])
        if not output2:
            return None

        o2 = output2[0] if isinstance(output2, list) else output2

        try:
            # 외화 평가금액 (보유주식 포함 전체 USD 자산)
            total_usd = float(o2.get('frcr_evlu_amt2') or 0)
            # 해외인출가능금액 (매수에 사용 가능한 USD)
            avail_usd = float(o2.get('ovrs_drwg_psbl_amt') or 0)

            if total_usd > 0 or avail_usd > 0:
                return {
                    'total_usd': max(total_usd, avail_usd),
                    'available_usd': avail_usd
                }
        except (ValueError, TypeError):
            pass

        return None

    def order_buy(self, ticker: str, quantity: int, price: float = 0) -> Optional[Dict[str, Any]]:
        """
        해외주식 매수 주문

        Args:
            ticker: 종목 티커 (예: TSLA)
            quantity: 수량
            price: 가격 (0이면 시장가)

        Returns:
            주문 결과
        """
        if not self.account_no:
            raise KISAPIError("계좌번호가 설정되지 않았습니다.")

        # 미국주식 주문: ORD_DVSN "00"=지정가만 유효 (모의/실전 공통)
        # 시장가(01) 없음 → 현재가로 지정가 주문 → 즉시 체결
        if price <= 0:
            print(f"❌ 매수 주문 실패: 현재가를 price로 전달해야 합니다 (price={price})")
            return None

        excg_cd = self._get_excg_cd(ticker)

        print(f"\n📈 매수 주문: {ticker}")
        print(f"  - 수량: {quantity}주")
        print(f"  - 가격: 지정가 ${price:.2f} / 거래소: {excg_cd}")

        url = f"{self.base_url}/uapi/overseas-stock/v1/trading/order"

        tr_id = "TTTT1002U"  # 실전: TTTT1002U, 모의: VTTT1002U

        headers = self._get_headers(tr_id)

        body = {
            "CANO": self.account_no,
            "ACNT_PRDT_CD": self.account_code,
            "OVRS_EXCG_CD": excg_cd,
            "PDNO": ticker,
            "ORD_QTY": str(quantity),
            "OVRS_ORD_UNPR": f"{price:.2f}",
            "CTAC_TLNO": "",
            "MGCO_APTM_ODNO": "",
            "SLL_TYPE": "",
            "ORD_SVR_DVSN_CD": "0",
            "ORD_DVSN": "00",  # 지정가 (미국주식 유일 옵션, 모의투자도 동일)
        }

        try:
            response = requests.post(url, headers=headers, data=json.dumps(body))

            if response.status_code == 200:
                data = response.json()

                if data.get('rt_cd') == '0':
                    order_no = data.get('output', {}).get('ODNO', 'N/A')
                    print(f"✅ 매수 주문 성공")
                    print(f"  - 주문번호: {order_no}")
                    return data
                else:
                    print(f"❌ 매수 주문 실패: {data.get('msg1')}")
                    return None
            else:
                print(f"❌ API 호출 실패: {response.status_code}")
                print(f"  - {response.text}")
                return None

        except Exception as e:
            print(f"❌ 매수 주문 오류: {e}")
            return None

    def order_sell(self, ticker: str, quantity: int, price: float = 0) -> Optional[Dict[str, Any]]:
        """
        해외주식 매도 주문

        Args:
            ticker: 종목 티커 (예: TSLA)
            quantity: 수량
            price: 가격 (0이면 시장가)

        Returns:
            주문 결과
        """
        if not self.account_no:
            raise KISAPIError("계좌번호가 설정되지 않았습니다.")

        # 미국주식 주문: ORD_DVSN "00"=지정가만 유효 (모의/실전 공통)
        if price <= 0:
            print(f"❌ 매도 주문 실패: 현재가를 price로 전달해야 합니다 (price={price})")
            return None

        excg_cd = self._get_excg_cd(ticker)

        print(f"\n📉 매도 주문: {ticker}")
        print(f"  - 수량: {quantity}주")
        print(f"  - 가격: 지정가 ${price:.2f} / 거래소: {excg_cd}")

        url = f"{self.base_url}/uapi/overseas-stock/v1/trading/order"

        tr_id = "TTTT1006U"  # 실전: TTTT1006U, 모의: VTTT1006U

        headers = self._get_headers(tr_id)

        body = {
            "CANO": self.account_no,
            "ACNT_PRDT_CD": self.account_code,
            "OVRS_EXCG_CD": excg_cd,
            "PDNO": ticker,
            "ORD_QTY": str(quantity),
            "OVRS_ORD_UNPR": f"{price:.2f}",
            "CTAC_TLNO": "",
            "MGCO_APTM_ODNO": "",
            "SLL_TYPE": "00",
            "ORD_SVR_DVSN_CD": "0",
            "ORD_DVSN": "00",  # 지정가 (미국주식 유일 옵션, 모의투자도 동일)
        }

        try:
            response = requests.post(url, headers=headers, data=json.dumps(body))

            if response.status_code == 200:
                data = response.json()

                if data.get('rt_cd') == '0':
                    order_no = data.get('output', {}).get('ODNO', 'N/A')
                    print(f"✅ 매도 주문 성공")
                    print(f"  - 주문번호: {order_no}")
                    return data
                else:
                    print(f"❌ 매도 주문 실패: {data.get('msg1')}")
                    return None
            else:
                print(f"❌ API 호출 실패: {response.status_code}")
                print(f"  - {response.text}")
                return None

        except Exception as e:
            print(f"❌ 매도 주문 오류: {e}")
            return None

    # =========================================================
    # 국내주식 주문/잔고 (KRW 기준)
    # TR_ID: 매수 TTTC0012U→VTTC0012U, 매도 TTTC0011U→VTTC0011U
    # =========================================================

    def order_buy_domestic(self, ticker: str, quantity: int, price: int) -> Optional[Dict[str, Any]]:
        """국내주식 현금 매수 (지정가, KRW)"""
        if not self.account_no:
            raise KISAPIError("계좌번호가 설정되지 않았습니다.")
        if price <= 0:
            print(f"❌ [국내] 매수 주문 실패: price={price} (현재가 필요)")
            return None

        print(f"\n📈 [국내] 매수 주문: {ticker} / {quantity}주 / {price:,}원 (지정가)")

        url = f"{self.base_url}/uapi/domestic-stock/v1/trading/order-cash"
        tr_id = "TTTC0012U"  # 실전→VTTC0012U (T→V 자동변환)
        headers = self._get_headers(tr_id)

        body = {
            "CANO": self.account_no,
            "ACNT_PRDT_CD": self.account_code,
            "PDNO": ticker,
            "ORD_DVSN": "00",
            "ORD_QTY": str(quantity),
            "ORD_UNPR": str(price),
            "EXCG_ID_DVSN_CD": "KRX",
            "SLL_TYPE": "",
            "CNDT_PRIC": ""
        }

        try:
            response = requests.post(url, headers=headers, data=json.dumps(body))
            if response.status_code == 200:
                data = response.json()
                if data.get('rt_cd') == '0':
                    order_no = data.get('output', {}).get('ODNO', 'N/A')
                    print(f"✅ [국내] 매수 주문 성공 - 주문번호: {order_no}")
                    return data
                else:
                    print(f"❌ [국내] 매수 주문 실패: {data.get('msg1')}")
                    return None
            else:
                print(f"❌ [국내] API 호출 실패: {response.status_code}")
                return None
        except Exception as e:
            print(f"❌ [국내] 매수 주문 오류: {e}")
            return None

    def order_sell_domestic(self, ticker: str, quantity: int, price: int) -> Optional[Dict[str, Any]]:
        """국내주식 현금 매도 (지정가, KRW)"""
        if not self.account_no:
            raise KISAPIError("계좌번호가 설정되지 않았습니다.")
        if price <= 0:
            print(f"❌ [국내] 매도 주문 실패: price={price} (현재가 필요)")
            return None

        print(f"\n📉 [국내] 매도 주문: {ticker} / {quantity}주 / {price:,}원 (지정가)")

        url = f"{self.base_url}/uapi/domestic-stock/v1/trading/order-cash"
        tr_id = "TTTC0011U"  # 실전→VTTC0011U (T→V 자동변환)
        headers = self._get_headers(tr_id)

        body = {
            "CANO": self.account_no,
            "ACNT_PRDT_CD": self.account_code,
            "PDNO": ticker,
            "ORD_DVSN": "00",
            "ORD_QTY": str(quantity),
            "ORD_UNPR": str(price),
            "EXCG_ID_DVSN_CD": "KRX",
            "SLL_TYPE": "01",
            "CNDT_PRIC": ""
        }

        try:
            response = requests.post(url, headers=headers, data=json.dumps(body))
            if response.status_code == 200:
                data = response.json()
                if data.get('rt_cd') == '0':
                    order_no = data.get('output', {}).get('ODNO', 'N/A')
                    print(f"✅ [국내] 매도 주문 성공 - 주문번호: {order_no}")
                    return data
                else:
                    print(f"❌ [국내] 매도 주문 실패: {data.get('msg1')}")
                    return None
            else:
                print(f"❌ [국내] API 호출 실패: {response.status_code}")
                return None
        except Exception as e:
            print(f"❌ [국내] 매도 주문 오류: {e}")
            return None

    def get_balance_domestic(self) -> Optional[Dict[str, Any]]:
        """국내주식 잔고 조회"""
        if not self.account_no:
            raise KISAPIError("계좌번호가 설정되지 않았습니다.")

        url = f"{self.base_url}/uapi/domestic-stock/v1/trading/inquire-balance"
        tr_id = "TTTC8434R"  # 실전→VTTC8434R (T→V 자동변환)
        headers = self._get_headers(tr_id)

        params = {
            "CANO": self.account_no,
            "ACNT_PRDT_CD": self.account_code,
            "AFHR_FLPR_YN": "N",
            "OFL_YN": "",
            "INQR_DVSN": "02",
            "UNPR_DVSN": "01",
            "FUND_STTL_ICLD_YN": "N",
            "FNCG_AMT_AUTO_RDPT_YN": "N",
            "PRCS_DVSN": "00",
            "CTX_AREA_FK100": "",
            "CTX_AREA_NK100": ""
        }

        try:
            response = requests.get(url, headers=headers, params=params)
            if response.status_code == 200:
                data = response.json()
                if data.get('rt_cd') == '0':
                    return data
                else:
                    print(f"❌ [국내] 잔고 조회 실패: {data.get('msg1')}")
                    return None
            else:
                print(f"❌ [국내] API 호출 실패: {response.status_code}")
                return None
        except Exception as e:
            print(f"❌ [국내] 잔고 조회 오류: {e}")
            return None

    def get_account_summary_domestic(self) -> Optional[Dict[str, float]]:
        """국내주식 계좌 요약 (KRW 기준)"""
        data = self.get_balance_domestic()
        if not data:
            return None

        output2 = data.get('output2', {})
        if not output2:
            return None

        o2 = output2[0] if isinstance(output2, list) else output2

        try:
            available_krw = float(o2.get('dnca_tot_amt') or 0)   # 예수금 총액
            total_krw = float(o2.get('tot_evlu_amt') or 0)        # 총 평가금액
            if total_krw > 0 or available_krw > 0:
                return {
                    'total_krw': max(total_krw, available_krw),
                    'available_krw': available_krw
                }
        except (ValueError, TypeError):
            pass

        return None

    def get_current_price(self, ticker: str) -> Optional[float]:
        """
        해외주식 현재가 조회

        Args:
            ticker: 종목 티커

        Returns:
            현재가 (달러)
        """
        print(f"\n💵 {ticker} 현재가 조회 중...")

        url = f"{self.base_url}/uapi/overseas-price/v1/quotations/price"

        tr_id = "HHDFS00000300"  # 해외주식 현재가 조회

        headers = self._get_headers(tr_id)

        params = {
            "AUTH": "",
            "EXCD": "NAS",  # 나스닥
            "SYMB": ticker
        }

        try:
            response = requests.get(url, headers=headers, params=params)

            if response.status_code == 200:
                data = response.json()

                if data.get('rt_cd') == '0':
                    output = data.get('output', {})
                    last_price = float(output.get('last', 0))

                    print(f"✅ 현재가: ${last_price:.2f}")
                    return last_price
                else:
                    print(f"❌ 현재가 조회 실패: {data.get('msg1')}")
                    return None
            else:
                print(f"❌ API 호출 실패: {response.status_code}")
                return None

        except Exception as e:
            print(f"❌ 현재가 조회 오류: {e}")
            return None


# 사용 예시
if __name__ == "__main__":
    # 모의투자 API 초기화
    api = KISTradingAPI(paper_trading=True)

    # 인증
    if api.authenticate():
        # 계좌번호 설정 (사용자가 입력해야 함)
        # api.set_account("12345678", "01")

        # 현재가 조회
        # price = api.get_current_price("TSLA")

        # 매수 주문 (시장가 1주)
        # result = api.order_buy("TSLA", 1)

        pass
