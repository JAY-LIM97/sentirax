# Sentirax - Oracle Cloud 무료 배포 가이드

## 왜 Oracle Cloud?
- **영구 무료** (Always Free Tier)
- ARM VM: 최대 4코어, 24GB RAM
- 200GB 스토리지
- PC 안 켜도 매일 밤 자동 실행

---

## STEP 1: Oracle Cloud 계정 생성

1. https://cloud.oracle.com 접속
2. **Start for free** 클릭
3. 회원가입 (신용카드 필요하지만 무료 티어는 과금 안 됨)
4. Home Region 선택: **Japan East (Tokyo)** 또는 **South Korea Central (Chuncheon)** 추천

---

## STEP 2: VM 인스턴스 생성

1. Oracle Cloud Console → **Compute** → **Instances** → **Create Instance**

2. 설정:
   - **Name**: sentirax-bot
   - **Image**: Ubuntu 22.04 (Canonical)
   - **Shape**: VM.Standard.A1.Flex (Ampere ARM)
     - **OCPUs**: 1 (무료 범위)
     - **Memory**: 6 GB (무료 범위)
   - **Network**: 기본 VCN 사용
   - **SSH Key**: 새 키 다운로드 또는 기존 키 업로드

3. **Create** 클릭 → 2~3분 대기

4. Public IP 주소 확인 (예: 150.xxx.xxx.xxx)

> ⚠️ ARM 인스턴스가 용량 부족이면 **VM.Standard.E2.1.Micro** (x86, 1GB RAM) 선택

---

## STEP 3: SSH 접속

```bash
# Windows (PowerShell)
ssh -i C:\path\to\ssh-key.key ubuntu@150.xxx.xxx.xxx

# Mac/Linux
chmod 400 ~/ssh-key.key
ssh -i ~/ssh-key.key ubuntu@150.xxx.xxx.xxx
```

---

## STEP 4: 자동 설치

서버 접속 후 아래 명령어 실행:

```bash
# 프로젝트 클론
git clone https://github.com/JAY-LIM97/sentirax.git ~/sentirax
cd ~/sentirax

# 설치 스크립트 실행
chmod +x cloud/setup_oracle.sh
./cloud/setup_oracle.sh
```

이 스크립트가 자동으로:
- Python 3.11 설치
- 가상환경 + 패키지 설치
- Cron 스케줄 설정

---

## STEP 5: .env 설정

```bash
nano ~/sentirax/.env
```

아래 내용 입력 (로컬 PC의 .env 내용 복사):

```
HT_API_FK_KEY=PSg5opYg...
HT_API_FK_SECRET_KEY=twclzFEZ...
HT_API_KEY=PSRZzPgv...
HT_API_SECRET_KEY=z85qLDOK...
```

저장: `Ctrl+O` → `Enter` → `Ctrl+X`

---

## STEP 6: 테스트

```bash
cd ~/sentirax
source venv/bin/activate

# 스윙 트레이딩 테스트
python cloud/run_cloud.py --swing

# 스캘핑 테스트
python cloud/run_cloud.py --scalping

# 둘 다 실행
python cloud/run_cloud.py --all
```

---

## STEP 7: Cron 확인

```bash
# 현재 설정된 cron 확인
crontab -l
```

결과:
```
# 모델 갱신: 매일 UTC 14:20 (KST 23:20)
20 14 * * 1-5 cd $HOME/sentirax && source venv/bin/activate && python cloud/run_cloud.py --refresh >> logs/cron_refresh.log 2>&1

# 스윙 트레이딩: 매일 UTC 14:30 (KST 23:30)
30 14 * * 1-5 cd $HOME/sentirax && source venv/bin/activate && python cloud/run_cloud.py --swing >> logs/cron_swing.log 2>&1

# 스캘핑 (2시간 연속): 매일 UTC 14:35 (KST 23:35)
35 14 * * 1-5 cd $HOME/sentirax && source venv/bin/activate && python cloud/run_cloud.py --scalping-continuous >> logs/cron_scalping.log 2>&1
```

---

## 자동 실행 스케줄 (한국시간 기준)

| 시간 (KST) | 작업 | 설명 |
|------------|------|------|
| 23:20 | Model Refresh | 급등주 스캔 + 스캘핑 모델 갱신 |
| 23:30 | Swing Trading | TOP 20 종목 매매 신호 예측 + 주문 |
| 23:35 | Scalping Start | 2시간 연속 스캘핑 (01:35까지) |

- 월~금만 실행 (미국 주식시장 영업일)
- 미국 장 시작 = 한국 23:30 (겨울) / 22:30 (여름)

---

## 관리 명령어

```bash
# 로그 확인
tail -f ~/sentirax/logs/cron_swing.log
tail -f ~/sentirax/logs/cron_scalping.log

# 최신 코드 업데이트
cd ~/sentirax && git pull origin main

# 패키지 업데이트
source venv/bin/activate && pip install -r cloud/requirements-cloud.txt

# cron 수정
crontab -e

# 수동 실행
cd ~/sentirax && source venv/bin/activate && python cloud/run_cloud.py --all

# 실행 중인 프로세스 확인
ps aux | grep sentirax
```

---

## 문제 해결

### VM이 안 만들어짐 (Out of capacity)
→ 다른 리전 시도하거나 E2.1.Micro (x86) 선택

### cron이 안 돌아감
```bash
# cron 서비스 확인
sudo systemctl status cron

# cron 로그 확인
grep CRON /var/log/syslog | tail -20
```

### 패키지 설치 오류
```bash
# ARM(aarch64)에서 sklearn 설치 오류 시
sudo apt-get install -y gfortran libopenblas-dev
pip install scikit-learn
```

### 시간대 확인
```bash
date         # 서버 현재 시간
timedatectl  # 시간대 설정 확인
# UTC로 설정되어 있어야 cron 시간이 맞음
```

---

## 비용
- **$0/월** (Always Free Tier)
- 신용카드 등록하지만 과금 없음
- 무료 초과 시 자동 정지 (과금 X)
