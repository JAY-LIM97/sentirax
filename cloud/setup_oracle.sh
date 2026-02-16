#!/bin/bash
# =============================================================================
# Sentirax - Oracle Cloud Free Tier 자동 설치 스크립트
#
# 사용법:
#   chmod +x cloud/setup_oracle.sh
#   ./cloud/setup_oracle.sh
#
# Oracle Cloud Always Free VM (Ubuntu 22.04) 기준
# =============================================================================

set -e

echo "=============================================="
echo " Sentirax - Oracle Cloud Setup"
echo "=============================================="

# 1. 시스템 업데이트
echo ""
echo "[1/6] System update..."
sudo apt-get update -y
sudo apt-get upgrade -y

# 2. Python 3.11 설치
echo ""
echo "[2/6] Installing Python 3.11..."
sudo apt-get install -y python3.11 python3.11-venv python3-pip git

# 3. 프로젝트 클론 (없으면)
echo ""
echo "[3/6] Setting up project..."
PROJECT_DIR="$HOME/sentirax"

if [ ! -d "$PROJECT_DIR" ]; then
    echo "  Cloning repository..."
    git clone https://github.com/JAY-LIM97/sentirax.git "$PROJECT_DIR"
else
    echo "  Project exists, pulling latest..."
    cd "$PROJECT_DIR" && git pull origin main
fi

cd "$PROJECT_DIR"

# 4. 가상환경 + 패키지 설치
echo ""
echo "[4/6] Setting up Python virtual environment..."
python3.11 -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install -r cloud/requirements-cloud.txt

# 5. .env 파일 체크
echo ""
echo "[5/6] Checking .env file..."
if [ ! -f "$PROJECT_DIR/.env" ]; then
    echo ""
    echo "=========================================="
    echo " .env 파일이 없습니다!"
    echo " 아래 내용을 $PROJECT_DIR/.env 에 작성하세요:"
    echo "=========================================="
    echo ""
    echo "HT_API_KEY=your_real_api_key"
    echo "HT_API_SECRET_KEY=your_real_api_secret"
    echo "HT_API_FK_KEY=your_paper_api_key"
    echo "HT_API_FK_SECRET_KEY=your_paper_api_secret"
    echo ""
    echo "나중에 설정 가능: nano $PROJECT_DIR/.env"
    echo ""
else
    echo "  .env file found!"
fi

# 6. Cron 설정
echo ""
echo "[6/6] Setting up cron jobs..."

# 기존 sentirax cron 제거
crontab -l 2>/dev/null | grep -v "sentirax" > /tmp/crontab_clean || true

# 새 cron 추가
# KST 23:25 = UTC 14:25 (겨울) / UTC 13:25 (여름)
# 장 시작 5분 전에 모델 갱신 + 장 시작 시 트레이딩
cat >> /tmp/crontab_clean << 'CRON'
# === Sentirax Trading Bot ===
# 모델 갱신: 매일 UTC 14:20 (KST 23:20) - 장 시작 10분 전
20 14 * * 1-5 cd $HOME/sentirax && source venv/bin/activate && python cloud/run_cloud.py --refresh >> logs/cron_refresh.log 2>&1

# 스윙 트레이딩: 매일 UTC 14:30 (KST 23:30) - 장 시작 시
30 14 * * 1-5 cd $HOME/sentirax && source venv/bin/activate && python cloud/run_cloud.py --swing >> logs/cron_swing.log 2>&1

# 스캘핑 (2시간 연속): 매일 UTC 14:35 (KST 23:35) - 장 시작 직후
35 14 * * 1-5 cd $HOME/sentirax && source venv/bin/activate && python cloud/run_cloud.py --scalping-continuous >> logs/cron_scalping.log 2>&1
# === End Sentirax ===
CRON

crontab /tmp/crontab_clean
rm /tmp/crontab_clean

echo ""
echo "=============================================="
echo " Setup Complete!"
echo "=============================================="
echo ""
echo " Cron Schedule (UTC -> KST):"
echo "   14:20 UTC (23:20 KST) - Model refresh"
echo "   14:30 UTC (23:30 KST) - Swing trading"
echo "   14:35 UTC (23:35 KST) - Scalping (2hr continuous)"
echo ""
echo " Verify cron: crontab -l"
echo " Manual test: cd $PROJECT_DIR && source venv/bin/activate && python cloud/run_cloud.py --all"
echo " Logs:        ls $PROJECT_DIR/logs/"
echo ""
echo " IMPORTANT: Set up .env file before first run!"
echo "   nano $PROJECT_DIR/.env"
echo ""
