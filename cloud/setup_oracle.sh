#!/bin/bash
# =============================================================================
# Sentirax - Oracle Cloud Free Tier 자동 설치 스크립트 (Oracle Linux 9 전용)
# =============================================================================

set -e

echo "=============================================="
echo " Sentirax - Oracle Cloud Setup (Oracle Linux 9)"
echo "=============================================="

# 1. 시스템 업데이트 (dnf 사용)
echo ""
echo "[1/6] System update..."
sudo dnf update -y

# 2. Python 3.11 및 관련 도구 설치
echo ""
echo "[2/6] Installing Python 3.11..."
# 오라클 리눅스 9에서는 python3.11 패키지를 직접 제공합니다.
sudo dnf install -y python3.11 python3.11-pip python3.11-devel git

# 3. 프로젝트 클론
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
mkdir -p logs  # 로그 폴더 생성

# 4. 가상환경 + 패키지 설치
echo ""
echo "[4/6] Setting up Python virtual environment..."
# 오라클 리눅스에서는 python3.11 명령어를 명시적으로 사용합니다.
python3.11 -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install -r cloud/requirements-cloud.txt

# 5. .env 파일 체크 (기존과 동일)
echo ""
echo "[5/6] Checking .env file..."
if [ ! -f "$PROJECT_DIR/.env" ]; then
    echo "  .env file not found, creating template..."
    cat > "$PROJECT_DIR/.env" << 'ENV'
HT_API_KEY=your_real_api_key
HT_API_SECRET_KEY=your_real_api_secret
HT_API_FK_KEY=your_paper_api_key
HT_API_FK_SECRET_KEY=your_paper_api_secret
ENV
    echo "  Created .env template. Please edit it with: nano $PROJECT_DIR/.env"
else
    echo "  .env file found!"
fi

# 6. Cron 설정
echo ""
echo "[6/6] Setting up cron jobs..."
crontab -l 2>/dev/null | grep -v "sentirax" > /tmp/crontab_clean || true

cat >> /tmp/crontab_clean << 'CRON'
# === Sentirax Trading Bot ===
20 14 * * 1-5 cd $HOME/sentirax && source venv/bin/activate && python cloud/run_cloud.py --refresh >> logs/cron_refresh.log 2>&1
30 14 * * 1-5 cd $HOME/sentirax && source venv/bin/activate && python cloud/run_cloud.py --swing >> logs/cron_swing.log 2>&1
35 14 * * 1-5 cd $HOME/sentirax && source venv/bin/activate && python cloud/run_cloud.py --scalping-continuous >> logs/cron_scalping.log 2>&1
# === End Sentirax ===
CRON

crontab /tmp/crontab_clean
rm /tmp/crontab_clean

echo ""
echo "=============================================="
echo " Setup Complete!"
echo "=============================================="
echo " Manual test: cd $PROJECT_DIR && source venv/bin/activate && python cloud/run_cloud.py --all"