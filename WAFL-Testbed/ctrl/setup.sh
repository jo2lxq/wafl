#!/bin/bash

# ==========================================
# WAFL-Testbed Node Setup Script
# ==========================================
# 説明: execution_config.json から設定を読み込み、
#       全実行ノードのセットアップを一括で行う
#
# 実行内容:
#   1. Sudo 権限でパスワード不要設定
#   2. 必要なパッケージのインストール
#   3. ホスト設定（Chrony, Docker, Kernel）
#   4. Docker イメージのビルドと配布
# ==========================================

set -e  # エラー時に即座に終了

# ==========================================
# 設定
# ==========================================
CONFIG_FILE="ctrl/execution_config.json"
NTP_SERVER_IP="192.168.11.10"  # NTP サーバーの IP（必要に応じて変更）
IMAGE_NAME="wafl-node"
IMAGE_TAG="v1.0"
FULL_IMAGE_NAME="${IMAGE_NAME}:${IMAGE_TAG}"

# 色の定義
RED='\033[0;31m'
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# ==========================================
# 前提条件チェック
# ==========================================
echo -e "${BLUE}=========================================${NC}"
echo -e "${BLUE}WAFL-Testbed Node Setup${NC}"
echo -e "${BLUE}=========================================${NC}"
echo ""

# jq の確認
if ! command -v jq &> /dev/null; then
    echo -e "${RED}Error: jq is not installed. Please install jq first.${NC}"
    echo "  sudo apt-get install -y jq"
    exit 1
fi

# sshpass の確認
if ! command -v sshpass &> /dev/null; then
    echo -e "${RED}Error: sshpass is not installed. Please install sshpass first.${NC}"
    echo "  sudo apt-get install -y sshpass"
    exit 1
fi

# execution_config.json の存在確認
if [ ! -f "$CONFIG_FILE" ]; then
    echo -e "${RED}Error: $CONFIG_FILE not found.${NC}"
    exit 1
fi

# ==========================================
# JSON から設定を読み込み
# ==========================================
REMOTE_USER=$(jq -r '.user' "$CONFIG_FILE")
HOSTS=($(jq -r '.nodes[].physical_ip' "$CONFIG_FILE" | sort -u))

echo -e "${CYAN}Configuration:${NC}"
echo "  Config file: $CONFIG_FILE"
echo "  Remote user: $REMOTE_USER"
echo "  Unique hosts: ${HOSTS[@]}"
echo "  NTP server: $NTP_SERVER_IP"
echo "  Docker image: $FULL_IMAGE_NAME"
echo ""

# ==========================================
# パスワード入力
# ==========================================
echo -e "${BLUE}🔑 Enter SSH password for remote nodes:${NC}"
read -s SSH_PASSWORD
echo ""

echo -e "${BLUE}🔑 Enter sudo password for remote nodes:${NC}"
read -s SUDO_PASSWORD
echo ""

# ==========================================
# Phase 1: Sudo 権限設定
# ==========================================
echo -e "${CYAN}=========================================${NC}"
echo -e "${CYAN}Phase 1: Configuring passwordless sudo${NC}"
echo -e "${CYAN}=========================================${NC}"
echo ""

for HOST in "${HOSTS[@]}"; do
    echo -e "${BLUE}⏳ Processing ${HOST} ...${NC}"

    sshpass -p "$SSH_PASSWORD" ssh -n -o StrictHostKeyChecking=no $REMOTE_USER@$HOST \
    "echo '$SUDO_PASSWORD' | sudo -S sh -c 'echo \"$REMOTE_USER ALL=(ALL) NOPASSWD: ALL\" > /etc/sudoers.d/wafl_nopasswd && chmod 0440 /etc/sudoers.d/wafl_nopasswd'" 2>/dev/null

    if [ $? -eq 0 ]; then
        echo -e "${GREEN}  ✅ Sudo config updated${NC}"
    else
        echo -e "${RED}  ❌ Failed to update sudo config${NC}"
        exit 1
    fi
done

echo ""

# ==========================================
# Phase 2: パッケージインストール
# ==========================================
echo -e "${CYAN}=========================================${NC}"
echo -e "${CYAN}Phase 2: Installing packages${NC}"
echo -e "${CYAN}=========================================${NC}"
echo ""

for HOST in "${HOSTS[@]}"; do
    echo -e "${BLUE}⏳ Processing ${HOST} ...${NC}"

    REMOTE_CMDS="
        export DEBIAN_FRONTEND=noninteractive &&

        echo -e '${YELLOW}📦 [1/2] Updating package lists...${NC}' &&
        sudo apt-get update &&

        echo -e '${YELLOW}📥 [2/2] Installing required tools...${NC}' &&
        sudo apt-get install -y docker.io docker-buildx chrony sysstat dstat iproute2 bridge-utils jq fwupd power-profiles-daemon rsync &&
        sudo systemctl enable --now chrony &&
        sudo systemctl enable --now docker
    "

    sshpass -p "$SSH_PASSWORD" ssh -n -o StrictHostKeyChecking=no $REMOTE_USER@$HOST "$REMOTE_CMDS"

    if [ $? -eq 0 ]; then
        echo -e "${GREEN}  ✅ Package installation completed${NC}"
    else
        echo -e "${RED}  ❌ Package installation failed${NC}"
        exit 1
    fi

    echo "---"
done

echo ""

# ==========================================
# Phase 3: ホスト設定
# ==========================================
echo -e "${CYAN}=========================================${NC}"
echo -e "${CYAN}Phase 3: Configuring hosts${NC}"
echo -e "${CYAN}=========================================${NC}"
echo ""

for HOST in "${HOSTS[@]}"; do
    echo -e "${BLUE}⏳ Configuring ${HOST} ...${NC}"

    REMOTE_CMDS="
        echo -e '${YELLOW}⏰ [1/3] Configuring Chrony...${NC}'
        sudo cp /etc/chrony/chrony.conf /etc/chrony/chrony.conf.bak 2>/dev/null || true
        echo 'server ${NTP_SERVER_IP} iburst' | sudo tee -a /etc/chrony/chrony.conf > /dev/null
        sudo systemctl restart chrony

        echo -e '${YELLOW}🐳 [2/3] Setting Docker permissions...${NC}'
        sudo usermod -aG docker ${REMOTE_USER}

        echo -e '${YELLOW}⚙️  [3/3] Tuning Kernel parameters...${NC}'
        cat <<EOF | sudo tee /etc/sysctl.d/99-wafl-tuning.conf > /dev/null
# WAFL Experiment Tuning
net.ipv4.ip_forward=1
fs.file-max=100000
net.core.somaxconn=4096
net.core.rmem_max=16777216
net.core.wmem_max=16777216
EOF
        sudo sysctl -p /etc/sysctl.d/99-wafl-tuning.conf > /dev/null
    "

    sshpass -p "$SSH_PASSWORD" ssh -n -o StrictHostKeyChecking=no $REMOTE_USER@$HOST "$REMOTE_CMDS"

    if [ $? -eq 0 ]; then
        echo -e "${GREEN}  ✅ Configuration applied${NC}"
    else
        echo -e "${RED}  ❌ Configuration failed${NC}"
        exit 1
    fi

    echo "---"
done

echo ""

# ==========================================
# 完了
# ==========================================
echo -e "${GREEN}=========================================${NC}"
echo -e "${GREEN}✅ All setup tasks completed successfully!${NC}"
echo -e "${GREEN}=========================================${NC}"
echo ""

