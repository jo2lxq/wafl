#!/bin/bash

# ==========================================
# WAFL-Testbed Node Setup Script (Parallel Execution)
# ==========================================
# 説明: execution_config.json から設定を読み込み、
#       全実行ノードのセットアップを並列で行う
#
# 実行内容:
#   1. Sudo 権限でパスワード不要設定
#   2. 必要なパッケージのインストール
#   3. ホスト設定（Chrony, Docker, Kernel）
#   4. デプロイ先ディレクトリのクリア
# ==========================================

set -e  # エラー時に即座に終了

# ==========================================
# 設定
# ==========================================
CONFIG_FILE="ctrl/execution_config.json"
NTP_SERVER_IP="192.168.11.10"  # NTP サーバーの IP（必要に応じて変更）
ENV_FILE=".env"  # 環境変数ファイル

# 色の定義
RED='\033[0;31m'
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# 一時ディレクトリ（並列処理の結果を保存）
TEMP_DIR=$(mktemp -d)
trap "rm -rf $TEMP_DIR" EXIT

# ==========================================
# ヘルパー関数
# ==========================================

# 並列ジョブの結果を確認
check_parallel_results() {
    local phase_name="$1"
    local failed=0
    
    for HOST in "${HOSTS[@]}"; do
        if [ -f "$TEMP_DIR/${phase_name}_${HOST}_failed" ]; then
            echo -e "${RED}  ❌ ${HOST}: Failed${NC}"
            if [ -f "$TEMP_DIR/${phase_name}_${HOST}_error" ]; then
                cat "$TEMP_DIR/${phase_name}_${HOST}_error" | head -5
            fi
            failed=1
        else
            echo -e "${GREEN}  ✅ ${HOST}: Success${NC}"
        fi
    done
    
    if [ $failed -eq 1 ]; then
        echo -e "${RED}Error: Phase ${phase_name} failed on one or more hosts.${NC}"
        exit 1
    fi
}

# ==========================================
# 前提条件チェック
# ==========================================
echo -e "${BLUE}=========================================${NC}"
echo -e "${BLUE}WAFL-Testbed Node Setup (Parallel)${NC}"
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

# .env ファイルの存在確認と読み込み
if [ ! -f "$ENV_FILE" ]; then
    echo -e "${RED}Error: $ENV_FILE not found. Please copy .env.sample to .env and configure it.${NC}"
    exit 1
fi

# .env ファイルから Docker Hub 認証情報を読み込み
set -a
source "$ENV_FILE"
set +a

# Docker Hub 認証情報の確認
SKIP_DOCKER_LOGIN=0
if [ -z "$DOCKER_HUB_USERNAME" ] || [ -z "$DOCKER_HUB_PASSWORD" ]; then
    echo -e "${YELLOW}Warning: DOCKER_HUB_USERNAME or DOCKER_HUB_PASSWORD is not set in $ENV_FILE${NC}"
    echo -e "${YELLOW}Phase 4 (Docker Hub login) will be skipped.${NC}"
    SKIP_DOCKER_LOGIN=1
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
echo "  Parallel execution: enabled"
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

# パスワードをエクスポート（サブシェルで使用）
export SSH_PASSWORD
export SUDO_PASSWORD
export REMOTE_USER
export TEMP_DIR

# ==========================================
# Phase 1: Sudo 権限設定（並列）
# ==========================================
echo -e "${CYAN}=========================================${NC}"
echo -e "${CYAN}Phase 1: Configuring passwordless sudo (parallel)${NC}"
echo -e "${CYAN}=========================================${NC}"
echo ""

echo -e "${BLUE}⏳ Configuring sudo on ${#HOSTS[@]} hosts...${NC}"

for HOST in "${HOSTS[@]}"; do
    (
        if sshpass -p "$SSH_PASSWORD" ssh -n -o StrictHostKeyChecking=no $REMOTE_USER@$HOST \
            "echo '$SUDO_PASSWORD' | sudo -S sh -c 'echo \"$REMOTE_USER ALL=(ALL) NOPASSWD: ALL\" > /etc/sudoers.d/wafl_nopasswd && chmod 0440 /etc/sudoers.d/wafl_nopasswd'" 2>"$TEMP_DIR/phase1_${HOST}_error"; then
            rm -f "$TEMP_DIR/phase1_${HOST}_failed"
        else
            touch "$TEMP_DIR/phase1_${HOST}_failed"
        fi
    ) &
done

wait
check_parallel_results "phase1"
echo ""

# ==========================================
# Phase 2: パッケージインストール（並列）
# ==========================================
echo -e "${CYAN}=========================================${NC}"
echo -e "${CYAN}Phase 2: Installing packages (parallel)${NC}"
echo -e "${CYAN}=========================================${NC}"
echo ""

echo -e "${BLUE}⏳ Installing packages on ${#HOSTS[@]} hosts (this may take a while)...${NC}"

for HOST in "${HOSTS[@]}"; do
    (
        REMOTE_CMDS="
            export DEBIAN_FRONTEND=noninteractive &&
            sudo apt-get update &&
            sudo apt-get install -y docker.io docker-buildx chrony sysstat dstat iproute2 bridge-utils jq fwupd power-profiles-daemon rsync iperf3 &&
            sudo systemctl enable --now chrony &&
            sudo systemctl enable --now docker &&
            sudo systemctl enable --now iperf3
        "

        if sshpass -p "$SSH_PASSWORD" ssh -n -o StrictHostKeyChecking=no $REMOTE_USER@$HOST "$REMOTE_CMDS" >"$TEMP_DIR/phase2_${HOST}_output" 2>"$TEMP_DIR/phase2_${HOST}_error"; then
            rm -f "$TEMP_DIR/phase2_${HOST}_failed"
        else
            touch "$TEMP_DIR/phase2_${HOST}_failed"
        fi
    ) &
done

wait
check_parallel_results "phase2"
echo ""

# ==========================================
# Phase 3: ホスト設定（並列）
# ==========================================
echo -e "${CYAN}=========================================${NC}"
echo -e "${CYAN}Phase 3: Configuring hosts (parallel)${NC}"
echo -e "${CYAN}=========================================${NC}"
echo ""

echo -e "${BLUE}⏳ Configuring ${#HOSTS[@]} hosts...${NC}"

for HOST in "${HOSTS[@]}"; do
    (
        REMOTE_CMDS="
            sudo cp /etc/chrony/chrony.conf /etc/chrony/chrony.conf.bak 2>/dev/null || true
            echo 'server ${NTP_SERVER_IP} iburst' | sudo tee -a /etc/chrony/chrony.conf > /dev/null
            sudo systemctl restart chrony

            sudo usermod -aG docker ${REMOTE_USER}
            sudo chmod 666 /var/run/docker.sock

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

        if sshpass -p "$SSH_PASSWORD" ssh -n -o StrictHostKeyChecking=no $REMOTE_USER@$HOST "$REMOTE_CMDS" 2>"$TEMP_DIR/phase3_${HOST}_error"; then
            rm -f "$TEMP_DIR/phase3_${HOST}_failed"
        else
            touch "$TEMP_DIR/phase3_${HOST}_failed"
        fi
    ) &
done

wait
check_parallel_results "phase3"
echo ""

# ==========================================
# Phase 4: デプロイ先ディレクトリのクリア（並列）
# ==========================================
echo -e "${CYAN}=========================================${NC}"
echo -e "${CYAN}Phase 4: Clearing deployment directory (parallel)${NC}"
echo -e "${CYAN}=========================================${NC}"
echo ""

DEPLOYMENT_LOCATION=$(jq -r '.deployment_location' "$CONFIG_FILE")
DEPLOY_DIR="${DEPLOYMENT_LOCATION}/WAFL-Testbed"

echo -e "${BLUE}⏳ Starting cleanup on ${#HOSTS[@]} hosts...${NC}"

for HOST in "${HOSTS[@]}"; do
    (
        DOCKER_CLEANUP="
            docker ps -aq --filter 'name=wafl' | xargs -r docker stop 2>/dev/null || true
            docker ps -aq --filter 'name=wafl' | xargs -r docker rm -f 2>/dev/null || true
            docker rmi -f wafl-node:latest 2>/dev/null || true
            docker image prune -f 2>/dev/null || true
            docker container prune -f 2>/dev/null || true
            sudo rm -rf ${DEPLOY_DIR} && mkdir -p ${DEPLOY_DIR}
        "

        if sshpass -p "$SSH_PASSWORD" ssh -n -o StrictHostKeyChecking=no $REMOTE_USER@$HOST "$DOCKER_CLEANUP" 2>"$TEMP_DIR/phase4_${HOST}_error"; then
            rm -f "$TEMP_DIR/phase4_${HOST}_failed"
        else
            touch "$TEMP_DIR/phase4_${HOST}_failed"
        fi
    ) &
done

wait
check_parallel_results "phase4"
echo ""

# ==========================================
# Phase 5: Docker Hub ログイン（並列）
# ==========================================
if [ $SKIP_DOCKER_LOGIN -eq 0 ]; then
    echo -e "${CYAN}=========================================${NC}"
    echo -e "${CYAN}Phase 5: Docker Hub login (parallel)${NC}"
    echo -e "${CYAN}=========================================${NC}"
    echo ""

    echo -e "${BLUE}⏳ Logging in to Docker Hub on ${#HOSTS[@]} hosts...${NC}"

    for HOST in "${HOSTS[@]}"; do
        (
            REMOTE_CMDS="
                echo '${DOCKER_HUB_PASSWORD}' | docker login -u '${DOCKER_HUB_USERNAME}' --password-stdin
            "

            if sshpass -p "$SSH_PASSWORD" ssh -n -o StrictHostKeyChecking=no $REMOTE_USER@$HOST "$REMOTE_CMDS" 2>"$TEMP_DIR/phase5_${HOST}_error"; then
                rm -f "$TEMP_DIR/phase5_${HOST}_failed"
            else
                touch "$TEMP_DIR/phase5_${HOST}_failed"
            fi
        ) &
    done

    wait
    check_parallel_results "phase5"
    echo ""
else
    echo -e "${YELLOW}Skipping Phase 5: Docker Hub login (credentials not provided)${NC}"
    echo ""
fi

# ==========================================
# Phase 6: Docker Registry Setup
# ==========================================
echo -e "${CYAN}=========================================${NC}"
echo -e "${CYAN}Phase 6: Docker Registry Setup${NC}"
echo -e "${CYAN}=========================================${NC}"
echo ""

# Get management server IP from environment (set by mise or .env)
REGISTRY_HOST="${REGISTRY_HOST:-${DEPLOY_CTRL_SERVER_HOST:-localhost}}"
REGISTRY_PORT="5000"
REGISTRY_URL="${REGISTRY_HOST}:${REGISTRY_PORT}"

echo -e "${BLUE}📦 Setting up Docker Registry on management server (${REGISTRY_HOST})...${NC}"

# Start Registry container on management server (localhost)
REGISTRY_CMD="
    if ! docker ps --format '{{.Names}}' | grep -q '^registry$'; then
        docker run -d -p ${REGISTRY_PORT}:5000 --restart=always --name registry registry:2
        echo 'Registry started'
    else
        echo 'Registry already running'
    fi
"
if ! eval "$REGISTRY_CMD" 2>/dev/null; then
    echo -e "${RED}Failed to start Registry on management server${NC}"
    exit 1
fi
echo -e "${GREEN}  ✅ Registry running on ${REGISTRY_URL}${NC}"

echo -e "${BLUE}⏳ Configuring insecure-registries on ${#HOSTS[@]} hosts...${NC}"

for HOST in "${HOSTS[@]}"; do
    (
        REMOTE_CMDS="
            # Configure insecure registry
            DAEMON_JSON='/etc/docker/daemon.json'
            REGISTRY_URL='${REGISTRY_URL}'
            
            if [ -f \"\$DAEMON_JSON\" ]; then
                # Check if already configured
                if grep -q \"\$REGISTRY_URL\" \"\$DAEMON_JSON\" 2>/dev/null; then
                    echo 'Already configured'
                    exit 0
                fi
                # Merge with existing config
                EXISTING=\$(cat \"\$DAEMON_JSON\")
                echo \"\$EXISTING\" | jq '. + {\"insecure-registries\": ([\"'\$REGISTRY_URL'\"] + (.\"insecure-registries\" // []))}' | sudo tee \"\$DAEMON_JSON\" > /dev/null
            else
                # Create new config
                echo '{\"insecure-registries\": [\"'\$REGISTRY_URL'\"]}' | sudo tee \"\$DAEMON_JSON\" > /dev/null
            fi
            
            # Restart Docker to apply changes
            sudo systemctl restart docker
        "

        if sshpass -p "$SSH_PASSWORD" ssh -n -o StrictHostKeyChecking=no $REMOTE_USER@$HOST "$REMOTE_CMDS" 2>"$TEMP_DIR/phase6_${HOST}_error"; then
            rm -f "$TEMP_DIR/phase6_${HOST}_failed"
        else
            touch "$TEMP_DIR/phase6_${HOST}_failed"
        fi
    ) &
done

wait
check_parallel_results "phase6"
echo ""

# ==========================================
# 完了
# ==========================================
echo -e "${GREEN}=========================================${NC}"
echo -e "${GREEN}✅ All setup tasks completed successfully!${NC}"
echo -e "${GREEN}  Registry URL: ${REGISTRY_URL}${NC}"
echo -e "${GREEN}=========================================${NC}"
echo ""
