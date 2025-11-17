#!/bin/bash
# WAFL Testbed deployment shell script

# Please run the script from the PROJECT_DIR 
# (not PROJECT_DIR/ctrl, for example)
TARGET_PATH=$(pwd)
TARGET_NAME=$(basename $TARGET_PATH)

# Improved SSH connection settings
SSH_OPTS="-o ConnectTimeout=10 -o ServerAliveInterval=60 -o ServerAliveCountMax=3 -o StrictHostKeyChecking=no"

# SSH multiplexing (connection reuse)
CTL_DIR="$HOME/.ssh/ctl"
mkdir -p "$CTL_DIR"
MUX_OPTS="-o ControlMaster=auto -o ControlPath=$CTL_DIR/%C -o ControlPersist=600"

# Log file setup - Clear existing log files before starting
LOGFILE="$TARGET_PATH/results/.deploy/ctrl.log"
mkdir -p "$TARGET_PATH/results/.deploy"

# Clear main log file and any existing device-specific log files
rm -f "$LOGFILE" 2>/dev/null || true
rm -f "$TARGET_PATH/results/.deploy/wafl"*.log 2>/dev/null || true

# Importing the environment variables
if [ ! -f "$TARGET_PATH/ctrl/execution_config" ]; then
    echo "❌ Error: Configuration file not found: $TARGET_PATH/ctrl/execution_config"
    exit 1
fi
source "$TARGET_PATH/ctrl/execution_config"

# Validate required environment variables
if [ -z "$DEPLOYMENT_LOCATION" ] || [ -z "$USER" ]; then
    echo "❌ Error: Required environment variables (DEPLOYMENT_LOCATION, USER) not set" | tee -a "$LOGFILE"
    exit 1
fi

# Deserializing Base Configuration File Lists
IFS=',' read -r -a WAFL_DEVICE_NAMES <<< "$WAFL_DEVICE_NAMES"
IFS=',' read -r -a WAFL_DEVICE_IPS <<< "$WAFL_DEVICE_IPS"

# Array size validation
if [ ${#WAFL_DEVICE_NAMES[@]} -ne ${#WAFL_DEVICE_IPS[@]} ]; then
    echo "❌ Error: Device names and IPs arrays have different lengths" | tee -a "$LOGFILE"
    exit 1
fi

# Deployment information
echo "🚀 $(date): Starting WAFL Testbed Deployment" | tee -a "$LOGFILE"
echo "📁 Project directory to be deployed: $TARGET_PATH" | tee -a "$LOGFILE"
echo "🎯 Target devices: ${#WAFL_DEVICE_NAMES[@]}" | tee -a "$LOGFILE"

CONFIRM="DEFAULT"
read -p "Please enter 'DEPLOY' to confirm: " CONFIRM
if [ "$CONFIRM" != "DEPLOY" ]
then
    echo "⛔ Aborting the process" | tee -a "$LOGFILE"
    exit 1
fi

# Ensuring the existence of the base directories on the 
# management server's project copy for replication
# on the execution servers (from the wafl sub-directory)
echo "📂 Creating base directories..." | tee -a "$LOGFILE"
mkdir -p "$TARGET_PATH/wafl/dataset/common/train"
mkdir -p "$TARGET_PATH/wafl/dataset/common/validate"
mkdir -p "$TARGET_PATH/wafl/dataset/common/test"
mkdir -p "$TARGET_PATH/wafl/config/common"
mkdir -p "$TARGET_PATH/wafl/src/common"

# Clearing the Unsuccessful Deployment List (improved error handling)
rm -f "$TARGET_PATH/ctrl/unsuccessful_deployment_list.txt" 2>/dev/null || true

echo "📁 Directory exists and will be replicated on all the execution servers via SSH" | tee -a "$LOGFILE"
echo "🎯 Directories will have the following path: $DEPLOYMENT_LOCATION/$TARGET_NAME" | tee -a "$LOGFILE"

# Function for deploying to individual device
deploy_to_device() {
    local counter=$1
    local device_name="${WAFL_DEVICE_NAMES[$counter]}"
    local device_ip="${WAFL_DEVICE_IPS[$counter]}"
    local device_logfile="$TARGET_PATH/results/.deploy/wafl${device_name}.log"
    local host="$USER@$device_ip"

    # 関数終了時にマスター接続を閉じる
    trap 'ssh $SSH_OPTS $MUX_OPTS -O exit "$host" >/dev/null 2>&1 || true' RETURN

    # Ensuring the existence of the device-specific directories on the management server
    mkdir -p "$TARGET_PATH/wafl/dataset/$device_name"
    mkdir -p "$TARGET_PATH/wafl/config/$device_name"
    mkdir -p "$TARGET_PATH/wafl/src/$device_name"
    
    echo "[wafl$device_name] 🔗 $(date): Connecting to Execution Server: $device_name ($device_ip)" | tee -a "$device_logfile"
    
    # マスター接続を確立
    ssh $SSH_OPTS $MUX_OPTS -fN "$host" || true
    
    ERROR_CHECK=0
    
    {
    # Setup remote directories
    ssh $SSH_OPTS $MUX_OPTS "$host" "rm -rf $DEPLOYMENT_LOCATION/$TARGET_NAME/dataset; \
        rm -rf $DEPLOYMENT_LOCATION/$TARGET_NAME/config; \
        rm -rf $DEPLOYMENT_LOCATION/$TARGET_NAME/src; \
        mkdir -p $DEPLOYMENT_LOCATION/$TARGET_NAME/dataset; \
        mkdir -p $DEPLOYMENT_LOCATION/$TARGET_NAME/config; \
        mkdir -p $DEPLOYMENT_LOCATION/$TARGET_NAME/results; \
        mkdir -p $DEPLOYMENT_LOCATION/$TARGET_NAME/src" &&

    # The Base Configuration shell script is also sent to the execution servers
    scp $SSH_OPTS $MUX_OPTS -r -q "$TARGET_PATH/ctrl/execution_config" \
    "$host:$DEPLOYMENT_LOCATION/$TARGET_NAME" &&

    # Check and send pyproject.toml and uv.lock if they exist
    if [ -f "$TARGET_PATH/pyproject.toml" ] && [ -f "$TARGET_PATH/uv.lock" ]; then
        echo "[wafl$device_name] 📦 Sending Python project files..." | tee -a "$device_logfile"
        scp $SSH_OPTS $MUX_OPTS -r -q "$TARGET_PATH/pyproject.toml" \
        "$host:$DEPLOYMENT_LOCATION/$TARGET_NAME" &&
        scp $SSH_OPTS $MUX_OPTS -r -q "$TARGET_PATH/uv.lock" \
        "$host:$DEPLOYMENT_LOCATION/$TARGET_NAME" ||
        { echo "[wafl$device_name] ⚠️ Warning: Failed to send Python project files" | tee -a "$device_logfile"; }
    else
        echo "[wafl$device_name] ⚠️ Warning: pyproject.toml or uv.lock not found" | tee -a "$device_logfile"
    fi &&
    
    # Setup Python virtual environment and install packages with uv
    echo "[wafl$device_name] 🐍 Setting up Python environment with uv..." | tee -a "$device_logfile"
    ssh $SSH_OPTS $MUX_OPTS "$host" "cd $DEPLOYMENT_LOCATION/$TARGET_NAME && \
        { command -v ~/.local/bin/uv >/dev/null 2>&1 || \
        { echo '📥 uv not found, installing uv...' && \
        curl -LsSf https://astral.sh/uv/install.sh | sh && \
        export PATH=\"\$HOME/.local/bin:\$PATH\"; }; } && \
        export UV_HTTP_TIMEOUT=3600 && \
        { ~/.local/bin/uv venv .venv --clear && \
        source .venv/bin/activate && \
        ~/.local/bin/uv sync || true; }" &&
    
    # File transfer operations with improved error handling
    { { [ "$(ls -A $TARGET_PATH/wafl/dataset/common 2>/dev/null)" ] && { ((++ERROR_CHECK)) || true; } &&
    echo "[wafl$device_name] 📂 Transferring common dataset files..." | tee -a "$device_logfile" &&
    scp $SSH_OPTS $MUX_OPTS -r -q "$TARGET_PATH/wafl/dataset/common/"* \
    "$host:$DEPLOYMENT_LOCATION/$TARGET_NAME/dataset" && ((--ERROR_CHECK)); } || true; } &&
    { { [ "$(ls -A $TARGET_PATH/wafl/config/common 2>/dev/null)" ] && { ((++ERROR_CHECK)) || true; } &&
    echo "[wafl$device_name] ⚙️ Transferring common config files..." | tee -a "$device_logfile" &&
    scp $SSH_OPTS $MUX_OPTS -r -q "$TARGET_PATH/wafl/config/common/"* \
    "$host:$DEPLOYMENT_LOCATION/$TARGET_NAME/config" && ((--ERROR_CHECK)); } || true; } &&
    { { [ "$(ls -A $TARGET_PATH/wafl/src/common 2>/dev/null)" ] && { ((++ERROR_CHECK)) || true; } && 
    echo "[wafl$device_name] 💻 Transferring common source files..." | tee -a "$device_logfile" &&
    scp $SSH_OPTS $MUX_OPTS -r -q "$TARGET_PATH/wafl/src/common/"* \
    "$host:$DEPLOYMENT_LOCATION/$TARGET_NAME/src" && ((--ERROR_CHECK)); } || true; } &&
    { { [ "$(ls -A $TARGET_PATH/wafl/dataset/$device_name 2>/dev/null)" ] && { ((++ERROR_CHECK)) || true; } &&
    echo "[wafl$device_name] 📂 Transferring device-specific dataset files for $device_name..." | tee -a "$device_logfile" &&
    scp $SSH_OPTS $MUX_OPTS -r -q "$TARGET_PATH/wafl/dataset/$device_name/"* \
    "$host:$DEPLOYMENT_LOCATION/$TARGET_NAME/dataset" && ((--ERROR_CHECK)); } || true; } &&
    { { [ "$(ls -A $TARGET_PATH/wafl/config/$device_name 2>/dev/null)" ] && { ((++ERROR_CHECK)) || true; } &&
    echo "[wafl$device_name] ⚙️ Transferring device-specific config files for $device_name..." | tee -a "$device_logfile" &&
    scp $SSH_OPTS $MUX_OPTS -r -q "$TARGET_PATH/wafl/config/$device_name/"* \
    "$host:$DEPLOYMENT_LOCATION/$TARGET_NAME/config/" && ((--ERROR_CHECK)); } || true; } &&
    { { [ "$(ls -A $TARGET_PATH/wafl/src/$device_name 2>/dev/null)" ] && { ((++ERROR_CHECK)) || true; } &&
    echo "[wafl$device_name] 💻 Transferring device-specific source files for $device_name..." | tee -a "$device_logfile" &&
    scp $SSH_OPTS $MUX_OPTS -r -q "$TARGET_PATH/wafl/src/$device_name/"* \
    "$host:$DEPLOYMENT_LOCATION/$TARGET_NAME/src" && ((--ERROR_CHECK)); } || true; 
    } && [ $ERROR_CHECK -eq 0 ] &&
    echo "[wafl$device_name] ✅ $(date): Successfully deployed project to $device_name ($device_ip)" | tee -a "$device_logfile"
    } ||
    {
        echo "$host" >> "$TARGET_PATH/ctrl/unsuccessful_deployment_list.txt"
        echo "[wafl$device_name] ❌ $(date): Failed to deploy project to $device_name ($device_ip)" | tee -a "$device_logfile"
        return 1
    }
}

# Deploy to all devices in parallel
echo "[ctrl] 🚀 Starting parallel deployment to all devices..." | tee -a "$LOGFILE"
pids=()
total_devices=${#WAFL_DEVICE_NAMES[@]}

for ((counter=0; counter<$total_devices; counter++))
do
    echo "[ctrl] 🔄 Starting deployment to ${WAFL_DEVICE_NAMES[$counter]} in background..." | tee -a "$LOGFILE"
    deploy_to_device $counter &
    pids+=($!)
done

# Wait for all background processes to complete
echo "[ctrl] ⏳ Waiting for all deployments to complete..." | tee -a "$LOGFILE"
successful_deployments=0
for pid in "${pids[@]}"; do
    if wait $pid; then
        ((successful_deployments++))
    fi
done

# Merge individual device logs into main log
echo "[ctrl] 📝 Merging deployment logs..." | tee -a "$LOGFILE"
for ((counter=0; counter<$total_devices; counter++))
do
    device_name="${WAFL_DEVICE_NAMES[$counter]}"
    device_logfile="$TARGET_PATH/results/.deploy/wafl${device_name}.log"
    if [ -f "$device_logfile" ]; then
        echo "--- Log from $device_name ---" >> "$LOGFILE"
        cat "$device_logfile" >> "$LOGFILE"
        echo "--- End of $device_name log ---" >> "$LOGFILE"
    fi
done

# Deployment summary
echo "[ctrl] 🎉 Parallel Deployment Complete!" | tee -a "$LOGFILE"
echo "[ctrl] 📈 Summary: $successful_deployments/$total_devices devices deployed successfully" | tee -a "$LOGFILE"

if [ -f "$TARGET_PATH/ctrl/unsuccessful_deployment_list.txt" ] && [ -s "$TARGET_PATH/ctrl/unsuccessful_deployment_list.txt" ]; then
    echo "[ctrl] ❌ Failed deployments listed in: $TARGET_PATH/ctrl/unsuccessful_deployment_list.txt" | tee -a "$LOGFILE"
    exit 1
else
    echo "[ctrl] ✅ All parallel deployments completed successfully!" | tee -a "$LOGFILE"
    exit 0
fi
