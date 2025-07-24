#!/bin/bash
# WAFL Testbed deployment shell script

# Please run the script from the PROJECT_DIR 
# (not PROJECT_DIR/ctrl, for example)
TARGET_PATH=$(pwd)
TARGET_NAME=$(basename $TARGET_PATH)

# Improved SSH connection settings
SSH_OPTS="-o ConnectTimeout=10 -o ServerAliveInterval=60 -o ServerAliveCountMax=3 -o StrictHostKeyChecking=no"

# Log file setup
LOGFILE="$TARGET_PATH/ctrl/deployment.log"
mkdir -p "$TARGET_PATH/ctrl"

# Importing the environment variables
if [ ! -f "$TARGET_PATH/ctrl/wafl_execution_base_config" ]; then
    echo "❌ Error: Configuration file not found: $TARGET_PATH/ctrl/wafl_execution_base_config"
    exit 1
fi
source "$TARGET_PATH/ctrl/wafl_execution_base_config"

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

    # Ensuring the existence of the device-specific directories on the management server
    mkdir -p "$TARGET_PATH/wafl/dataset/$device_name"
    mkdir -p "$TARGET_PATH/wafl/config/$device_name"
    mkdir -p "$TARGET_PATH/wafl/src/$device_name"
    
    echo "🔗 $(date): Connecting to Execution Server: $device_name ($device_ip)" | tee -a "$LOGFILE"
    ERROR_CHECK=0
    
    {
    # Setup remote directories
    ssh $SSH_OPTS "$USER@$device_ip" "rm -rf $DEPLOYMENT_LOCATION/$TARGET_NAME; \
        mkdir -p $DEPLOYMENT_LOCATION/$TARGET_NAME/dataset; \
        mkdir -p $DEPLOYMENT_LOCATION/$TARGET_NAME/config;
        mkdir -p $DEPLOYMENT_LOCATION/$TARGET_NAME/src;
        mkdir -p $DEPLOYMENT_LOCATION/$TARGET_NAME/results" &&

    # The Base Configuration shell script is also sent to the execution servers
    scp $SSH_OPTS -r -q "$TARGET_PATH/ctrl/wafl_execution_base_config" \
    "$USER@$device_ip:$DEPLOYMENT_LOCATION/$TARGET_NAME" &&

    # Check and send pyproject.toml and uv.lock if they exist
    if [ -f "$TARGET_PATH/ctrl/pyproject.toml" ] && [ -f "$TARGET_PATH/ctrl/uv.lock" ]; then
        echo "📦 Sending Python project files..." | tee -a "$LOGFILE"
        scp $SSH_OPTS -r -q "$TARGET_PATH/ctrl/pyproject.toml" \
        "$USER@$device_ip:$DEPLOYMENT_LOCATION/$TARGET_NAME" &&
        scp $SSH_OPTS -r -q "$TARGET_PATH/ctrl/uv.lock" \
        "$USER@$device_ip:$DEPLOYMENT_LOCATION/$TARGET_NAME" ||
        { echo "⚠️ Warning: Failed to send Python project files" | tee -a "$LOGFILE"; }
    else
        echo "⚠️ Warning: pyproject.toml or uv.lock not found in ctrl directory" | tee -a "$LOGFILE"
    fi &&
    
    # Setup Python virtual environment and install packages with uv
    echo "🐍 Setting up Python environment with uv..." | tee -a "$LOGFILE"
    ssh $SSH_OPTS "$USER@$device_ip" "cd $DEPLOYMENT_LOCATION/$TARGET_NAME && \
        { command -v ~/.local/bin/uv >/dev/null 2>&1 || \
        { echo '📥 uv not found, installing uv...' && \
        curl -LsSf https://astral.sh/uv/install.sh | sh && \
        export PATH=\"\$HOME/.local/bin:\$PATH\"; }; } && \
        { ~/.local/bin/uv venv .venv && \
        source .venv/bin/activate && \
        ~/.local/bin/uv sync || true; }" &&
    
    # File transfer operations with improved error handling
    { { [ "$(ls -A $TARGET_PATH/wafl/dataset/common 2>/dev/null)" ] && { ((++ERROR_CHECK)) || true; } &&
    echo "📂 Transferring common dataset files..." | tee -a "$LOGFILE" &&
    scp $SSH_OPTS -r -q "$TARGET_PATH/wafl/dataset/common/"* \
    "$USER@$device_ip:$DEPLOYMENT_LOCATION/$TARGET_NAME/dataset" && ((--ERROR_CHECK)); } || true; } &&
    { { [ "$(ls -A $TARGET_PATH/wafl/config/common 2>/dev/null)" ] && { ((++ERROR_CHECK)) || true; } &&
    echo "⚙️ Transferring common config files..." | tee -a "$LOGFILE" &&
    scp $SSH_OPTS -r -q "$TARGET_PATH/wafl/config/common/"* \
    "$USER@$device_ip:$DEPLOYMENT_LOCATION/$TARGET_NAME/config" && ((--ERROR_CHECK)); } || true; } &&
    { { [ "$(ls -A $TARGET_PATH/wafl/src/common 2>/dev/null)" ] && { ((++ERROR_CHECK)) || true; } && 
    echo "💻 Transferring common source files..." | tee -a "$LOGFILE" &&
    scp $SSH_OPTS -r -q "$TARGET_PATH/wafl/src/common/"* \
    "$USER@$device_ip:$DEPLOYMENT_LOCATION/$TARGET_NAME/src" && ((--ERROR_CHECK)); } || true; } &&
    { { [ "$(ls -A $TARGET_PATH/wafl/dataset/$device_name 2>/dev/null)" ] && { ((++ERROR_CHECK)) || true; } &&
    echo "📂 Transferring device-specific dataset files for $device_name..." | tee -a "$LOGFILE" &&
    scp $SSH_OPTS -r -q "$TARGET_PATH/wafl/dataset/$device_name/"* \
    "$USER@$device_ip:$DEPLOYMENT_LOCATION/$TARGET_NAME/dataset" && ((--ERROR_CHECK)); } || true; } &&
    { { [ "$(ls -A $TARGET_PATH/wafl/config/$device_name 2>/dev/null)" ] && { ((++ERROR_CHECK)) || true; } &&
    echo "⚙️ Transferring device-specific config files for $device_name..." | tee -a "$LOGFILE" &&
    scp $SSH_OPTS -r -q "$TARGET_PATH/wafl/config/$device_name/"* \
    "$USER@$device_ip:$DEPLOYMENT_LOCATION/$TARGET_NAME/config/" && ((--ERROR_CHECK)); } || true; } &&
    { { [ "$(ls -A $TARGET_PATH/wafl/src/$device_name 2>/dev/null)" ] && { ((++ERROR_CHECK)) || true; } &&
    echo "💻 Transferring device-specific source files for $device_name..." | tee -a "$LOGFILE" &&
    scp $SSH_OPTS -r -q "$TARGET_PATH/wafl/src/$device_name/"* \
    "$USER@$device_ip:$DEPLOYMENT_LOCATION/$TARGET_NAME/src" && ((--ERROR_CHECK)); } || true; 
    } && [ $ERROR_CHECK -eq 0 ] &&
    echo "✅ $(date): Successfully deployed project to $device_name ($device_ip)" | tee -a "$LOGFILE"
    } ||
    {
        echo "$USER@$device_ip" >> "$TARGET_PATH/ctrl/unsuccessful_deployment_list.txt"
        echo "❌ $(date): Failed to deploy project to $device_name ($device_ip)" | tee -a "$LOGFILE"
        return 1
    }
}

# Deploy to all devices sequentially
successful_deployments=0
total_devices=${#WAFL_DEVICE_NAMES[@]}

for ((counter=0; counter<$total_devices; counter++))
do
    if deploy_to_device $counter; then
        ((successful_deployments++))
    fi
done

# Deployment summary
echo "🎉 Deployment Complete!" | tee -a "$LOGFILE"
echo "📈 Summary: $successful_deployments/$total_devices devices deployed successfully" | tee -a "$LOGFILE"

if [ -f "$TARGET_PATH/ctrl/unsuccessful_deployment_list.txt" ] && [ -s "$TARGET_PATH/ctrl/unsuccessful_deployment_list.txt" ]; then
    echo "❌ Failed deployments listed in: $TARGET_PATH/ctrl/unsuccessful_deployment_list.txt" | tee -a "$LOGFILE"
    exit 1
else
    echo "✅ All deployments completed successfully!" | tee -a "$LOGFILE"
    exit 0
fi
