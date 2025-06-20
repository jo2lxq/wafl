#!/bin/bash
# WAFL Testbed deployment shell script

# Please run the script from the PROJECT_DIR 
# (not PROJECT_DIR/ctrl, for example)
TARGET_PATH=$(pwd)
TARGET_NAME=$(basename $TARGET_PATH)
#Importing the environment variables
source "$TARGET_PATH/ctrl/wafl_base_execution_config"
#Deserializing Base Configuration File Lists
IFS=',' read -r -a WAFL_DEVICE_NAMES <<< "$WAFL_DEVICE_NAMES"
IFS=',' read -r -a WAFL_DEVICE_IPS <<< "$WAFL_DEVICE_IPS"

CONFIRM="DEFAULT"
echo "The project directory to be deployed (PWD): $TARGET_PATH"
read -p "Please enter 'DEPLOY' to confirm: " CONFIRM
if [ "$CONFIRM" != "DEPLOY" ]
then
    echo "Aborting the process"
    exit 1
fi
# Ensuring the existence of the base directories on the 
# management server's project copy for replication
# on the execution servers (from the wafl sub-directory)
mkdir -p "$TARGET_PATH/wafl/dataset/common/train"
mkdir -p "$TARGET_PATH/wafl/dataset/common/validate"
mkdir -p "$TARGET_PATH/wafl/dataset/common/test"
mkdir -p "$TARGET_PATH/wafl/config/common"
mkdir -p "$TARGET_PATH/wafl/src/common"
# Clearing the Unsuccessful Deployment List
rm "$TARGET_PATH/ctrl/unsuccessful_deployment_list.txt"
echo "Directory exists and will be replicated on all the execution servers via SSH"
echo "Directories will have the following path: $DEPLOYMENT_LOCATION/$TARGET_NAME"
for ((counter=0; counter<${#WAFL_DEVICE_NAMES[@]}; counter++))
do
    # Ensuring the existence of the device-specific
    # directories on the management server
    mkdir -p "$TARGET_PATH/wafl/dataset/${WAFL_DEVICE_NAMES[$counter]}"
    mkdir -p "$TARGET_PATH/wafl/config/${WAFL_DEVICE_NAMES[$counter]}"
    mkdir -p "$TARGET_PATH/wafl/src/${WAFL_DEVICE_NAMES[$counter]}"
    echo "Connecting to Execution Server: ${WAFL_DEVICE_NAMES[$counter]}"
    (ssh -o ConnectTimeout=5 "$USER@${WAFL_DEVICE_IPS[$counter]}" "rm -rf $DEPLOYMENT_LOCATION/$TARGET_NAME; \
        mkdir -p $DEPLOYMENT_LOCATION/$TARGET_NAME/dataset; \
        mkdir -p $DEPLOYMENT_LOCATION/$TARGET_NAME/config;
        mkdir -p $DEPLOYMENT_LOCATION/$TARGET_NAME/src;
        mkdir -p $DEPLOYMENT_LOCATION/$TARGET_NAME/results" &&
    # The Base Configuration shell script is also sent to the execution servers
    scp -r -q "$TARGET_PATH/ctrl/wafl_base_execution_config" \
    "$USER@${WAFL_DEVICE_IPS[$counter]}:$DEPLOYMENT_LOCATION/$TARGET_NAME" &&
    ([ "$(ls -A $TARGET_PATH/wafl/dataset/common)" ] && 
    scp -r -q "$TARGET_PATH/wafl/dataset/common/"* \
    "$USER@${WAFL_DEVICE_IPS[$counter]}:$DEPLOYMENT_LOCATION/$TARGET_NAME/dataset" || true) &&
    ([ "$(ls -A $TARGET_PATH/wafl/config/common)" ] && 
    scp -r -q "$TARGET_PATH/wafl/config/common/"* \
    "$USER@${WAFL_DEVICE_IPS[$counter]}:$DEPLOYMENT_LOCATION/$TARGET_NAME/config" || true) &&
    ([ "$(ls -A $TARGET_PATH/wafl/src/common)" ] && 
    scp -r -q "$TARGET_PATH/wafl/src/common/"* \
    "$USER@${WAFL_DEVICE_IPS[$counter]}:$DEPLOYMENT_LOCATION/$TARGET_NAME/src" || true) &&
    ([ "$(ls -A $TARGET_PATH/wafl/dataset/${WAFL_DEVICE_NAMES[$counter]})" ] &&
    scp -r -q "$TARGET_PATH/wafl/dataset/${WAFL_DEVICE_NAMES[$counter]}/"* \
    "$USER@${WAFL_DEVICE_IPS[$counter]}:$DEPLOYMENT_LOCATION/$TARGET_NAME/dataset" || true) &&
    ([ "$(ls -A $TARGET_PATH/wafl/config/${WAFL_DEVICE_NAMES[$counter]})" ] &&
    scp -r -q "$TARGET_PATH/wafl/config/${WAFL_DEVICE_NAMES[$counter]}/"* \
    "$USER@${WAFL_DEVICE_IPS[$counter]}:$DEPLOYMENT_LOCATION/$TARGET_NAME/config/" || true) &&
    ([ "$(ls -A $TARGET_PATH/wafl/src/${WAFL_DEVICE_NAMES[$counter]})" ] &&
    scp -r -q "$TARGET_PATH/wafl/src/${WAFL_DEVICE_NAMES[$counter]}/"* \
    "$USER@${WAFL_DEVICE_IPS[$counter]}:$DEPLOYMENT_LOCATION/$TARGET_NAME/src" || true) &&
    echo "Successfully deployed project to the device") ||
    (echo "$USER@${WAFL_DEVICE_IPS[$counter]}" >>  "$TARGET_PATH/ctrl/unsuccessful_deployment_list.txt";
    echo "Unsuccessful in deploying project to the device")
done
echo "Deployment Complete!"
