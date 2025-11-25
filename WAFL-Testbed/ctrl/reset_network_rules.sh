#!/bin/bash

# reset_network_rules.sh
# Usage: ./reset_network_rules.sh <container_name_or_id>

CONTAINER=$1

if [ -z "$CONTAINER" ]; then
    echo "Usage: $0 <container_name_or_id>"
    exit 1
fi

# 1. Get Container PID
PID=$(docker inspect -f '{{.State.Pid}}' $CONTAINER)
if [ -z "$PID" ]; then
    echo "Error: Container $CONTAINER not found or not running."
    exit 1
fi

# 2. Find host veth
HOST_VETH_INDEX=$(sudo nsenter -t $PID -n cat /sys/class/net/eth0/iflink)
HOST_VETH=$(ip link | grep "^${HOST_VETH_INDEX}:" | awk -F': ' '{print $2}' | awk -F'@' '{print $1}')

if [ -z "$HOST_VETH" ]; then
    echo "Error: Could not find host veth interface."
    exit 1
fi

echo "Resetting rules for $HOST_VETH (Container: $CONTAINER)"

# 3. Delete tc rules
sudo tc qdisc del dev $HOST_VETH root 2>/dev/null

if [ $? -eq 0 ]; then
    echo "✅ Network rules reset successfully."
else
    echo "⚠️  Failed to reset rules (or no rules existed)."
fi
