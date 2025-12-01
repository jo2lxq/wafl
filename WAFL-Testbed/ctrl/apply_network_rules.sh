#!/bin/bash

# apply_network_rules.sh
# Usage: ./apply_network_rules.sh <container_name_or_id> <delay> <loss> <rate>
# Example: ./apply_network_rules.sh wafl-node-0 50ms 1% 100mbit

CONTAINER=$1
DELAY=$2
LOSS=$3
RATE=$4

if [ -z "$CONTAINER" ]; then
    echo "Usage: $0 <container_name_or_id> <delay> <loss> <rate>"
    exit 1
fi

# 1. Check if container exists and is running
CONTAINER_STATE=$(docker inspect -f '{{.State.Running}}' $CONTAINER 2>/dev/null)
if [ "$CONTAINER_STATE" != "true" ]; then
    echo "Error: Container $CONTAINER is not running (State: $CONTAINER_STATE)."
    echo "Waiting 5 seconds for container to start..."
    sleep 5
    CONTAINER_STATE=$(docker inspect -f '{{.State.Running}}' $CONTAINER 2>/dev/null)
    if [ "$CONTAINER_STATE" != "true" ]; then
        echo "Error: Container $CONTAINER still not running after wait."
        exit 1
    fi
fi

echo "Container $CONTAINER is running"

# 2. Get host veth interface index using docker exec
echo "Finding host veth interface..."

MAX_WAIT=30
WAIT_COUNT=0
HOST_VETH_INDEX=""

while [ $WAIT_COUNT -lt $MAX_WAIT ]; do
    # Use docker exec to read iflink from inside the container
    HOST_VETH_INDEX=$(docker exec $CONTAINER cat /sys/class/net/eth0/iflink 2>/dev/null)
    
    if [ ! -z "$HOST_VETH_INDEX" ]; then
        echo "Found iflink: $HOST_VETH_INDEX"
        break
    fi
    
    echo "Waiting for container network interface... ($((WAIT_COUNT + 1))/$MAX_WAIT)"
    sleep 1
    WAIT_COUNT=$((WAIT_COUNT + 1))
done

if [ -z "$HOST_VETH_INDEX" ]; then
    echo "Error: Could not read eth0/iflink from container"
    echo "Debug info:"
    echo "  - Container state: $(docker inspect -f '{{.State.Status}}' $CONTAINER 2>/dev/null)"
    echo "  - Container network mode: $(docker inspect -f '{{.HostConfig.NetworkMode}}' $CONTAINER 2>/dev/null)"
    echo "  - Interfaces in container:"
    docker exec $CONTAINER ip link show 2>&1 || echo "  Could not execute ip link in container"
    exit 1
fi

# 3. Find the veth interface name on host
HOST_VETH=$(ip link | grep "^${HOST_VETH_INDEX}:" | awk -F': ' '{print $2}' | awk -F'@' '{print $1}')

if [ -z "$HOST_VETH" ]; then
    echo "Error: Could not find host veth interface for index $HOST_VETH_INDEX"
    echo "Debug info:"
    echo "  - Looking for interface with index: $HOST_VETH_INDEX"
    echo "  - Available interfaces on host:"
    ip link show | head -20
    exit 1
fi

echo "Identified host interface: $HOST_VETH"

# 4. Apply tc rules
echo "Applying TC rules to $HOST_VETH: Delay=$DELAY, Loss=$LOSS, Rate=$RATE"

# Clear existing rules
sudo tc qdisc del dev $HOST_VETH root 2>/dev/null

# Build tc command
# Note: tc on veth shapes traffic GOING INTO the container (egress from host perspective)
CMD="sudo tc qdisc add dev $HOST_VETH root netem"
[ ! -z "$DELAY" ] && CMD="$CMD delay $DELAY"
[ ! -z "$LOSS" ] && CMD="$CMD loss $LOSS"
[ ! -z "$RATE" ] && CMD="$CMD rate $RATE"

echo "Executing: $CMD"
eval $CMD

EXIT_CODE=$?
if [ $EXIT_CODE -eq 0 ]; then
    echo "✅ Network rules applied successfully."
    
    # Verify the rules were applied
    echo "Verification:"
    sudo tc qdisc show dev $HOST_VETH
else
    echo "❌ Failed to apply network rules (exit code: $EXIT_CODE)."
    exit 1
fi
