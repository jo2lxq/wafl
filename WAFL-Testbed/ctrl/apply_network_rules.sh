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
    echo "Waiting 2 seconds for container to start..."
    sleep 2
    CONTAINER_STATE=$(docker inspect -f '{{.State.Running}}' $CONTAINER 2>/dev/null)
    if [ "$CONTAINER_STATE" != "true" ]; then
        echo "Error: Container $CONTAINER still not running after wait."
        exit 1
    fi
fi

# 2. Get Container PID
PID=$(docker inspect -f '{{.State.Pid}}' $CONTAINER 2>/dev/null)
if [ -z "$PID" ] || [ "$PID" = "0" ]; then
    echo "Error: Container $CONTAINER not found or PID is 0."
    echo "Container status:"
    docker inspect -f 'Running: {{.State.Running}}, Status: {{.State.Status}}, PID: {{.State.Pid}}' $CONTAINER 2>/dev/null || echo "Container not found"
    exit 1
fi

echo "Container $CONTAINER found with PID $PID"

# 3. Get veth interface index from container's namespace
# We need to run ip link inside the container's network namespace
# Since we are on the host, we can use nsenter
VETH_INDEX=$(sudo nsenter --net=/proc/$PID/ns/net ip link | grep "eth0@" | awk -F': ' '{print $1}')

if [ -z "$VETH_INDEX" ]; then
    echo "Error: Could not find eth0 interface index in container."
    exit 1
fi

echo "Container eth0 index: $VETH_INDEX"

# 4. Find the corresponding veth interface on the host
# The host veth interface will have an index that matches the container's iflink,
# but usually we look for the interface that links TO the container's index.
# Actually, `ip link` on host shows `vethXXXX@ifYYYY`. YYYY is the index inside the container.
# Wait, it's the other way around or paired.
# A more reliable way:
# Inside container: `cat /sys/class/net/eth0/iflink` -> gets the index of the peer on host.

HOST_VETH_INDEX=$(sudo nsenter --net=/proc/$PID/ns/net cat /sys/class/net/eth0/iflink)
HOST_VETH=$(ip link | grep "^${HOST_VETH_INDEX}:" | awk -F': ' '{print $2}' | awk -F'@' '{print $1}')

if [ -z "$HOST_VETH" ]; then
    echo "Error: Could not find host veth interface for index $HOST_VETH_INDEX."
    exit 1
fi

echo "Identified host interface: $HOST_VETH"

# 5. Apply tc rules
echo "Applying TC rules to $HOST_VETH: Delay=$DELAY, Loss=$LOSS, Rate=$RATE"

# Clear existing rules
sudo tc qdisc del dev $HOST_VETH root 2>/dev/null

# Add root qdisc (htb for rate limiting)
# sudo tc qdisc add dev $HOST_VETH root handle 1: htb default 11

# Add class with rate limit
# sudo tc class add dev $HOST_VETH parent 1: classid 1:1 htb rate $RATE

# Add netem qdisc for delay and loss
# sudo tc qdisc add dev $HOST_VETH parent 1:1 handle 10: netem delay $DELAY loss $LOSS

# Simplified version using only netem if rate limit is not strict or if we just want to shape egress
# Note: tc on veth shapes traffic GOING INTO the container (egress from host perspective)
CMD="sudo tc qdisc add dev $HOST_VETH root netem"
[ ! -z "$DELAY" ] && CMD="$CMD delay $DELAY"
[ ! -z "$LOSS" ] && CMD="$CMD loss $LOSS"
[ ! -z "$RATE" ] && CMD="$CMD rate $RATE"

echo "Executing: $CMD"
eval $CMD

if [ $? -eq 0 ]; then
    echo "✅ Network rules applied successfully."
else
    echo "❌ Failed to apply network rules."
    exit 1
fi
