#!/bin/bash

# WAFL Testbed deployment shell script
target_name=""
target_location=""
target_confirm=""
# Enter the target directory name: 
# Eg. WAFL_DEMO
read -p "Please enter the target directory name: " target_name
# Enter the target directory location: 
# Eg. /home/workspace/namit/
read -p "Please enter the target directory location: " target_location
# Please confirm the full directory path: 
# Eg. /home/workspace/namit/WAFL_DEMO
read -p "Please confirm the full directory path: " target_confirm
# The directory will be set on the execution servers as follows: /home/testbed/WAFL_DEMO
if [ "$target_location$target_name" != "$target_confirm" ]
then
    echo "Directory names do not match!"
    exit 1
fi
if [ ! -d "$target_confirm" ] 
then
    echo "Directory does not exist"
    exit 1
fi
# Ensuring the existence of the base directories on the 
# management server's project copy for replication
# on the execution servers (from the wafl sub-directory)
mkdir -p "$target_confirm/wafl/dataset"
mkdir -p "$target_confirm/wafl/config"
mkdir -p "$target_confirm/wafl/src"
final_location="/home/denjo/testbed/"
echo "Directory exists and will be replicated on all the execution servers via SSH"
echo "Directories will have the following path: $final_location$target_name"

# The Devices of the Network
devices=("192.168.11.100" "192.168.11.101" "192.168.11.102" 
        "192.168.11.103" "192.168.11.104" "192.168.11.105" 
        "192.168.11.106" "192.168.11.107" "192.168.11.108" 
        "192.168.11.109")
user="denjo"
for device in "${devices[@]}" 
do
    echo "Connecting to Execution Server: $device"
    if [ -f "$final_location$target_name" ]
    then
        ssh "$user@$device" "rm -r $final_location$target_name"
    fi
    ssh "$user@$device" "mkdir -p $final_location$target_name"
    scp -r -q "$target_confirm/wafl/dataset" "$user@$device:$final_location$target_name"
    scp -r -q "$target_confirm/wafl/config" "$user@$device:$final_location$target_name"
    scp -r -q "$target_confirm/wafl/src" "$user@$device:$final_location$target_name"
    ssh "$user@$device" "mkdir -p $final_location$target_name/results"
    echo "Successfully deployed project to the device"
done
echo "Deployment Complete!"