#!/bin/bash
# Test network connectivity between nodes for distributed training

MAIN_IP="10.15.38.17"
MAIN_PORT="29500"
NODES=("g369" "g375" "g255" "g265" "g337" "g340" "g341" "g345")

echo "Testing network connectivity for multi-node training..."
echo ""

# Test 1: Check if nodes can reach the main node's IP
echo "Test 1: Ping main node IP (${MAIN_IP})"
for node in "${NODES[@]}"; do
    echo -n "  ${node} -> ${MAIN_IP}: "
    ssh ${node} "ping -c 1 -W 1 ${MAIN_IP} > /dev/null 2>&1 && echo 'OK' || echo 'FAILED'"
done
echo ""

# Test 2: Check if port 29500 is accessible
echo "Test 2: Check if nodes can connect to port ${MAIN_PORT}"
echo "  Starting test server on g369..."
ssh g369 "python3 -c 'import socket; s=socket.socket(); s.bind((\"0.0.0.0\", ${MAIN_PORT})); s.listen(1); print(\"Listening on ${MAIN_PORT}\")' &" &
sleep 2

for node in "${NODES[@]}"; do
    if [ "$node" != "g369" ]; then
        echo -n "  ${node} -> g369:${MAIN_PORT}: "
        ssh ${node} "timeout 2 bash -c 'cat < /dev/null > /dev/tcp/${MAIN_IP}/${MAIN_PORT}' 2>/dev/null && echo 'OK' || echo 'FAILED'"
    fi
done

# Cleanup
ssh g369 "pkill -f 'socket.*${MAIN_PORT}'" 2>/dev/null
echo ""

# Test 3: Check hostname resolution
echo "Test 3: Hostname to IP resolution"
for node in "${NODES[@]}"; do
    echo -n "  ${node}: "
    ssh ${node} "hostname -I | awk '{print \$1}'"
done
echo ""

echo "✅ Network test complete!"
echo ""
echo "If any tests FAILED, multi-node training won't work."
echo "All nodes must be able to reach ${MAIN_IP}:${MAIN_PORT}"

