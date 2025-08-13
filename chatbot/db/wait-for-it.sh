#!/bin/bash
# wait-for-it.sh: Wait for a service to be available before executing a command.
#
# Usage: ./wait-for-it.sh host:port [-- command args]
# Example: ./wait-for-it.sh mongo:27017 -- uvicorn server:app --host 0.0.0.0 --port 8000

set -e

hostport="$1"
shift
cmd="$@"

if [ -z "$hostport" ]; then
    echo "Usage: $0 host:port [-- command args]"
    exit 1
fi

host=$(echo $hostport | cut -d: -f1)
port=$(echo $hostport | cut -d: -f2)

echo "Waiting for $host:$port..."

while ! nc -z $host $port; do
    sleep 0.1
done

echo "$host:$port is available"

if [ ! -z "$cmd" ]; then
    echo "Executing
