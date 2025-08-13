#!/bin/bash

timeout=15
waitfor=27017
host=mongo
port=$waitfor

while ! nc -z $host $port; do
  sleep 1
  timeout=$((timeout - 1))
  if [ $timeout -eq 0 ]; then
    echo "Timeout waiting for $host:$port"
    exit 1
  fi
done

echo "$host:$port is available"
exec "$@"
