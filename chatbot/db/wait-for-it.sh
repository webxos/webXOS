#!/bin/bash

set -e

host="$1"
port="$2"
timeout="${3:-30}"

until nc -z -w 5 "$host" "$port"; do
  echo "Waiting for $host:$port..."
  sleep 1
  ((timeout--))
  if [ $timeout -le 0 ]; then
    echo "Timeout waiting for $host:$port"
    exit 1
  fi
done

echo "$host:$port is available"
