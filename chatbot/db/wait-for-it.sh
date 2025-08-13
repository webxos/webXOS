#!/bin/bash

host="$1"
shift
cmd="$@"

until nc -z -v -w30 $host; do
  echo "Waiting for $host to be available..."
  sleep 1
done

echo "$host is available, executing command..."
exec $cmd
