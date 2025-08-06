#!/bin/bash
# Test script for Vial MCP Controller
# Runs unit and integration tests
# Rebuild: Run `chmod +x test.sh` and `./test.sh`

echo "Running tests for Vial MCP Controller..."

# Install test dependencies
npm install mocha chai --save-dev

# Run unit tests
for test in tests/unit/*.test.js; do
  npx mocha $test
  if [ $? -ne 0 ]; then
    echo "[ERROR] Unit test failed: $test"
    exit 1
  fi
done

# Run integration tests
npx mocha tests/integration/server.test.js
if [ $? -ne 0 ]; then
  echo "[ERROR] Integration test failed"
  exit 1
fi

echo "Tests passed."

# Rebuild Instructions: Place in /vial/scripts/. Run `chmod +x test.sh` and `./test.sh`. Ensure /vial/tests/ exists. Check /vial/errorlog.md for issues.
