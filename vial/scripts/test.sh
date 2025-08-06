#!/bin/bash
set -e
npm install -g mocha
mocha tests/unit/*.test.js
mocha tests/integration/*.test.js
