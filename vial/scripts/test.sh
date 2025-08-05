#!/bin/bash
# Run tests
npm test
pytest src/pipelines
echo "Tests complete"

# Instructions:
# - Runs Jest and Pytest
# - Install: `npm install jest supertest` and `pip install pytest`
# - Run: `chmod +x scripts/test.sh && ./scripts/test.sh`
