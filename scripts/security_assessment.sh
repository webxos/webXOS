#!/bin/bash

# Quarterly Security Assessment Script for vial-mcp
# Run this script manually or schedule via cron for quarterly execution

set -e

echo "Starting quarterly security assessment for vial-mcp..."

# 1. Dependency Vulnerability Scanning
echo "Scanning Python dependencies..."
pip install --upgrade safety
safety check -r main/api/mcp/requirements.txt > security_assessment_deps.txt
if grep -q "vulnerabilities found" security_assessment_deps.txt; then
  echo "WARNING: Vulnerabilities found in dependencies. Check security_assessment_deps.txt"
else
  echo "No dependency vulnerabilities found."
fi

echo "Scanning Node.js dependencies..."
npm install -g npm-audit
npm audit --prefix main/frontend > security_assessment_npm.txt
if grep -q "vulnerabilities found" security_assessment_npm.txt; then
  echo "WARNING: Vulnerabilities found in npm dependencies. Check security_assessment_npm.txt"
else
  echo "No npm vulnerabilities found."
fi

# 2. Security Log Review
echo "Reviewing security logs for the last 90 days..."
export PGPASSWORD=$DATABASE_URL_PASSWORD
psql $DATABASE_URL -c "SELECT event_type, COUNT(*) as count FROM security_events WHERE created_at > NOW() - INTERVAL '90 days' GROUP BY event_type ORDER BY count DESC;" > security_assessment_logs.txt
if grep -q "anomaly_detected" security_assessment_logs.txt; then
  echo "WARNING: Anomalies detected in logs. Review security_assessment_logs.txt"
else
  echo "No anomalies detected in logs."
fi

# 3. OWASP ZAP Full Scan
echo "Running OWASP ZAP full scan..."
docker run -u zap -t owasp/zap2docker-stable zap-full-scan.py -t https://webxos.netlify.app -r security_assessment_zap.html
echo "OWASP ZAP scan completed. Report saved to security_assessment_zap.html"

# 4. Code Quality Analysis
echo "Running code quality analysis..."
pip install --upgrade flake8
flake8 main/api/mcp --max-line-length=120 --output-file=security_assessment_flake8.txt
if [ -s security_assessment_flake8.txt ]; then
  echo "WARNING: Code quality issues found. Check security_assessment_flake8.txt"
else
  echo "No code quality issues found."
fi

# 5. Archive Results
echo "Archiving assessment results..."
tar -czf security_assessment_$(date +%Y%m%d).tar.gz security_assessment_*.txt security_assessment_zap.html
echo "Assessment completed. Results archived in security_assessment_$(date +%Y%m%d).tar.gz"
