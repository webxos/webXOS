#!/bin/bash

# Security Assessment Script for vial-mcp

# Configuration
DATABASE_URL=${DATABASE_URL:-"postgres://localhost:5432/vial_mcp"}
REPORT_FILE="security_assessment_$(date +%Y%m%d_%H%M%S).txt"
ALERT_EMAIL=${ALERT_EMAIL:-"security@webxos.netlify.app"}
SMTP_SERVER=${SMTP_SERVER:-"smtp.gmail.com"}
SMTP_PORT=${SMTP_PORT:-587}
SMTP_USER=${SMTP_USER}
SMTP_PASSWORD=${SMTP_PASSWORD}

# Function to send email report
send_email() {
    local subject=$1
    local body=$2
    echo -e "Subject: $subject\n\n$body" | sendmail -f "$SMTP_USER" -t "$ALERT_EMAIL"
}

# Function to query database
query_db() {
    local query=$1
    psql "$DATABASE_URL" -t -A -c "$query"
}

# Initialize report
echo "Security Assessment Report - $(date)" > "$REPORT_FILE"
echo "=====================================" >> "$REPORT_FILE"

# 1. Check for outdated sessions
echo "Checking for outdated sessions..." >> "$REPORT_FILE"
outdated_sessions=$(query_db "SELECT COUNT(*) FROM sessions WHERE expires_at < CURRENT_TIMESTAMP;")
echo "Outdated sessions found: $outdated_sessions" >> "$REPORT_FILE"
if [ "$outdated_sessions" -gt 0 ]; then
    echo "Cleaning up outdated sessions..." >> "$REPORT_FILE"
    query_db "DELETE FROM sessions WHERE expires_at < CURRENT_TIMESTAMP;"
    echo "Outdated sessions cleaned." >> "$REPORT_FILE"
fi
echo "" >> "$REPORT_FILE"

# 2. Check for recent anomalies
echo "Checking for recent anomalies..." >> "$REPORT_FILE"
anomalies=$(query_db "SELECT COUNT(*) FROM security_events WHERE event_type = 'anomaly_detected' AND created_at > CURRENT_TIMESTAMP - INTERVAL '24 hours';")
echo "Anomalies detected in last 24 hours: $anomalies" >> "$REPORT_FILE"
if [ "$anomalies" -gt 0 ]; then
    echo "Recent anomalies:" >> "$REPORT_FILE"
    query_db "SELECT details FROM security_events WHERE event_type = 'anomaly_detected' AND created_at > CURRENT_TIMESTAMP - INTERVAL '24 hours';" >> "$REPORT_FILE"
fi
echo "" >> "$REPORT_FILE"

# 3. Analyze audit logs
echo "Analyzing audit logs..." >> "$REPORT_FILE"
audit_log_count=$(query_db "SELECT COUNT(*) FROM audit_logs WHERE created_at > CURRENT_TIMESTAMP - INTERVAL '24 hours';")
echo "Audit log entries in last 24 hours: $audit_log_count" >> "$REPORT_FILE"
if [ "$audit_log_count" -gt 0 ]; then
    echo "Top user actions:" >> "$REPORT_FILE"
    query_db "SELECT action, COUNT(*) as count FROM audit_logs WHERE created_at > CURRENT_TIMESTAMP - INTERVAL '24 hours' GROUP BY action ORDER BY count DESC LIMIT 5;" >> "$REPORT_FILE"
    
    suspicious_actions=$(query_db "SELECT COUNT(*) FROM audit_logs WHERE action IN ('data_erasure', 'anomaly_detected') AND created_at > CURRENT_TIMESTAMP - INTERVAL '24 hours';")
    echo "Suspicious actions (data erasure, anomalies): $suspicious_actions" >> "$REPORT_FILE"
    if [ "$suspicious_actions" -gt 0 ]; then
        echo "Details of suspicious actions:" >> "$REPORT_FILE"
        query_db "SELECT user_id, action, details, created_at FROM audit_logs WHERE action IN ('data_erasure', 'anomaly_detected') AND created_at > CURRENT_TIMESTAMP - INTERVAL '24 hours';" >> "$REPORT_FILE"
    fi
fi
echo "" >> "$REPORT_FILE"

# 4. Check for high authentication failure rates
echo "Checking authentication failure rates..." >> "$REPORT_FILE"
auth_failures=$(query_db "SELECT COUNT(*) FROM security_events WHERE event_type = 'auth_error' AND created_at > CURRENT_TIMESTAMP - INTERVAL '1 hour';")
if [ "$auth_failures" -gt 5 ]; then
    echo "High authentication failure rate detected: $auth_failures failures in last hour" >> "$REPORT_FILE"
    send_email "Security Alert: High Authentication Failure Rate" "Detected $auth_failures authentication failures in the last hour. Please review the attached report."
fi
echo "Authentication failures in last hour: $auth_failures" >> "$REPORT_FILE"
echo "" >> "$REPORT_FILE"

# 5. Check for API credential leaks
echo "Checking for exposed API credentials..." >> "$REPORT_FILE"
exposed_credentials=$(query_db "SELECT COUNT(*) FROM users WHERE api_key IS NOT NULL AND api_secret IS NOT NULL;")
if [ "$exposed_credentials" -gt 0 ]; then
    echo "WARNING: Found $exposed_credentials users with active API credentials" >> "$REPORT_FILE"
    send_email "Security Alert: Active API Credentials Detected" "Found $exposed_credentials users with active API credentials. Please review and revoke unnecessary credentials."
fi
echo "" >> "$REPORT_FILE"

# Finalize report
echo "Assessment completed at $(date)" >> "$REPORT_FILE"

# Send report via email
send_email "Vial MCP Security Assessment Report" "$(cat $REPORT_FILE)"
