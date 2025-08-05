#!/bin/bash
# Build APK
cd static
npx cap init --web-dir . vial-mcp com.webxos.vial
npx cap add android
npx cap sync
npx cap open android
echo "Open Android Studio to build APK"

# Instructions:
# - Generates Android APK
# - Install: `cd static && npm install @capacitor/core @capacitor/cli @capacitor/android`
# - Run: `chmod +x scripts/build-apk.sh && ./scripts/build-apk.sh`
