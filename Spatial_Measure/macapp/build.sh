#!/bin/bash
# Builds "build/Spatial Measure.app" with the Command Line Tools (no Xcode needed).
#   ./macapp/build.sh && open "build/Spatial Measure.app"
set -euo pipefail
cd "$(dirname "$0")"
ROOT="$(cd .. && pwd)"
APP="$ROOT/build/Spatial Measure.app"

rm -rf "$APP"
mkdir -p "$APP/Contents/MacOS" "$APP/Contents/Resources"
swiftc -O -swift-version 5 -parse-as-library \
    -target "$(uname -m)-apple-macos14.0" \
    -o "$APP/Contents/MacOS/SpatialMeasure" Sources/*.swift
cp Info.plist "$APP/Contents/Info.plist"
# The app runs engine_server.py from this project with the project's .venv.
plutil -replace SMProjectDir -string "$ROOT" "$APP/Contents/Info.plist"
codesign --force --sign - "$APP"
echo "Built: $APP"
