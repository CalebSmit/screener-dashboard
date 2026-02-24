#!/usr/bin/env bash
# deploy_dashboard.sh — Push latest dashboard.html to GitHub Pages
# Usage: bash deploy_dashboard.sh

set -euo pipefail

# ---- Configuration ----
SCREENER_DIR="$(cd "$(dirname "$0")" && pwd)"
DASHBOARD_REPO="/c/Users/Caleb/OneDrive - Dordt University/Desktop/screener-dashboard"
SOURCE="$SCREENER_DIR/dashboard.html"
TARGET="$DASHBOARD_REPO/index.html"

# ---- Preflight checks ----
if [ ! -f "$SOURCE" ]; then
    echo "ERROR: dashboard.html not found at: $SOURCE"
    echo "Run 'python generate_dashboard.py' first."
    exit 1
fi

if [ ! -d "$DASHBOARD_REPO/.git" ]; then
    echo "ERROR: Dashboard repo not found at: $DASHBOARD_REPO"
    echo "Create the screener-dashboard folder first (see setup instructions)."
    exit 1
fi

# ---- Deploy ----
echo "Copying dashboard.html -> index.html ..."
cp "$SOURCE" "$TARGET"

cd "$DASHBOARD_REPO"

# Check if there are actually changes
if git diff --quiet index.html 2>/dev/null; then
    echo "No changes detected. Dashboard is already up to date."
    exit 0
fi

TIMESTAMP=$(date '+%Y-%m-%d %H:%M')
git add index.html
git commit -m "Update dashboard — $TIMESTAMP"
git push origin main

echo ""
echo "Deployed! Site will update in ~60 seconds at:"
echo "  https://calebsmit.github.io/screener-dashboard/"
