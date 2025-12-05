#!/bin/bash
# Wrapper script to run Streamlit app with custom temp directory

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
TEMP_DIR="$PROJECT_ROOT/.tmp"

# Check disk space
AVAILABLE_SPACE=$(df -h "$PROJECT_ROOT" | tail -1 | awk '{print $4}')
echo "Available disk space: $AVAILABLE_SPACE"

# Check if we have at least 100MB free (basic check)
if ! df "$PROJECT_ROOT" | tail -1 | awk '{if ($4+0 < 100) exit 1}'; then
    echo "⚠️  WARNING: Low disk space detected. You may encounter errors."
    echo "   Please free up at least 100MB of disk space."
fi

# Create temp directory if it doesn't exist
mkdir -p "$TEMP_DIR" || {
    echo "❌ ERROR: Cannot create temp directory: $TEMP_DIR"
    echo "   Please check disk space and permissions."
    exit 1
}

# Set environment variables for temp directory (BEFORE any Python imports)
export TMPDIR="$TEMP_DIR"
export TMP="$TEMP_DIR"
export TEMP="$TEMP_DIR"

# Verify temp directory is writable
if ! touch "$TEMP_DIR/.test_write" 2>/dev/null; then
    echo "❌ ERROR: Temp directory is not writable: $TEMP_DIR"
    exit 1
fi
rm -f "$TEMP_DIR/.test_write"

echo "✅ Using temp directory: $TEMP_DIR"

# Change to emoji_mirror directory
cd "$SCRIPT_DIR"

# Run Streamlit
streamlit run app.py

