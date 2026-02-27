#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# setup.sh — One-shot Linux installer for Eco-Material Property Predictor
# Usage: bash setup.sh
# Handles Debian/Ubuntu where python3-full may be missing (ensurepip issue)
# ─────────────────────────────────────────────────────────────────────────────
set -e

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="$PROJECT_DIR/venv"

echo "════════════════════════════════════════════════════════════════"
echo "  🌿  Eco-Material Property Predictor — Setup"
echo "════════════════════════════════════════════════════════════════"
echo ""

# ── Check Python 3 ──────────────────────────────────────────────────────────
if ! command -v python3 &>/dev/null; then
  echo "❌  python3 not found. Please install Python 3.9+ first:"
  echo "    sudo apt install python3 python3-full -y"
  exit 1
fi

PYTHON_VER=$(python3 -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
echo "✅  Python $PYTHON_VER detected"

# ── Create virtual environment (with pip) ───────────────────────────────────
echo ""
echo "📦  Creating virtual environment at: $VENV_DIR"

# Remove stale/broken venv if pip is missing from it
if [ -d "$VENV_DIR" ] && [ ! -f "$VENV_DIR/bin/pip" ]; then
  echo "⚠️   Existing venv has no pip — removing and rebuilding..."
  rm -rf "$VENV_DIR"
fi

# Try standard venv first
if python3 -m venv "$VENV_DIR" 2>/dev/null && [ -f "$VENV_DIR/bin/pip" ]; then
  echo "✅  venv created with pip"
else
  # venv was created but pip is missing — common on Debian without python3-full
  echo "⚠️   pip not available in venv. Trying to install python3-full..."
  if command -v apt-get &>/dev/null; then
    sudo apt-get install -y python3-full python3-pip -q 2>/dev/null || true
  fi
  rm -rf "$VENV_DIR"
  python3 -m venv "$VENV_DIR"
  if [ ! -f "$VENV_DIR/bin/pip" ]; then
    # Last resort: bootstrap pip manually via get-pip.py
    echo "📥  Bootstrapping pip via ensurepip..."
    "$VENV_DIR/bin/python3" -m ensurepip --upgrade 2>/dev/null || \
    "$VENV_DIR/bin/python3" -c "import urllib.request; exec(urllib.request.urlopen('https://bootstrap.pypa.io/get-pip.py').read())"
  fi
fi

# ── Install dependencies ─────────────────────────────────────────────────────
echo "📥  Installing dependencies..."
"$VENV_DIR/bin/pip" install --upgrade pip -q
"$VENV_DIR/bin/pip" install -r "$PROJECT_DIR/requirements.txt" -q

# ── Create required directories ──────────────────────────────────────────────
mkdir -p "$PROJECT_DIR/models" "$PROJECT_DIR/results" "$PROJECT_DIR/data/processed"

echo ""
echo "════════════════════════════════════════════════════════════════"
echo "  ✅  Setup complete!"
echo "════════════════════════════════════════════════════════════════"
echo ""
echo "  Activate your environment:  source venv/bin/activate"
echo "  Train the model:            make train"
echo "  Evaluate performance:       make evaluate"
echo "  Run predictions CLI:        make predict"
echo "  Run tests:                  make test"
echo ""
