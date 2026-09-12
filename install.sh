#!/usr/bin/env bash
# ==============================================================================
# 🎩 NanoHat 2.1 — One-Line Installer for Fedora & Linux
# ==============================================================================
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "============================================================"
echo " 🎩 Installing NanoHat 2.1: Sub-1B Fedora OS Agent"
echo "============================================================"

# 1. Check Python
if ! command -v python3 &> /dev/null; then
    echo "❌ python3 not found. Please install Python 3.10+ first."
    exit 1
fi

# 2. Check Ollama
if ! command -v ollama &> /dev/null; then
    echo "⚠️  Ollama binary not found in PATH."
    echo "   Install Ollama from https://ollama.com before running."
else
    echo "✓ Ollama detected."
    # Check if Ollama daemon is running
    if curl -s http://127.0.0.1:11434/api/tags &> /dev/null; then
        echo "✓ Ollama service is running."
        # Check if nanohat2.1:360m exists
        if ollama list 2>/dev/null | grep -q "nanohat2.1:360m"; then
            echo "✓ Model 'nanohat2.1:360m' already available in Ollama."
        elif [ -f "$SCRIPT_DIR/Modelfile" ] && [ -f "$SCRIPT_DIR/NanoHat-360m/nanohat2.1.F16.gguf" ]; then
            echo "⚙️  Creating Ollama model 'nanohat2.1:360m' from local GGUF..."
            ollama create nanohat2.1:360m -f "$SCRIPT_DIR/Modelfile"
            echo "✓ Model 'nanohat2.1:360m' created successfully."
        fi
    else
        echo "⚠️  Ollama daemon is not active. Run 'ollama serve' to enable local inference."
    fi
fi

# 3. Install Python package
echo "⚙️  Installing nanohat Python CLI package..."
pip install -e . --no-warn-script-location

echo ""
echo "============================================================"
echo " 🎉 NanoHat 2.1 Installed Successfully!"
echo "============================================================"
echo " Try running:"
echo "   nanohat \"What is 15 percent of 800?\""
echo "   nanohat \"What is my current RAM usage?\""
echo "   nanohat \"Check battery status\""
echo "============================================================"
