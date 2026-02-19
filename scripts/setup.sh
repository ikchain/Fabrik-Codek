#!/bin/bash
# Fabrik-Codek Setup Script
# This script sets up the local AI dev assistant

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

echo "╔══════════════════════════════════════════════════════════╗"
echo "║     Fabrik-Codek Setup - Claude's Little Brother 🤖      ║"
echo "╚══════════════════════════════════════════════════════════╝"
echo ""

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

print_status() {
    echo -e "${GREEN}✓${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}!${NC} $1"
}

print_error() {
    echo -e "${RED}✗${NC} $1"
}

# Check Python version
echo "Checking Python..."
if command -v python3.11 &> /dev/null; then
    PYTHON=python3.11
elif command -v python3 &> /dev/null; then
    PYTHON=python3
    version=$($PYTHON --version 2>&1 | cut -d' ' -f2 | cut -d'.' -f1,2)
    if [[ "$version" < "3.11" ]]; then
        print_error "Python 3.11+ required, found $version"
        exit 1
    fi
else
    print_error "Python 3 not found"
    exit 1
fi
print_status "Python: $($PYTHON --version)"

# Check Ollama
echo ""
echo "Checking Ollama..."
if command -v ollama &> /dev/null; then
    print_status "Ollama installed"

    # Check if Ollama is running
    if curl -s http://localhost:11434/api/tags > /dev/null 2>&1; then
        print_status "Ollama server running"
    else
        print_warning "Ollama not running. Start with: ollama serve"
    fi
else
    print_warning "Ollama not installed"
    echo "  Install: curl -fsSL https://ollama.com/install.sh | sh"
fi

# Create virtual environment
echo ""
echo "Setting up virtual environment..."
cd "$PROJECT_ROOT"

if [ ! -d ".venv" ]; then
    $PYTHON -m venv .venv
    print_status "Virtual environment created"
else
    print_status "Virtual environment exists"
fi

# Activate and install
source .venv/bin/activate
print_status "Virtual environment activated"

echo ""
echo "Installing dependencies..."
pip install --upgrade pip -q
pip install -e ".[dev]" -q
print_status "Dependencies installed"

# Create .env if not exists
if [ ! -f ".env" ]; then
    cp .env.example .env 2>/dev/null || cat > .env << 'EOF'
# Fabrik-Codek Configuration
FABRIK_OLLAMA_HOST=http://localhost:11434
FABRIK_DEFAULT_MODEL=qwen2.5-coder:7b
FABRIK_FLYWHEEL_ENABLED=true
FABRIK_LOG_LEVEL=INFO
EOF
    print_status ".env file created"
fi

# Download models if Ollama is available
echo ""
echo "Checking models..."
if curl -s http://localhost:11434/api/tags > /dev/null 2>&1; then
    # Check if model exists
    if ! ollama list | grep -q "qwen2.5-coder:7b"; then
        print_warning "Downloading qwen2.5-coder:7b (this may take a while)..."
        ollama pull qwen2.5-coder:7b
        print_status "Model downloaded"
    else
        print_status "qwen2.5-coder:7b available"
    fi

    # Check embedding model
    if ! ollama list | grep -q "nomic-embed-text"; then
        print_warning "Downloading nomic-embed-text..."
        ollama pull nomic-embed-text
        print_status "Embedding model downloaded"
    else
        print_status "nomic-embed-text available"
    fi
else
    print_warning "Ollama not running, skipping model download"
    echo "  After starting Ollama, run:"
    echo "    ollama pull qwen2.5-coder:7b"
    echo "    ollama pull nomic-embed-text"
fi

# Create data directories
mkdir -p data/{raw,processed,embeddings}
mkdir -p data/raw/{interactions,training_pairs}
print_status "Data directories created"

echo ""
echo "╔══════════════════════════════════════════════════════════╗"
echo "║                    Setup Complete! 🎉                    ║"
echo "╚══════════════════════════════════════════════════════════╝"
echo ""
echo "Quick start:"
echo "  source .venv/bin/activate"
echo "  fabrik status       # Check system status"
echo "  fabrik chat         # Start interactive chat"
echo "  fabrik datalake     # Explore connected datalakes"
echo ""
echo "Flywheel data will be captured automatically while you work."
echo ""
