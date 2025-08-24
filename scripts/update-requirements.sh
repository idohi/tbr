#!/bin/bash

# Update Requirements Script
# Compiles requirements.in files to generate locked requirements.txt files

set -e

# Colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}Updating requirements files...${NC}"

# Check if pip-tools is installed
if ! command -v pip-compile &> /dev/null; then
    echo "Installing pip-tools..."
    pip install pip-tools
fi

# Compile requirements.in to requirements.txt
echo -e "${BLUE}Compiling requirements.in...${NC}"
pip-compile --upgrade requirements.in

# Compile requirements-dev.in to requirements-dev.txt
echo -e "${BLUE}Compiling requirements-dev.in...${NC}"
pip-compile --upgrade requirements-dev.in

echo -e "${GREEN}✅ Requirements files updated successfully!${NC}"
echo ""
echo "Files updated:"
echo "  - requirements.txt (from requirements.in)"
echo "  - requirements-dev.txt (from requirements-dev.in)"
echo ""
echo "To install updated dependencies:"
echo "  pip install -r requirements.txt"
echo "  pip install -r requirements-dev.txt"
