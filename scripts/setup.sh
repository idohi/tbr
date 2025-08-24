#!/bin/bash

# TBR Package Development Environment Setup Script
# This script sets up the complete development environment for the TBR package

set -e  # Exit on any error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

print_header() {
    echo -e "\n${BLUE}================================${NC}"
    echo -e "${BLUE} TBR Package Development Setup${NC}"
    echo -e "${BLUE}================================${NC}\n"
}

# Check if we're in the right directory
check_directory() {
    if [[ ! -f "pyproject.toml" ]] || [[ ! -d "src/tbr" ]]; then
        print_error "This script must be run from the TBR project root directory"
        print_error "Make sure you're in the directory containing pyproject.toml"
        exit 1
    fi
}

# Check if pyenv is installed
check_pyenv() {
    print_status "Checking pyenv installation..."
    if ! command -v pyenv &> /dev/null; then
        print_error "pyenv is not installed. Please install pyenv first:"
        echo "  macOS: brew install pyenv"
        echo "  Linux: curl https://pyenv.run | bash"
        echo "  Then restart your shell and run this script again."
        exit 1
    fi
    print_success "pyenv is installed"
}

# Install Python version if needed
setup_python() {
    local python_version="3.11.9"
    print_status "Setting up Python ${python_version}..."

    if ! pyenv versions --bare | grep -q "^${python_version}$"; then
        print_status "Installing Python ${python_version}..."
        pyenv install ${python_version}
    else
        print_success "Python ${python_version} already installed"
    fi

    # Set local Python version
    pyenv local ${python_version}
    print_success "Set local Python version to ${python_version}"
}

# Create and activate virtual environment
setup_venv() {
    print_status "Setting up virtual environment..."

    if [[ -d ".venv" ]]; then
        print_warning "Virtual environment already exists. Removing old one..."
        rm -rf .venv
    fi

    python -m venv .venv
    print_success "Created virtual environment"

    # Activate virtual environment
    source .venv/bin/activate
    print_success "Activated virtual environment"

    # Upgrade pip
    print_status "Upgrading pip..."
    pip install --upgrade pip
    print_success "pip upgraded"
}

# Install dependencies
install_dependencies() {
    print_status "Installing dependencies..."

    # Install build tools first
    print_status "Installing build tools..."
    pip install build wheel setuptools

    # Install runtime dependencies
    if [[ -f "requirements.txt" ]]; then
        print_status "Installing runtime dependencies..."
        pip install -r requirements.txt
    fi

    # Install development dependencies
    if [[ -f "requirements-dev.txt" ]]; then
        print_status "Installing development dependencies..."
        pip install -r requirements-dev.txt
    fi

    # Install package in development mode
    print_status "Installing TBR package in development mode..."
    pip install -e .

    print_success "All dependencies installed"
}

# Verify installation
verify_installation() {
    print_status "Verifying installation..."

    # Test core imports
    python -c "import pandas, numpy, scipy, statsmodels; print('✅ Core dependencies working')"

    # Test development tools
    python -c "import pytest, black, ruff; print('✅ Development tools working')"

    # Test TBR package import
    python -c "import tbr; print('✅ TBR package importable')"

    print_success "Installation verified successfully!"
}

# Setup pre-commit hooks
setup_precommit() {
    print_status "Setting up pre-commit hooks..."

    if [[ -f ".pre-commit-config.yaml" ]]; then
        pre-commit install
        print_success "Pre-commit hooks installed"
    else
        print_warning "No .pre-commit-config.yaml found, skipping pre-commit setup"
    fi
}

# Print final instructions
print_final_instructions() {
    echo -e "\n${GREEN}🎉 Setup completed successfully!${NC}\n"
    echo -e "${BLUE}Next steps:${NC}"
    echo "1. Activate the virtual environment: ${YELLOW}source .venv/bin/activate${NC}"
    echo "2. Run tests: ${YELLOW}make test${NC} or ${YELLOW}pytest${NC}"
    echo "3. Format code: ${YELLOW}make format${NC} or ${YELLOW}black src tests${NC}"
    echo "4. Run linting: ${YELLOW}make lint${NC} or ${YELLOW}ruff check src tests${NC}"
    echo "5. Build package: ${YELLOW}make build${NC} or ${YELLOW}python -m build${NC}"
    echo ""
    echo -e "${BLUE}Development workflow:${NC}"
    echo "• Always activate venv: ${YELLOW}source .venv/bin/activate${NC}"
    echo "• Run ${YELLOW}make help${NC} to see all available commands"
    echo "• Check ${YELLOW}README.md${NC} for detailed documentation"
    echo ""
    echo -e "${GREEN}Happy coding! 🚀${NC}"
}

# Main execution
main() {
    print_header

    check_directory
    check_pyenv
    setup_python
    setup_venv
    install_dependencies
    verify_installation
    setup_precommit

    print_final_instructions
}

# Run main function
main "$@"
