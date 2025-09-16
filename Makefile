# TBR Package Development Makefile
# Professional development workflow automation

.PHONY: help setup clean install install-dev test test-cov lint format type-check docstring build upload docs serve-docs pre-commit all

# Default target
.DEFAULT_GOAL := help

# Variables
PYTHON := python
PIP := pip
PACKAGE_NAME := tbr
SRC_DIR := src
TEST_DIR := tests
DOCS_DIR := docs

# Colors for output
BLUE := \033[36m
GREEN := \033[32m
YELLOW := \033[33m
RED := \033[31m
NC := \033[0m # No Color

help: ## Show this help message
	@echo "$(BLUE)TBR Package Development Commands$(NC)"
	@echo "=================================="
	@echo ""
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "$(GREEN)%-20s$(NC) %s\n", $$1, $$2}'
	@echo ""
	@echo "$(YELLOW)Quick Start:$(NC)"
	@echo "  make setup     # First time setup"
	@echo "  make install   # Install package in dev mode"
	@echo "  make test      # Run tests"
	@echo "  make all       # Run full development pipeline"

setup: ## Run complete development environment setup
	@echo "$(BLUE)Setting up development environment...$(NC)"
	@./scripts/setup.sh

clean: ## Clean build artifacts and cache files
	@echo "$(BLUE)Cleaning build artifacts...$(NC)"
	@rm -rf build/
	@rm -rf dist/
	@rm -rf *.egg-info/
	@rm -rf .pytest_cache/
	@rm -rf .coverage
	@rm -rf htmlcov/
	@rm -rf .mypy_cache/
	@rm -rf .ruff_cache/
	@find . -type d -name __pycache__ -exec rm -rf {} +
	@find . -type f -name "*.pyc" -delete
	@echo "$(GREEN)✅ Cleaned successfully$(NC)"

install: ## Install package in development mode
	@echo "$(BLUE)Installing package in development mode...$(NC)"
	@$(PIP) install -e .
	@echo "$(GREEN)✅ Package installed$(NC)"

install-dev: ## Install development dependencies
	@echo "$(BLUE)Installing development dependencies...$(NC)"
	@$(PIP) install -e ".[dev,docs,examples]"
	@echo "$(GREEN)✅ Development dependencies installed$(NC)"



test: ## Run tests
	@echo "$(BLUE)Running tests...$(NC)"
	@pytest $(TEST_DIR) -v
	@echo "$(GREEN)✅ Tests completed$(NC)"

test-cov: ## Run tests with coverage report
	@echo "$(BLUE)Running tests with coverage...$(NC)"
	@pytest $(TEST_DIR) --cov=$(SRC_DIR)/$(PACKAGE_NAME) --cov-report=html --cov-report=term-missing -v
	@echo "$(GREEN)✅ Coverage report generated in htmlcov/$(NC)"

lint: ## Run linting checks
	@echo "$(BLUE)Running linting checks...$(NC)"
	@ruff check $(SRC_DIR) $(TEST_DIR)
	@echo "$(GREEN)✅ Linting completed$(NC)"

format: ## Format code with black and isort
	@echo "$(BLUE)Formatting code...$(NC)"
	@black $(SRC_DIR) $(TEST_DIR)
	@isort $(SRC_DIR) $(TEST_DIR)
	@echo "$(GREEN)✅ Code formatted$(NC)"

format-check: ## Check code formatting without making changes
	@echo "$(BLUE)Checking code formatting...$(NC)"
	@black --check $(SRC_DIR) $(TEST_DIR)
	@isort --check-only $(SRC_DIR) $(TEST_DIR)
	@echo "$(GREEN)✅ Format check completed$(NC)"

type-check: ## Run type checking with mypy
	@echo "$(BLUE)Running type checks...$(NC)"
	@mypy $(SRC_DIR)/$(PACKAGE_NAME)
	@echo "$(GREEN)✅ Type checking completed$(NC)"

docstring: ## Check docstring style and coverage
	@echo "$(BLUE)Checking docstring style and coverage...$(NC)"
	@pydocstyle $(SRC_DIR) --convention=numpy
	@interrogate $(SRC_DIR) --fail-under=90 -v
	@echo "$(GREEN)✅ Docstring validation completed$(NC)"

build: clean ## Build package for distribution
	@echo "$(BLUE)Building package...$(NC)"
	@$(PYTHON) -m build
	@echo "$(GREEN)✅ Package built successfully$(NC)"
	@echo "$(YELLOW)Built files:$(NC)"
	@ls -la dist/

upload-test: build ## Upload package to TestPyPI
	@echo "$(BLUE)Uploading to TestPyPI...$(NC)"
	@twine upload --repository testpypi dist/*
	@echo "$(GREEN)✅ Uploaded to TestPyPI$(NC)"

upload: build ## Upload package to PyPI (PRODUCTION)
	@echo "$(RED)⚠️  Uploading to PRODUCTION PyPI...$(NC)"
	@read -p "Are you sure? (y/N): " confirm && [ "$$confirm" = "y" ]
	@twine upload dist/*
	@echo "$(GREEN)✅ Uploaded to PyPI$(NC)"

docs: ## Generate documentation
	@echo "$(BLUE)Generating documentation...$(NC)"
	@cd $(DOCS_DIR) && make html
	@echo "$(GREEN)✅ Documentation generated$(NC)"

serve-docs: docs ## Serve documentation locally
	@echo "$(BLUE)Serving documentation at http://localhost:8000$(NC)"
	@cd $(DOCS_DIR)/_build/html && $(PYTHON) -m http.server 8000

pre-commit: ## Run pre-commit hooks on all files
	@echo "$(BLUE)Running pre-commit hooks...$(NC)"
	@pre-commit run --all-files
	@echo "$(GREEN)✅ Pre-commit hooks completed$(NC)"

install-pre-commit: ## Install pre-commit hooks
	@echo "$(BLUE)Installing pre-commit hooks...$(NC)"
	@pre-commit install
	@echo "$(GREEN)✅ Pre-commit hooks installed$(NC)"

check: lint type-check format-check docstring ## Run all code quality checks
	@echo "$(GREEN)✅ All checks passed$(NC)"

test-all: ## Run comprehensive test suite
	@echo "$(BLUE)Running comprehensive test suite...$(NC)"
	@pytest $(TEST_DIR) -v --cov=$(SRC_DIR)/$(PACKAGE_NAME) --cov-report=term-missing
	@echo "$(GREEN)✅ Comprehensive tests completed$(NC)"

test-tox: ## Run tests across multiple Python versions with tox
	@echo "$(BLUE)Running tests with tox...$(NC)"
	@tox
	@echo "$(GREEN)✅ Tox tests completed$(NC)"

test-tox-py: ## Run tests for current Python version only
	@echo "$(BLUE)Running tox for current Python version...$(NC)"
	@tox -e py$$(python -c "import sys; print(f'{sys.version_info.major}{sys.version_info.minor}')")
	@echo "$(GREEN)✅ Tox Python-specific tests completed$(NC)"

ci-local: ## Run exact CI tests locally (mirrors GitHub Actions)
	@echo "$(BLUE)Running CI pipeline locally...$(NC)"
	@echo "$(YELLOW)Step 1: Installing dependencies$(NC)"
	@python -m pip install --upgrade pip
	@pip install -e .[dev]
	@echo "$(YELLOW)Step 2: Running unit tests with coverage$(NC)"
	@pytest tests/unit/ -v --cov=src/tbr --cov-report=xml --cov-report=term-missing
	@echo "$(YELLOW)Step 3: Running integration tests$(NC)"
	@pytest tests/integration/ -v
	@echo "$(YELLOW)Step 4: Running mathematical validation tests$(NC)"
	@pytest tests/mathematical/ -v
	@echo "$(YELLOW)Step 5: Running performance tests$(NC)"
	@pytest tests/performance/ -v
	@echo "$(GREEN)✅ CI pipeline completed locally$(NC)"

all: clean install-dev check test-cov ## Run complete development pipeline
	@echo "$(GREEN)🎉 Complete development pipeline finished successfully!$(NC)"

# Development workflow targets
dev-setup: setup install-pre-commit ## Complete development setup
	@echo "$(GREEN)🚀 Development environment ready!$(NC)"

quick-check: format lint test ## Quick development check
	@echo "$(GREEN)✅ Quick check completed$(NC)"

release-check: clean check test-cov build ## Pre-release validation
	@echo "$(GREEN)✅ Release validation completed$(NC)"

# Environment info
info: ## Show environment information
	@echo "$(BLUE)Environment Information$(NC)"
	@echo "======================="
	@echo "Python: $$(python --version)"
	@echo "Pip: $$(pip --version)"
	@echo "Virtual Environment: $$VIRTUAL_ENV"
	@echo "Package Version: $$(python -c 'import $(PACKAGE_NAME); print($(PACKAGE_NAME).__version__)' 2>/dev/null || echo 'Not installed')"
	@echo "Working Directory: $$(pwd)"
