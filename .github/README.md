# TBR Package CI/CD Documentation

This directory contains GitHub Actions workflows for automated testing, quality assurance, and deployment of the TBR (Time-Based Regression) package.

## 📋 Workflows Overview

### 1. **CI Workflow** (`ci.yml`)
**Triggers**: Push to `main`/`develop`, Pull Requests
**Purpose**: Comprehensive testing and quality assurance

#### Jobs:
- **Code Quality**: Pre-commit hooks (linting, formatting, type checking)
- **Core Testing**: Unit, integration, and mathematical tests across multiple Python versions (3.8-3.11) and OS (Ubuntu, Windows, macOS)
- **Performance Testing**: Performance and scalability tests (separate job)
- **Package Building**: Test package build and installation
- **Comprehensive**: Full test suite with coverage reporting

#### Test Categories:
```bash
# Unit tests (fast, core functionality)
pytest tests/unit/ -v

# Integration tests (package imports, workflows)
pytest tests/integration/ -v

# Mathematical validation (against reference implementations)
pytest tests/mathematical/ -v

# Performance tests (scalability, timing)
pytest tests/performance/ -v
```

### 2. **Release Workflow** (`release.yml`)
**Triggers**: Git tags matching `v*` (e.g., `v1.0.0`, `v0.1.0a1`)
**Purpose**: Automated PyPI deployment

#### Process:
1. **Full Test Suite**: Run all tests across all supported platforms
2. **Build Package**: Create source distribution and wheel
3. **PyPI Deployment**: Publish to PyPI using API token
4. **GitHub Release**: Create GitHub release with notes

#### Usage:
```bash
# Create and push a release tag
git tag v0.1.0a1
git push origin v0.1.0a1
```

### 3. **Performance Monitoring** (`performance.yml`)
**Triggers**: Daily schedule (2 AM UTC), manual dispatch, changes to performance code
**Purpose**: Continuous performance monitoring and regression detection

#### Features:
- **Daily Monitoring**: Automated performance benchmarks
- **Cross-Platform**: Performance testing on Ubuntu, Windows, macOS
- **Artifact Storage**: Performance results saved for analysis
- **Manual Triggering**: On-demand performance testing

## 🚀 Getting Started

### Prerequisites
1. **Repository Secrets** (for PyPI deployment):
   - `PYPI_API_TOKEN`: PyPI API token for package publishing

### Local Development
```bash
# Install development dependencies
pip install -e .[dev]

# Run pre-commit hooks locally
pre-commit run --all-files

# Run specific test categories
pytest tests/unit/ -v              # Fast unit tests
pytest tests/integration/ -v       # Integration tests
pytest tests/mathematical/ -v      # Mathematical validation
pytest tests/performance/ -v       # Performance tests
pytest tests/ -v                   # All tests

# Run tests with coverage
pytest tests/ --cov=src/tbr --cov-report=html
```

### Test Markers
Our tests use pytest markers for organization:

```python
@pytest.mark.unit           # Unit tests
@pytest.mark.integration    # Integration tests
@pytest.mark.mathematical   # Mathematical validation
@pytest.mark.performance    # Performance tests
@pytest.mark.slow          # Slow-running tests
```

Run specific markers:
```bash
pytest -m "not performance"     # Skip performance tests (faster)
pytest -m "performance"         # Only performance tests
pytest -m "slow"               # Only slow tests
```

## 📊 Quality Gates

### CI Requirements (all must pass):
- ✅ **Code Quality**: All pre-commit hooks pass
- ✅ **Cross-Platform**: Tests pass on Ubuntu, Windows, macOS
- ✅ **Multi-Python**: Support Python 3.8, 3.9, 3.10, 3.11
- ✅ **Test Coverage**: Minimum 85% code coverage
- ✅ **Performance**: No significant performance regressions
- ✅ **Package Build**: Package builds and installs successfully

### Release Requirements:
- ✅ All CI requirements met
- ✅ Git tag follows semantic versioning (`v*`)
- ✅ PyPI API token configured
- ✅ Full test suite passes across all platforms

## 🔧 Maintenance

### Adding New Tests
1. **Unit Tests**: Add to `tests/unit/`
2. **Integration Tests**: Add to `tests/integration/`
3. **Mathematical Tests**: Add to `tests/mathematical/`
4. **Performance Tests**: Add to `tests/performance/` with `@pytest.mark.performance`

### Updating Workflows
- **Dependencies**: Update in `pyproject.toml`
- **Python Versions**: Update matrix in workflows
- **Test Commands**: Modify pytest commands as needed

### Monitoring Performance
- **Daily Reports**: Check GitHub Actions for daily performance runs
- **Artifacts**: Download performance results from workflow artifacts
- **Regression Detection**: Monitor performance trends over time

## 🎯 Best Practices

### Development Workflow:
1. **Local Testing**: Run `pytest tests/ -m "not performance"` for fast feedback
2. **Pre-Commit**: Hooks run automatically on commit
3. **Pull Requests**: Full CI runs on PR creation
4. **Performance**: Monitor via scheduled runs and manual triggers

### Release Process:
1. **Prepare Release**: Update version, changelog, documentation
2. **Tag Release**: Create and push version tag
3. **Automated Deployment**: Workflow handles PyPI publication
4. **Verify Release**: Check PyPI and GitHub releases

This CI/CD setup ensures high code quality, comprehensive testing, and reliable automated deployments for the TBR package.
