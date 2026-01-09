# Contributing to Molecular Property Prediction

Thank you for your interest in contributing to this project! This document provides guidelines for contributing.

## Getting Started

1. **Fork the repository** on GitHub
2. **Clone your fork** locally:
   ```bash
   git clone https://github.com/YOUR_USERNAME/molecular-property-prediction.git
   cd molecular-property-prediction
   ```
3. **Set up the development environment**:
   ```bash
   pip install -r requirements.txt
   pip install -e ".[dev]"
   pre-commit install
   ```

## Development Workflow

### Making Changes

1. Create a new branch for your feature:
   ```bash
   git checkout -b feature/your-feature-name
   ```

2. Make your changes and write tests

3. Run the test suite:
   ```bash
   make test
   ```

4. Run linting:
   ```bash
   make lint
   ```

5. Commit your changes:
   ```bash
   git add .
   git commit -m "Add your descriptive commit message"
   ```

### Code Style

- Follow PEP 8 guidelines
- Use type hints where appropriate
- Write docstrings for all public functions and classes
- Keep functions focused and small
- Use meaningful variable names

### Testing

- Write tests for new functionality
- Maintain or improve code coverage
- Tests should be in the `tests/` directory
- Use pytest for testing

### Commit Messages

Use clear, descriptive commit messages:
- Start with a verb (Add, Fix, Update, Remove)
- Keep the first line under 72 characters
- Reference issues when applicable

Examples:
- `Add SMILES validation utility`
- `Fix bug in fingerprint generation`
- `Update model training documentation`

## Pull Request Process

1. Update documentation if needed
2. Add tests for new functionality
3. Ensure all tests pass
4. Update CHANGELOG.md if applicable
5. Submit a pull request with a clear description

## Reporting Issues

When reporting issues, please include:
- Python version
- Operating system
- Steps to reproduce
- Expected vs actual behavior
- Error messages (if any)

## Code of Conduct

Please be respectful and constructive in all interactions.

## Questions?

Open an issue for any questions about contributing.
