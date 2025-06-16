# Contributing to AKAB

Thank you for your interest in contributing to AKAB! This document provides guidelines and instructions for contributing to the project.

## 🤝 Code of Conduct

By participating in this project, you agree to maintain a respectful and inclusive environment for all contributors.

## 🚀 Getting Started

1. **Fork the repository** on GitHub
2. **Clone your fork** locally:
   ```bash
   git clone https://github.com/yourusername/akab.git
   cd akab
   ```
3. **Create a virtual environment**:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```
4. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

## 🔧 Development Setup

### Running Locally

1. Copy `.env.example` to `.env` and configure
2. Start the development server:
   ```bash
   python -m akab.server
   ```

### Running with Docker

```bash
docker compose up --build
```

### Running Tests

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=akab --cov-report=html

# Run specific test file
pytest tests/test_providers.py
```

## 📝 Development Guidelines

### Code Style

- Follow PEP 8 Python style guide
- Use type hints where appropriate
- Maximum line length: 88 characters (Black default)
- Format code with Black: `black src/`
- Lint with Ruff: `ruff src/`

### Commit Messages

Follow conventional commits format:

- `feat:` New feature
- `fix:` Bug fix
- `docs:` Documentation changes
- `style:` Code style changes (formatting, etc.)
- `refactor:` Code refactoring
- `test:` Test additions or changes
- `chore:` Maintenance tasks

Example: `feat: add Mistral provider support`

### Branch Naming

- `feature/description` - New features
- `fix/description` - Bug fixes
- `docs/description` - Documentation updates
- `refactor/description` - Code refactoring

## 🎯 What to Contribute

### High Priority

1. **New Providers**: Add support for more LLM providers
2. **Evaluation Metrics**: Improve or add new evaluation metrics
3. **Documentation**: Improve docs, add examples, fix typos
4. **Bug Fixes**: Check issues labeled "bug"
5. **Tests**: Increase test coverage

### Feature Ideas

- Parallel experiment execution
- Advanced visualization tools
- Export formats (CSV, Excel, etc.)
- Web UI for management
- Plugin system for custom evaluations

## 🔄 Pull Request Process

1. **Create a feature branch** from `main`
2. **Make your changes** following the guidelines
3. **Add/update tests** as needed
4. **Update documentation** if applicable
5. **Run tests and linting** locally
6. **Submit a pull request** with clear description

### PR Checklist

- [ ] Code follows project style guidelines
- [ ] Tests pass locally
- [ ] Documentation updated if needed
- [ ] Commit messages follow format
- [ ] PR description explains changes

## 🏗️ Architecture Notes

### Adding a New Provider

1. Create provider class in `src/akab/providers/`
2. Inherit from `Provider` base class
3. Implement required methods:
   - `check_availability()`
   - `execute()`
   - `estimate_cost()`
4. Add to `ProviderManager._initialize_providers()`
5. Add tests in `tests/test_providers.py`

Example:
```python
class MistralProvider(Provider):
    def __init__(self, model: str = "mistral-large"):
        super().__init__(f"mistral/{model}", ProviderType.REMOTE)
        # Implementation...
```

### Adding a New Evaluation Metric

1. Add method to `EvaluationEngine` class
2. Follow naming pattern: `calculate_[metric_name]`
3. Return numeric score or structured data
4. Add to `self.metrics` in `__init__`
5. Document the metric's purpose

### Adding a New MCP Tool

1. Add method to `AKABTools` class
2. Decorate with `@mcp.tool()` in `server.py`
3. Follow naming pattern: `akab_[action]`
4. Return consistent response format
5. Update meta prompt if needed

## 📊 Testing Guidelines

### Test Structure

```
tests/
├── unit/           # Unit tests for individual components
├── integration/    # Integration tests
└── fixtures/       # Test data and fixtures
```

### Writing Tests

- Use pytest fixtures for setup
- Test both success and error cases
- Mock external API calls
- Use meaningful test names

Example:
```python
def test_openai_provider_cost_estimation():
    provider = OpenAIProvider("gpt-3.5-turbo")
    cost = provider.estimate_cost(1000, 500)
    assert cost == 0.0023  # Expected cost
```

## 🐛 Reporting Issues

### Bug Reports

Include:
- Python version
- OS (Windows/Linux/Mac)
- Steps to reproduce
- Expected vs actual behavior
- Error messages/logs

### Feature Requests

Include:
- Use case description
- Proposed solution
- Alternative approaches
- Impact on existing features

## 📚 Documentation

### Where to Document

- **Code**: Docstrings for all public methods
- **README**: High-level features and setup
- **User Journeys**: Detailed usage examples
- **API Docs**: Tool and provider references
- **Architecture**: Design decisions and patterns

### Documentation Style

- Clear and concise
- Include code examples
- Explain the "why" not just "what"
- Keep updated with code changes

## 🙏 Recognition

Contributors will be:
- Listed in CONTRIBUTORS.md
- Mentioned in release notes
- Thanked in project announcements

## 💬 Getting Help

- Check existing issues and discussions
- Join our Discord server
- Tag maintainers in complex PRs
- Ask in PR comments if unsure

## 🚢 Release Process

1. Maintainers review and merge PRs
2. Changes accumulated for release
3. Version bumped following semver
4. Release notes prepared
5. Docker images built and pushed
6. GitHub release created

---

Thank you for contributing to AKAB! Your efforts help make AI experimentation more accessible and scientific for everyone. 🎉
