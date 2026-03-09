# Contributing to MiniGPT-Forge

<p align="center">
  <img src="https://static.wixstatic.com/media/68ad1b_96c952149d584505bdfc30a3cbf36795~mv2.jpg" width="60" alt="MiniGPT-Forge"/>
</p>

Thank you for your interest in contributing to **MiniGPT-Forge**! This document provides guidelines for contributing to this project.

## Getting Started

1. **Fork** the repository on [GitHub](https://github.com/prajitdatta/MiniGPT-Forge)
2. **Clone** your fork locally:
   ```bash
   git clone https://github.com/<your-username>/MiniGPT-Forge.git
   cd MiniGPT-Forge
   ```
3. **Install** development dependencies:
   ```bash
   pip install -e ".[dev]"
   ```
4. **Create a branch** for your feature:
   ```bash
   git checkout -b feature/your-feature-name
   ```

## Development Workflow

### Running Tests

```bash
make test        # Full test suite with coverage
make test-fast   # Quick test run (stop on first failure)
```

### Code Quality

```bash
make lint        # Run ruff + mypy
make format      # Auto-format with black + ruff
```

### Code Style

We use **Black** for formatting and **Ruff** for linting. Key conventions:

- Line length: 100 characters
- Type hints for all public functions
- Docstrings for all public classes and methods (Google style)
- Meaningful variable names

## What to Contribute

- Bug fixes with test coverage
- Performance improvements with benchmarks
- New model architectures (e.g., Llama, Mistral variants)
- Documentation improvements
- Training recipes and example configs
- Dataset preprocessing scripts

## Pull Request Process

1. Ensure all tests pass: `make test`
2. Ensure code is formatted: `make format`
3. Update documentation if needed
4. Write a clear PR description explaining what and why
5. Reference any related issues

## Reporting Issues

Use our [issue templates](https://github.com/prajitdatta/MiniGPT-Forge/issues/new/choose) for:
- **Bug reports**: Include reproduction steps, expected vs actual behavior
- **Feature requests**: Describe the use case and proposed solution

## Code of Conduct

Please read our [Code of Conduct](CODE_OF_CONDUCT.md) before contributing.

---

<p align="center">
  Maintained by <a href="https://github.com/prajitdatta">Prajit Datta</a> •
  <a href="https://prajitdatta.github.io/">Website</a>
</p>
