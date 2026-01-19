# Contributing to Artifacta

Thank you for your interest in contributing to Artifacta! 🎉

## How to Contribute

1. **Report bugs or request features** - Open a [GitHub issue](https://github.com/walkerbdev/artifacta/issues)
2. **Fix bugs or add features** - Fork the repo, make changes, and submit a pull request
3. **Improve documentation** - Help make our docs clearer and more helpful

## Development Setup

```bash
# Clone and install
git clone https://github.com/walkerbdev/artifacta.git
cd artifacta
pip install -e .[dev]
npm install

# Run tests
pytest
npm run test

# Run linters
pre-commit run --all-files
```

## Pull Request Guidelines

- Write clear commit messages describing what and why
- Add tests for new features
- Update documentation if needed
- Run `pre-commit run --all-files` before submitting
- Keep PRs focused on a single change

## Code Style

We use automated tools to maintain code quality:
- **Python**: ruff, mypy, pydocstyle
- **JavaScript**: eslint
- Pre-commit hooks enforce these automatically

## Questions?

Open an issue or start a discussion. We're here to help!

## License

By contributing, you agree that your contributions will be licensed under the [Elastic License 2.0](LICENSE).
