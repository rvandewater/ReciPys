# Contributing to ReciPies

Thank you for your interest in contributing! We welcome all contributions to improve ReciPies.

## How to Contribute

1. **Fork the repository** and create your branch from `development`.
2. **Install dependencies** using:
   ```
   pip install -e '.[dev]'
   ```
   or use the provided `environment.yml` for conda environments. 
For docs, use: 
   ```
   pip install -e '.[docs]'
   ```

3. **Run tests** before submitting changes:
   ```
   pytest
   ```

4. **Lint your code** with [Ruff](https://github.com/astral-sh/ruff):
   ```
   ruff src/ tests/
   ```

5. **Document your changes**. Update docstrings and, if needed, the documentation in `docs/`.

6. **Submit a pull request** with a clear description of your changes.

## Guidelines

- Follow [PEP8](https://peps.python.org/pep-0008/) style.
- Write tests for new features or bug fixes.
- Keep pull requests focused and concise.
- Be respectful and collaborative.

## Reporting Issues

- Use [GitHub Issues](https://github.com/rvandewater/ReciPies/issues) for bugs, feature requests, or questions.
- Provide as much detail as possible.

## Code of Conduct

Please be kind and inclusive. See [CODE_OF_CONDUCT.md](CODE_OF_CONDUCT.md) if available.

