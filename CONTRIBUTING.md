# Contributing

Thank you for your interest in contributing to this project!

## Development Setup

1. Clone the repository:
```bash
git clone https://github.com/anonymous/eeag-learning.git
cd eeag-learning
```

2. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate
```

3. Install development dependencies:
```bash
pip install -e ".[dev]"
```

## Code Style

We follow PEP 8 guidelines. Please ensure your code:
- Uses 4 spaces for indentation
- Has docstrings for all public functions
- Includes type hints where appropriate

## Running Tests

```bash
pytest tests/ -v
```

## Adding New Experiments

1. Create a new experiment file in `experiments/`
2. Follow the existing experiment structure
3. Add a configuration file in `experiments/configs/`
4. Update `run_all.sh` to include the new experiment
5. Document results in `RESULTS.md`

## Submitting Changes

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Run tests to ensure nothing is broken
5. Submit a pull request

## Questions?

Open an issue on GitHub for any questions or concerns.
