#!/bin/bash

# check uv is installed
if ! command -v uv &> /dev/null; then
    echo "uv could not be found"
    echo "use brew or 'curl -LsSf https://astral.sh/uv/install.sh | sh' to install uv"
    exit 1
fi

# Create virtual environment if it doesn't exist
if ! uv venv; then
    echo "venv could not be created"
    exit 1
fi

# Install dependencies from pyproject.toml
if ! uv pip install -e .; then
    echo "Dependencies from pyproject.toml could not be installed"
    exit 1
fi

echo "Setup complete"
echo
echo "Run python scripts with 'uv run main.py'"
echo
exit 0
