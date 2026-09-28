"""Shared test settings. The package is imported from src/ (see pythonpath in
pyproject.toml), so the tests run without installing it."""

import os

REPO_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
