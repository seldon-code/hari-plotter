import subprocess
import pytest


def test_examples():
    result = subprocess.run(
        ['tests/examples_check', 'run', '--all-files'], capture_output=True, text=True)
    assert result.returncode == 0, f"Pre-commit checks failed:\n{result.stdout}{result.stderr}"
