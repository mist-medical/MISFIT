"""Packaging guards for the release process.

The version lives in two hand-edited files — ``pyproject.toml`` (what PyPI and
the Docker publish workflow read) and ``misfit/__init__.py`` (what
``_build_config`` writes into every ``config.json``). Cutting a release bumps
both; this keeps them from drifting. Also checks that every declared console
script still resolves, and that every entrypoint module is runnable via
``python -m`` (the form ``torchrun -m misfit.cli.train_entrypoint`` and
container/Kubernetes manifests rely on).
"""
import re
import subprocess
import sys
from importlib.metadata import entry_points
from pathlib import Path

import pytest
from packaging.version import Version

import misfit

_PYPROJECT = Path(misfit.__file__).resolve().parent.parent / "pyproject.toml"

# One console script per CLI command — keep in sync with [project.scripts].
_EXPECTED_SCRIPTS = {
    "misfit_index",
    "misfit_train",
    "misfit_evaluate",
    "misfit_inspect",
    "misfit_encode",
    "misfit_embed",
    "misfit_embed_train",
}

# Every CLI is also runnable as `python -m misfit.cli.<name>_entrypoint` — this
# is what `torchrun -m ...` and shell-less container/Kubernetes manifests use
# instead of `$(which misfit_train)`.
_ENTRYPOINT_MODULES = [f"misfit.cli.{name.replace('misfit_', '')}_entrypoint" for name in sorted(_EXPECTED_SCRIPTS)]


def _pyproject_version() -> str:
    text = _PYPROJECT.read_text(encoding="utf-8")
    # First `version = "..."` line (under [project]).
    match = re.search(r'^\s*version\s*=\s*"([^"]+)"', text, re.MULTILINE)
    assert match, 'no `version = "..."` in pyproject.toml'
    return match.group(1)


@pytest.mark.skipif(not _PYPROJECT.exists(), reason="not a source checkout")
def test_version_matches_pyproject():
    assert Version(misfit.__version__) == Version(_pyproject_version())


def test_console_scripts_resolve():
    scripts = {
        ep.name: ep
        for ep in entry_points(group="console_scripts")
        if ep.name.startswith("misfit_")
    }
    assert set(scripts) == _EXPECTED_SCRIPTS
    for ep in scripts.values():
        assert callable(ep.load())  # imports the module + resolves the attr


@pytest.mark.parametrize("module", _ENTRYPOINT_MODULES)
def test_entrypoint_module_has_main_guard(module):
    """Every CLI module has an `if __name__ == "__main__"` block.

    Without it, `python -m misfit.cli.<x>_entrypoint` (and so
    `torchrun -m misfit.cli.train_entrypoint`) is a silent no-op — the module
    imports and exits without parsing args or doing anything. Containers and
    Kubernetes manifests use the `-m` form because there is no shell to
    evaluate `$(which misfit_train)`.
    """
    src = (Path(misfit.__file__).resolve().parent.parent / module.replace(".", "/")).with_suffix(
        ".py"
    ).read_text(encoding="utf-8")
    assert re.search(r'^if __name__ == ["\']__main__["\']:', src, re.MULTILINE), (
        f"{module} has no `if __name__ == \"__main__\"` block"
    )


def test_train_entrypoint_runs_under_dash_m():
    """End-to-end: `python -m misfit.cli.train_entrypoint --help` exits 0.

    This is the exact invocation `torchrun -m misfit.cli.train_entrypoint`
    launches on every worker. One module is enough to exercise the `-m` path;
    the guard check above covers the rest.
    """
    result = subprocess.run(
        [sys.executable, "-m", "misfit.cli.train_entrypoint", "--help"],
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.lower().startswith("usage:")
