"""Tools for loading the configuration files associated with a run"""
from typing import Sequence
from pathlib import Path
import os


def load_variable(config_path: str | Path,
                  variable_name: str | Sequence[str],
                  working_dir: Path | None = None) -> object | list[object]:
    """Execute a Python file and retrieve variables within its scopy

    Args:
        config_path:
        variable_name: Name(s) of variable to be retrieved
        working_dir: Directory in which to execute Python file
    Returns:
        The target object
    """

    home_dir = Path.cwd()
    try:
        if working_dir is not None:
            os.chdir(working_dir)

        spec_ns = {}
        code = Path(config_path).read_text()
        obj = compile(code, filename=config_path, mode='exec')
        exec(obj, spec_ns)
        if isinstance(variable_name, str):
            if variable_name not in spec_ns:
                raise ValueError(f'Variable "{variable_name}" not found in {config_path}')
            return spec_ns[variable_name]
        else:
            for name in variable_name:
                if name not in spec_ns:
                    raise ValueError(f'Variable "{variable_name}" not found in {config_path}')
            return [spec_ns[s] for s in variable_name]
    finally:
        if working_dir is not None:
            os.chdir(home_dir)
