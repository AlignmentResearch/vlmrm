import datetime
import json
import os
import secrets
from pathlib import Path
from typing import Any, Dict


def get_run_hash() -> str:
    return (
        f'{datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")}_'
        f"{secrets.token_hex(4)}"
    )


def get_git_commit_hash() -> str:
    loc = {}
    # Executed as separate process to avoid leaking system resources for long-running
    # processes. Source:
    # https://gitpython.readthedocs.io/en/stable/intro.html#leakage-of-system-resources
    with open(Path(__file__).parent.resolve() / "git_commit_hash.py") as f:
        exec(f.read(), None, loc)
    return loc["commit_hash"]


def save_experiment_config(path: str, experiment_config: Dict) -> None:
    with open(path, "w") as f:
        json.dump(experiment_config, f, indent=2, cls=PathlibCompatibleJSONEncoder)


class PathlibCompatibleJSONEncoder(json.JSONEncoder):
    """JSON encoder that can handle pathlib objects."""

    def default(self, o: Any) -> Any:
        if isinstance(o, Path):
            return str(o)
        return super().default(o)


def set_egl_env_vars() -> None:
    os.environ["NVIDIA_VISIBLE_DEVICES"] = "all"
    os.environ["NVIDIA_DRIVER_CAPABILITIES"] = "compute,graphics,utility,video"
    os.environ["MUJOCO_GL"] = "egl"
    os.environ["PYOPENGL_PLATFORM"] = "egl"
    os.environ["EGL_PLATFORM"] = "device"
