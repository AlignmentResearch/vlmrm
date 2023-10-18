# Executed as separate process to avoid leaking system resources for long-running
# processes. Source:
# https://gitpython.readthedocs.io/en/stable/intro.html#leakage-of-system-resources

from pathlib import Path

from git import Repo

commit_hash = Repo(
    path=Path(__file__).parent.parent.parent.parent.resolve(),
    search_parent_directories=True,
).head.object.hexsha
