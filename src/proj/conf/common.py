import os
from pathlib import Path

import proj

ENV_PREFIX = "PROJ_"

# CODE_DIR points to the base directory where the code is
CODE_DIR = Path(proj.__file__).parent.parent.parent.resolve()

NBS_DIR = CODE_DIR / "nbs"

# N_JOBS
options = [
    (
        m
        if (m := os.environ.get(f"{ENV_PREFIX}N_JOBS")) is not None and m.strip() != ""
        else None
    ),
    1,
]
N_JOBS = next(int(opt) for opt in options if opt is not None)
