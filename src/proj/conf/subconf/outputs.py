from pathlib import Path

from proj.conf.common import CODE_DIR as __CODE_DIR
from proj.conf.common import ENV_PREFIX as __ENV_PREFIX


__ENV_PREFIX += f"{Path(__file__).stem.upper()}_"

DIR = __CODE_DIR / "outputs"
FIGURES = DIR / "figures"
