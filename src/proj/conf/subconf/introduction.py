from pathlib import Path

from proj.conf.common import CODE_DIR as __CODE_DIR
from proj.conf.common import ENV_PREFIX as __ENV_PREFIX

__ENV_PREFIX += f"{Path(__file__).stem.upper()}_"

# Input data
DIR = __CODE_DIR / "introduction"
CASES = DIR / "cases"
PROMPTS = DIR / "prompts"
