"""
This package gets user settings (from settings.py module or environment variables)
and creates the final configuration values.
This file is intended to be modified by project maintainers only.
"""

try:
    from dotenv import load_dotenv

    load_dotenv(override=True)
except ImportError:
    pass

import os
import types
from pathlib import PurePath

# make all main config modules available at this level
from proj.conf import common  # noqa: F401
from proj.conf.subconf import *  # noqa: F403


def generate_env_vars(conf_dict):
    """
    Generates environment variables and their values from a dictionary.
    The keys of the directory are variable names (strings), and values can be strings, integers, PurePath objects,
    other dictionaries, or modules. The function will recursively traverse the dictionary and generate environment
    variables for all the values that are not None and do not contain "__" in their name. For modules, it will make
    sure that they are from pvae.conf.
    """
    for var_name, var_value in conf_dict.items():
        if var_value is None or "__" in var_name:
            continue

        if isinstance(var_value, (str, int, PurePath)):
            new_var_name = f"{common.ENV_PREFIX}{var_name}"

            yield new_var_name, str(var_value)
        elif isinstance(var_value, dict):
            new_dict = {f"{var_name}_{k}": v for k, v in var_value.items()}
            for x in generate_env_vars(new_dict):
                yield x
        elif isinstance(var_value, types.ModuleType):
            if not var_value.__name__.startswith("proj.conf"):
                continue

            module_name = var_value.__name__.split(".")[-1]

            new_dict = {
                f"{module_name.upper()}_{k}": v for k, v in var_value.__dict__.items()
            }
            for x in generate_env_vars(new_dict):
                yield x


local_variables = dict(locals().items())

env_vars = dict(generate_env_vars(local_variables))

for k, v in env_vars.items():
    os.environ[k] = v
