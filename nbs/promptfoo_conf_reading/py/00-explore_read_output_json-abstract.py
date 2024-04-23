# ---
# jupyter:
#   jupytext:
#     cell_metadata_filter: all,-execution,-papermill,-trusted
#     notebook_metadata_filter: -jupytext.text_representation.jupytext_version
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Modules

# %%
import json
from io import StringIO
import pandas as pd
from proj.conf import abstract, introduction

# %% [markdown]
# # Explore `latest.json`

# %%
abstract.CASES

# %%
INPUT_FILE = abstract.CASES / "phenoplier" / "outputs" / "mixtral-8x22b-instruct-v0.1-q4_1" / "output" / "latest.json"

# %%
data = pd.read_json(INPUT_FILE)

# %%
data.shape

# %%
data

# %%
with StringIO(json.dumps(data.iloc[1, 0])) as f:
    r = pd.read_json(f)

# %%
r.shape

# %%
r.head()

# %%
r.loc[0]

# %%
r.loc[0, "provider"]

# %%
r.loc[0, "prompt"]

# %%
r.loc[0, "vars"]

# %%
r.loc[0, "response"]

# %%
r.loc[0, "gradingResult"]

# %%
