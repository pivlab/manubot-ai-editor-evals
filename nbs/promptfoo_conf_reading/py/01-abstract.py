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
from glob import glob
from io import StringIO

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from proj.conf import abstract, introduction
from proj.promptfoo import read_results

# %% [markdown]
# # Get list of outputs for each model

# %%
INPUT_DIR = abstract.CASES / "phenoplier" / "outputs"
assert INPUT_DIR.exists
display(INPUT_DIR)

# %%
INPUT_MODELS = sorted(list(INPUT_DIR.glob("*")))

# %%
INPUT_MODELS

# %% [markdown]
# # Read all results

# %%
results = []
for model in INPUT_MODELS:
    model_results = read_results(model)
    results.append(model_results)

# %%
len(results)

# %%
results = pd.concat(results, ignore_index=True)

# %%
results.shape

# %%
results.head()

# %%
results["model"].value_counts()

# %%
results["prompt"].value_counts()

# %%
results["test_description"].value_counts()

# %%
results["comp_type"].value_counts()

# %%
results["comp_desc"].value_counts()

# %% [markdown]
# # Plot: pass rate

# %%
results.groupby(["model", "prompt"]).count()

# %% [markdown]
# ## Prompt: `candidate_with_metadata`

# %%
pass_rate = results[results["prompt"].isin(("candidate_with_metadata",))].groupby(["model"])["passed"].sum().to_frame()

# %%
pass_rate = pass_rate.assign(pass_rate=lambda x: x / 140.0)

# %%
pass_rate.shape

# %%
pass_rate.head()

# %%
pass_rate.sort_values("pass_rate")

# %%
sorted_models = pass_rate.sort_values("pass_rate").index.get_level_values("model").tolist()

# %%
sorted_models[-5:]

# %%
# sorted_models = list(dict.fromkeys(sorted_models))

# %%
# sorted_models[-5:]

# %%
g = sns.catplot(
    data=pass_rate,
    x="model",
    y="pass_rate",
    # hue="prompt",
    kind="bar",
    order=sorted_models,
    errorbar=None,
    aspect=2,
)
g.set_xticklabels(rotation=30, ha="right")
g.set(xlabel="Model", ylabel="Pass rate")

# %% [markdown]
# ## Prompt: `candidate`

# %%
pass_rate = results[results["prompt"].isin(("candidate",))].groupby(["model"])["passed"].sum().to_frame()

# %%
pass_rate = pass_rate.assign(pass_rate=lambda x: x / 140.0)

# %%
pass_rate.shape

# %%
pass_rate.head()

# %%
pass_rate.sort_values("pass_rate")

# %%
sorted_models = pass_rate.sort_values("pass_rate").index.get_level_values("model").tolist()

# %%
sorted_models[-5:]

# %%
# sorted_models = list(dict.fromkeys(sorted_models))

# %%
# sorted_models[-5:]

# %%
g = sns.catplot(
    data=pass_rate,
    x="model",
    y="pass_rate",
    # hue="prompt",
    kind="bar",
    order=sorted_models,
    errorbar=None,
    aspect=2,
)
g.set_xticklabels(rotation=30, ha="right")
g.set(xlabel="Model", ylabel="Pass rate")

# %% [markdown]
# # Plot: pass rate by prompt

# %%
results

# %%
results.groupby(["model", "prompt"]).count()

# %%
pass_rate = results[results["prompt"] != "baseline"].groupby(["model", "prompt"])["passed"].sum().to_frame()

# %%
pass_rate = pass_rate.assign(pass_rate=lambda x: x / 140.0).reset_index()

# %%
pass_rate.shape

# %%
pass_rate.head()

# %%
pass_rate.sort_values("pass_rate")

# %%
sorted_models = pass_rate[pass_rate["prompt"] == "candidate_with_metadata"].sort_values("pass_rate")["model"].tolist()

# %%
sorted_models[-5:]

# %%
# sorted_models = list(dict.fromkeys(sorted_models))

# %%
# sorted_models[-5:]

# %%
g = sns.catplot(
    data=pass_rate,
    x="model",
    y="pass_rate",
    hue="prompt",
    kind="bar",
    order=sorted_models,
    errorbar=None,
    height=4,
    aspect=2,
    legend_out=False,
)
g.set_xticklabels(rotation=30, ha="right")
g.set(xlabel="Model", ylabel="Pass rate")

leg = g.axes.flat[0].get_legend()
leg.set_title("")
new_labels = ['Prompt', 'Prompt + Manuscript metadata']
for t, l in zip(leg.texts, new_labels):
    t.set_text(l)

# %%
leg = g.axes.flat[0].get_legend()

# %%
leg.get_title()

# %% [markdown]
# # Plot: pass rate by test_description

# %%
results

# %%
results.groupby(["model", "prompt", "test_description"]).count()

# %% [markdown]
# ## Prompt: `candidate_with_metadata`

# %%
pass_rate = results[results["prompt"].isin(("candidate_with_metadata",))].groupby(["model", "test_description"])["passed"].sum().to_frame()

# %%
pass_rate = pass_rate.assign(pass_rate=lambda x: x / 35.0)

# %%
pass_rate.shape

# %%
pass_rate.head()

# %%
pass_rate.sort_values("pass_rate")

# %%
sorted_models = pass_rate.sort_values("pass_rate").index.get_level_values("model").tolist()

# %%
sorted_models[-5:]

# %%
sorted_models = list(dict.fromkeys(sorted_models))

# %%
sorted_models[-5:]

# %%
g = sns.catplot(
    data=pass_rate,
    x="model",
    y="pass_rate",
    hue="test_description",
    kind="bar",
    order=sorted_models,
    errorbar=None,
    aspect=2,
)
g.set_xticklabels(rotation=30, ha="right")

# %% [markdown]
# # Plot: score

# %%
results

# %% [markdown]
# ## Prompt: `candidate_with_metadata`

# %%
df = results[results["prompt"].isin(("candidate_with_metadata",))]#.groupby(["model"])["score_avg"].sum().to_frame()

# %%
df.shape

# %%
df.head()

# %%
df.groupby("model")["score_avg"].mean().sort_values()

# %%
sorted_models = df.groupby("model")["score_avg"].mean().sort_values().index.get_level_values("model").tolist()

# %%
sorted_models[-5:]

# %%
# sorted_models = list(dict.fromkeys(sorted_models))

# %%
# sorted_models[-5:]

# %%
g = sns.catplot(
    data=df,
    x="model",
    y="score_avg",
    # hue="prompt",
    kind="bar",
    order=sorted_models,
    # errorbar=None,
    aspect=2,
)
g.set_xticklabels(rotation=30, ha="right")

# %% [markdown]
# ## Prompt: `candidate`

# %%
df = results[results["prompt"].isin(("candidate",))]#.groupby(["model"])["score_avg"].sum().to_frame()

# %%
df.shape

# %%
df.head()

# %%
df.groupby("model")["score_avg"].mean().sort_values()

# %%
sorted_models = df.groupby("model")["score_avg"].mean().sort_values().index.get_level_values("model").tolist()

# %%
sorted_models[-5:]

# %%
# sorted_models = list(dict.fromkeys(sorted_models))

# %%
# sorted_models[-5:]

# %%
g = sns.catplot(
    data=df,
    x="model",
    y="score_avg",
    # hue="prompt",
    kind="bar",
    order=sorted_models,
    # errorbar=None,
    aspect=2,
)
g.set_xticklabels(rotation=30, ha="right")

# %% [markdown]
# # Plot: comp_score

# %%
results

# %% [markdown]
# ## Prompt: `candidate_with_metadata`

# %%
df = results[results["prompt"].isin(("candidate_with_metadata",))]#.groupby(["model"])["score_avg"].sum().to_frame()

# %%
df.shape

# %%
df.head()

# %%
df.groupby("model")["comp_score"].mean().sort_values()

# %%
sorted_models = df.groupby("model")["comp_score"].mean().sort_values().index.get_level_values("model").tolist()

# %%
sorted_models[-5:]

# %%
# sorted_models = list(dict.fromkeys(sorted_models))

# %%
df.loc[df["comp_type"] == "Formatting", "comp_score"] = df.loc[df["comp_type"] == "Formatting", "comp_score"] / 0.25
df.loc[df["comp_type"] == "Spelling/grammar", "comp_score"] = df.loc[df["comp_type"] == "Spelling/grammar", "comp_score"] / 2.0
df.loc[df["comp_type"] == "Structure", "comp_score"] = df.loc[df["comp_type"] == "Structure", "comp_score"] / 2.0

# %%
g = sns.catplot(
    data=df,
    x="model",
    y="comp_score",
    hue="comp_type",
    kind="bar",
    order=sorted_models,
    # errorbar=None,
    height=4,
    aspect=2,
    legend_out=True,
)
g.set_xticklabels(rotation=30, ha="right")
g.set(xlabel="Model", ylabel="Score")

# leg = g.axes.flat[0].get_legend()
# leg.set_title("")
g._legend.set_title("Test type")
new_labels = ['Formatting', 'Spelling/grammar', 'Structure (C-C-C)']
for t, l in zip(g._legend.texts, new_labels):
    t.set_text(l)

# %%
