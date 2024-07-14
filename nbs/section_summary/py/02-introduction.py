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

# %%
import sys
sys.path.append('/Users/faisala/Checkouts/chai/pivlab/manubot-ai-editor-evals/src')

# %%
from proj.conf import introduction, outputs
from proj.promptfoo import read_results

# %% [markdown]
# # Settings

# %% [markdown]
# These settings depend on the manuscript section

# %%
TEST_TYPES = [
    "Spelling/grammar",
    "Formatting",
    "Information accuracy",
    "Structure",
]

# %%
OUTPUT_FIGURES_DIR = outputs.FIGURES
OUTPUT_FIGURES_DIR.mkdir(parents=True, exist_ok=True)
display(OUTPUT_FIGURES_DIR)

# %% [markdown]
# # Get list of outputs for each model

# %%
INPUT_DIR = introduction.CASES / "phenoplier" / "outputs"
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
# # Pass rate (`passed`, by promptfoo)

# %% [markdown]
# ## Stats

# %%
df = results.copy()

# %%
df.shape

# %%
df.head()

# %%
df["prompt"].value_counts()

# %%
df["test_description"].value_counts()

# %%
df["comp_type"].value_counts()

# %% [markdown]
# ## Test level

# %% [markdown]
# ### Sum and normalize `passed` by model and prompt

# %% [markdown]
# **Description:** This statistic computed below, based on the `passed` column, measures if the test (by model and prompt) as a whole passed or failed, which is computed by promptfoo.
# The test passes if all the assertions in it pass, so it's a stringent criteria.

# %%
passed_unique = results.groupby(["model", "prompt"])["passed"].count()

# %%
assert passed_unique.unique().shape[0] == 1
n_tests_per_group = float(passed_unique.unique()[0])
display(n_tests_per_group)

# %%
df = results.groupby(["model", "prompt"])["passed"].sum().to_frame().reset_index()

# %%
df.head()

# %%
df = df.assign(pass_rate=lambda x: x["passed"] / n_tests_per_group)

# %%
df.shape

# %%
df.head()

# %%
df.sort_values("pass_rate")

# %%
_df_stats = df["pass_rate"].describe()
display(_df_stats)
assert _df_stats["max"] <= 1.0

# %% [markdown]
# ### Plot by prompt

# %%
# sort models by mean pass_rate in candidate prompts only (not baseline)
sorted_models = (
    df[df["prompt"].isin(("candidate", "candidate_with_metadata"))]
    .groupby("model")["pass_rate"]
    .mean()
    .sort_values()
    .index.get_level_values("model")
    .tolist()
)
assert len(set(sorted_models)) == len(sorted_models)

# %%
sorted_models[-5:]

# %%
g = sns.catplot(
    data=df,
    x="model",
    y="pass_rate",
    hue="prompt",
    hue_order=["baseline", "candidate_with_metadata", "candidate"],
    kind="bar",
    order=sorted_models,
    # errorbar=None,
    aspect=2,
)
g.fig.suptitle("Pass rate by prompt\n(models sorted by\navg on candidate prompts)")
g.set_xticklabels(rotation=30, ha="right")
g.set(xlabel="Model", ylabel="Pass rate")

g._legend.set_title("Prompt")
new_labels = ["Baseline prompt", "Candidate prompt\n+ metadata", "Candidate prompt"]
for t, l in zip(g._legend.texts, new_labels):
    t.set_text(l)

# %%
# this list is to check whether results match with those shown in the promptfoo's web interface
df.groupby(["model", "prompt"])["pass_rate"].mean()

# %% [markdown]
# ## Assertion level and type

# %% [markdown]
# ### Sum and normalize `passed` by model and test type

# %% [markdown]
# **Description:** This statistic computed below, based on the `comp_pass` column, measures if the assertion (by model, prompt and assertion type/`comp_type`) inside each test passed or failed, which is computed by promptfoo.
# The previous plot used the `passed` column, which is by test, but this is is by assertions (which belong to a single test).
# The test the assertion belongs to fails if a single assertion in it fails.

# %%
passed_unique = results.groupby(["model", "prompt"])["comp_pass"].count()
display(passed_unique.head())
assert passed_unique.unique().shape[0] == 1

# %%
n_tests_per_group = passed_unique.unique()[0]
display(n_tests_per_group)

# %% [markdown]
# Understanding the above numbers:
# * 145 = 7 * 5 + 8 * 5 + 7 * 5 + 7 * 5 = 145

# %%
df = results.groupby(["model", "prompt"])["comp_pass"].sum().to_frame().reset_index()

# %%
df.head()

# %%
# for testing purposes
df[df["model"].str.contains("mixtral-8x22")]

# %%
df = df.assign(pass_rate=df.apply(lambda x: x["comp_pass"] / n_tests_per_group, axis=1))

# %%
df.shape

# %%
df.head()

# %%
df.sort_values("pass_rate")

# %% scrolled=true
_df_stats = df["pass_rate"].describe()
display(_df_stats)
assert _df_stats["max"] <= 1.0

# %% [markdown]
# ### Plot by prompt

# %%
# sort models by comp_score in candidate prompts only (not baseline)
sorted_models = (
    df[df["prompt"].isin(("candidate", "candidate_with_metadata"))]
    .groupby("model")["pass_rate"]
    .mean()
    .sort_values()
    .index.get_level_values("model")
    .tolist()
)
assert len(set(sorted_models)) == len(sorted_models)

# %%
sorted_models[-5:]

# %%
g = sns.catplot(
    data=df,
    x="model",
    y="pass_rate",
    hue="prompt",
    hue_order=["baseline", "candidate_with_metadata", "candidate"],
    kind="bar",
    order=sorted_models,
    # errorbar=None,
    aspect=2,
)
g.fig.suptitle(
    "Assertion pass rate by prompt (models sorted by avg on candidate prompts)",
    ha="left",
    x=0.11,
)
g.set_xticklabels(rotation=30, ha="right")
g.set(xlabel="Model", ylabel="Assertion pass rate")

g._legend.set_title("Prompt")
new_labels = ["Baseline prompt", "Candidate prompt\n+ metadata", "Candidate prompt"]
for t, l in zip(g._legend.texts, new_labels):
    t.set_text(l)

# %%
# this list is to check whether results match with those shown in the promptfoo's web interface
df.groupby(["model", "prompt"])["pass_rate"].mean()

# %% [markdown]
# ### Sum and normalize `comp_pass` by model and assertion type

# %% [markdown]
# **Description:** This statistic computed below, based on the `comp_pass` column, measures if the assertion (by model, prompt and assertion type/`comp_type`) inside each test passed or failed, which is computed by promptfoo.
# The test the assertion belongs to fails if a single assertion in it fails.

# %%
passed_unique = results.groupby(["model", "prompt", "comp_type"])["comp_pass"].count()
display(passed_unique)
assert passed_unique.unique().shape[0] == (len(TEST_TYPES) - 1)

# %%
n_tests_per_group = passed_unique.loc[
    passed_unique.index[0][0], passed_unique.index[0][1]
]
display(n_tests_per_group)

# %% [markdown]
# Understanding the above numbers:
# * Formatting: 120 -> 6 per test * 4 test descriptions * 5 repetitions = 120
# * Information accuracy: 10 -> 6 per test * 4 test descriptions * 5 repetitions = 120
# * Spelling/grammar: 5 -> 2 per test * 1 test descriptions * 5 repetitions = 10
# * Structure: 10 -> 1 per test * 2 test descriptions * 5 repetitions = 10

# %%
df = (
    results.groupby(["model", "prompt", "comp_type"])["comp_pass"]
    .sum()
    .to_frame()
    .reset_index()
)

# %%
df.head()

# %%
# for testing purposes
df[df["model"].str.contains("opus")]

# %%
df.apply(lambda x: x["comp_pass"] / n_tests_per_group[x["comp_type"]], axis=1)

# %%
df = df.assign(
    pass_rate=df.apply(
        lambda x: x["comp_pass"] / n_tests_per_group[x["comp_type"]], axis=1
    )
)

# %%
df.shape

# %%
df.head()

# %%
df.sort_values("pass_rate")

# %% scrolled=true
_df_stats = df["pass_rate"].describe()
display(_df_stats)
assert _df_stats["max"] <= 1.0

# %% [markdown]
# ### Plot by assertion type on `candidate` prompt

# %%
# sort models by comp_score in candidate prompt only (which is the best performing one in most models)
sorted_models = (
    df[df["prompt"].isin(("candidate",))]
    .groupby(["model", "comp_type"])["pass_rate"]
    .mean()
    .groupby("model")
    .mean()
    .sort_values()
    .index.get_level_values("model")
    .tolist()
)
assert len(set(sorted_models)) == len(sorted_models)

# %%
sorted_models[-5:]

# %%
g = sns.catplot(
    data=df[df["prompt"] == "candidate"],
    # data=df,
    # col="prompt",
    # col_wrap=1,
    x="model",
    y="pass_rate",
    hue="comp_type",
    hue_order=[
        "Spelling/grammar",
        "Formatting",
        "Structure",
    ],
    kind="bar",
    order=sorted_models,
    # errorbar=None,
    height=4,
    aspect=2,
    legend_out=True,
)
g.fig.suptitle(
    "Assertion pass rate by test type ('candidate' prompt only)",
    ha="left",
    x=0.11,
)
g.set_xticklabels(rotation=30, ha="right")
g.set(xlabel="Model", ylabel="Assertion pass rate")

g._legend.set_title("Test type")
new_labels = [
    "Spelling/grammar",
    "Formatting",
    "Structure\n(C-C-C)",
]
for t, l in zip(g._legend.texts, new_labels):
    t.set_text(l)

# %%
# for testing:
_tmp = (
    df[df["prompt"] == "candidate"].groupby(["model", "comp_type"])["pass_rate"].mean()
)

# %%
_tmp.index.get_level_values(0).unique()

# %%
_tmp.loc["starling-lm-7b-alpha-fp16"]

# %% [markdown]
# ### Plot by assertion type (no spelling/grammar) on `candidate` prompt

# %%
# sort models by score_avg in candidate prompts only (not baseline)
sorted_models = (
    df[
        df["prompt"].isin(("candidate",))
        & df["comp_type"].isin(("Formatting", "Structure"))
    ]
    .groupby(["model", "comp_type"])["pass_rate"]
    .mean()
    .groupby("model")
    .mean()
    .sort_values()
    .index.get_level_values("model")
    .tolist()
)
assert len(set(sorted_models)) == len(sorted_models)

# %%
sorted_models[-5:]

# %%
g = sns.catplot(
    data=df[df["prompt"] == "candidate"],
    x="model",
    y="pass_rate",
    hue="comp_type",
    hue_order=[
        # "Spelling/grammar",
        "Formatting",
        "Structure",
    ],
    kind="bar",
    order=sorted_models,
    errorbar=None,
    height=4,
    aspect=2,
    legend_out=True,
)
g.fig.suptitle(
    "Assertion pass rate by test type on 'candidate' prompt ('candidate' prompt only)",
    ha="left",
    x=0.11,
)
g.set_xticklabels(rotation=30, ha="right")
g.set(xlabel="Model", ylabel="Assertion pass rate")

g._legend.set_title("Test type")
new_labels = [
    # "Spelling/grammar",
    "Formatting",
    "Structure\n(C-C-C)",
]
for t, l in zip(g._legend.texts, new_labels):
    t.set_text(l)

# %% [markdown]
# ## Assertion level and description (`comp_desc`)

# %% [markdown]
# ### Sum and normalize `passed` by model and test type

# %% [markdown]
# **TODO**: agregate Formatting assertions? Otherwise, the plot is too cluttered.

# %%
passed_unique = results.groupby(["model", "prompt", "comp_desc"])["comp_pass"].count()
display(passed_unique)
assert passed_unique.unique().shape[0] == 2

# %%
n_tests_per_group = passed_unique.loc[
    passed_unique.index[0][0], passed_unique.index[0][1]
]
display(n_tests_per_group)

# %% [markdown]
# Understanding the above numbers:
# * 5 -> 1 per test * 1 test descriptions * 5 repetitions = 5
# * 20 -> 1 per test * 4 test descriptions * 5 repetitions = 20

# %%
df = (
    results.groupby(["model", "prompt", "comp_desc"])["comp_pass"]
    .sum()
    .to_frame()
    .reset_index()
)

# %%
df.head()

# %%
# for testing purposes
df[df["model"].str.contains("opus")]

# %%
df.apply(lambda x: x["comp_pass"] / n_tests_per_group[x["comp_desc"]], axis=1)

# %%
df = df.assign(
    pass_rate=df.apply(
        lambda x: x["comp_pass"] / n_tests_per_group[x["comp_desc"]], axis=1
    )
)

# %%
df.shape

# %%
df.head()

# %%
df.sort_values("pass_rate")

# %%
_df_stats = df["pass_rate"].describe()
display(_df_stats)
assert _df_stats["max"] <= 1.0

# %% [markdown]
# ### Plot by assertion on `candidate` prompt

# %%
# sort models by comp_score in candidate prompt only (which is the best performing one in most models)
sorted_models = (
    df[df["prompt"].isin(("candidate",))]
    .groupby(["model", "comp_desc"])["pass_rate"]
    .mean()
    .groupby("model")
    .mean()
    .sort_values()
    .index.get_level_values("model")
    .tolist()
)
assert len(set(sorted_models)) == len(sorted_models)

# %%
sorted_models[-5:]

# %%
g = sns.catplot(
    data=df[df["prompt"] == "candidate"],
    # data=df,
    # col="prompt",
    # col_wrap=1,
    x="model",
    y="pass_rate",
    hue="comp_desc",
    # hue_order=[
    #     "Spelling/grammar",
    #     "Formatting",
    #     "Structure",
    # ],
    kind="bar",
    order=sorted_models,
    # errorbar=None,
    height=5,
    aspect=2,
    legend_out=True,
)
# g.fig.suptitle(
#     "Assertion pass rate by test type on 'candidate' prompt (models sorted by avg on 'candidate' prompt)",
#     ha="left",
#     x=0.11,
# )
g.set_xticklabels(rotation=30, ha="right")
g.set(xlabel="Model", ylabel="Assertion pass rate")

# g._legend.set_title("Test type")
# new_labels = [
#     "Spelling/grammar",
#     "Formatting",
#     "Structure\n(C-C-C)",
# ]
# for t, l in zip(g._legend.texts, new_labels):
#     t.set_text(l)

# %% [markdown]
# # Average score (`score_avg`, by promptfoo)

# %% [markdown]
# ## Stats

# %%
results["test_description"].unique()

# %%
# used to normalize
MAX_SCORE = {
    "Has no spelling errors": 0.50,
    "Keeps most references to other articles and doesn't make them up": ((0.25 * 6) + 4)
    / 8.0,
    "starts with context": ((0.25 * 6) + 2) / 7.0,
    "ends with conclusion": ((0.25 * 6) + 2) / 7.0,
}

# %% [markdown]
# ## Test level

# %% [markdown]
# ### Normalize `score_avg`

# %%
df = results.copy()

# %%
df.shape

# %%
df.head()

# %%
df.groupby("model")["score_avg"].mean().sort_values()

# %%
df.apply(lambda x: x["score_avg"] / MAX_SCORE[x["test_description"]], axis=1)

# %%
df = df.assign(
    score_avg=df.apply(
        lambda x: x["score_avg"] / MAX_SCORE[x["test_description"]], axis=1
    )
)

# %%
_df_stats = df.groupby("test_description")["score_avg"].describe()
display(_df_stats)

# %%
_df_stats = df["score_avg"].describe()
display(_df_stats)
assert _df_stats["max"].max() <= 1.0

# %%
df.groupby("model")["score_avg"].mean().sort_values()

# %% [markdown]
# ### Plot by prompt

# %%
# sort models by score_avg in candidate prompts only (not baseline)
sorted_models = (
    df[df["prompt"].isin(("candidate", "candidate_with_metadata"))]
    .groupby("model")["score_avg"]
    .mean()
    .sort_values()
    .index.get_level_values("model")
    .tolist()
)
assert len(set(sorted_models)) == len(sorted_models)

# %%
sorted_models[-5:]

# %%
g = sns.catplot(
    data=df,
    x="model",
    y="score_avg",
    hue="prompt",
    hue_order=["baseline", "candidate_with_metadata", "candidate"],
    kind="bar",
    order=sorted_models,
    # errorbar=None,
    aspect=2,
)
g.fig.suptitle(
    "Standardized avg. score by prompt\n(models sorted by avg on candidate prompts)",
    ha="left",
    x=0.11,
)
g.set_xticklabels(rotation=30, ha="right")
g.set(xlabel="Model", ylabel="Avg. score")

g._legend.set_title("Test type")
new_labels = ["Baseline prompt", "Candidate prompt\n+ metadata", "Candidate prompt"]
for t, l in zip(g._legend.texts, new_labels):
    t.set_text(l)

# %% [markdown]
# # Score per assertion (`comp_score`)

# %% [markdown]
# ## Stats

# %%
df = results.copy()

# %%
df.shape

# %%
df.head()

# %%
df["prompt"].value_counts()

# %%
df["test_description"].value_counts()

# %%
df["comp_type"].value_counts()

# %% [markdown]
# ## Normalize `comp_score`

# %%
df.groupby("model")["comp_score"].mean().sort_values()

# %%
# Warning: this values depend on those assigned in the `promptfooconfig.yaml` file
df.loc[df["comp_type"] == "Formatting", "comp_score"] = (
    df.loc[df["comp_type"] == "Formatting", "comp_score"] / 0.25
)
df.loc[df["comp_type"] == "Spelling/grammar", "comp_score"] = (
    df.loc[df["comp_type"] == "Spelling/grammar", "comp_score"] / 2.0
)
df.loc[df["comp_type"] == "Structure", "comp_score"] = (
    df.loc[df["comp_type"] == "Structure", "comp_score"] / 2.0
)
# only in introduction:
df.loc[df["comp_type"] == "Information accuracy", "comp_score"] = (
    df.loc[df["comp_type"] == "Information accuracy", "comp_score"] / 2.0
)

# %%
df.groupby("model")["comp_score"].mean().sort_values()

# %%
_df_stats = df["comp_score"].describe()
display(_df_stats)
assert _df_stats["max"] <= 1.0

# %% [markdown]
# ## Plot by prompt

# %%
# sort models by comp_score in candidate prompts only (not baseline)
sorted_models = (
    df[df["prompt"].isin(("candidate", "candidate_with_metadata"))]
    .groupby("model")["comp_score"]
    .mean()
    .sort_values()
    .index.get_level_values("model")
    .tolist()
)
assert len(set(sorted_models)) == len(sorted_models)

# %%
sorted_models[-5:]

# %%
g = sns.catplot(
    data=df,
    x="model",
    y="comp_score",
    hue="prompt",
    hue_order=["baseline", "candidate_with_metadata", "candidate"],
    kind="bar",
    order=sorted_models,
    # errorbar=None,
    aspect=2,
)
g.fig.suptitle(
    "Assertion score by prompt\n(models sorted by\navg on candidate prompts)",
    ha="left",
    x=0.11,
)
g.set_xticklabels(rotation=30, ha="right")
g.set(xlabel="Model", ylabel="Assertion score")

g._legend.set_title("Prompt")
new_labels = ["Baseline prompt", "Candidate prompt\n+ metadata", "Candidate prompt"]
for t, l in zip(g._legend.texts, new_labels):
    t.set_text(l)

# %% [markdown]
# ## Plot by test type on `candidate` prompt

# %%
# sort models by score_avg in candidate prompts only (not baseline)
sorted_models = (
    df[df["prompt"].isin(("candidate",))]
    .groupby(["model", "comp_type"])["comp_score"]
    .mean()
    .groupby("model")
    .mean()
    .sort_values()
    .index.get_level_values("model")
    .tolist()
)
assert len(set(sorted_models)) == len(sorted_models)

# %%
sorted_models[-5:]

# %%
g = sns.catplot(
    data=df[df["prompt"] == "candidate"],
    x="model",
    y="comp_score",
    hue="comp_type",
    hue_order=[
        "Spelling/grammar",
        "Information accuracy",
        "Formatting",
        "Structure",
    ],
    kind="bar",
    order=sorted_models,
    # errorbar=None,
    height=4,
    aspect=2,
    legend_out=True,
)
g.fig.suptitle(
    "Assertion score by test type (models sorted by avg on 'candidate' prompt)",
    ha="left",
    x=0.11,
)
g.set_xticklabels(rotation=30, ha="right")
g.set(xlabel="Model", ylabel="Assertion score")

# leg = g.axes.flat[0].get_legend()
# leg.set_title("")
g._legend.set_title("Test type")
new_labels = [
    "Spelling/grammar",
    "Information accuracy",
    "Formatting",
    "Structure\n(C-C-C)",
]
for t, l in zip(g._legend.texts, new_labels):
    t.set_text(l)

# %% [markdown]
# ## Plot by test type (no spelling/grammar) + `candidate` prompt

# %%
# sort models by score_avg in candidate prompts only (not baseline)
sorted_models = (
    df[
        df["prompt"].isin(("candidate",))
        & df["comp_type"].isin(("Formatting", "Information accuracy", "Structure"))
    ]
    .groupby(["model", "comp_type"])["comp_score"]
    .mean()
    .groupby("model")
    .mean()
    .sort_values()
    .index.get_level_values("model")
    .tolist()
)
assert len(set(sorted_models)) == len(sorted_models)

# %%
sorted_models[-5:]

# %%
# figure dims in inches
fig_width = 4.25
fig_height = 8

USE_TITLE = False
USE_LEGEND = False

g = sns.catplot(
    data=df[df["prompt"] == "candidate"],
    y="model",
    x="comp_score",
    hue="comp_type",
    hue_order=[
        # "Spelling/grammar",
        "Formatting",
        "Information accuracy",
        "Structure",
    ],
    kind="bar",
    order=list(reversed(sorted_models)),
    # errorbar=None,
    # height=4,
    # aspect=2,
    height=fig_height,
    aspect=fig_width/fig_height,  # aspect ratio to get the correct width
    legend=USE_LEGEND,
    # legend_out=True,
)

if USE_TITLE:
    # adjust the top margin to make room for the title
    g.fig.subplots_adjust(top=0.95)

    g.fig.suptitle(
        "Assertion score by test type (models sorted by avg on 'candidate' prompt)",
        ha="left",
        x=0.11,
    )
g.set_xticklabels(rotation=30, ha="right")
g.set(ylabel="Model", xlabel="Assertion score", xlim=(0,1))

if USE_LEGEND:
    # leg = g.axes.flat[0].get_legend()
    # leg.set_title("")
    g._legend.set_title("Test type")
    new_labels = [
        # "Spelling/grammar",
        "Formatting",
        "Information accuracy",
        "Structure\n(C-C-C)",
    ]
    for t, l in zip(g._legend.texts, new_labels):
        t.set_text(l)

# %% scrolled=true
g.fig.savefig(
    OUTPUT_FIGURES_DIR / "introduction-assertion_score_by_test_type.png",
    bbox_inches="tight",
    dpi=300,
)

g.fig.savefig(
    OUTPUT_FIGURES_DIR / "introduction-assertion_score_by_test_type.svg",
    bbox_inches="tight",
    facecolor="white",
)
