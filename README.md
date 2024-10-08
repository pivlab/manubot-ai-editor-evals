# A benchmark framework for large language models (LLMs) on scholarly manuscript revision

This repository contains code used to evaluate the effectiveness of prompts and LLMs in the context of scholarly manuscript revision.
Initially, the goal of the evaluations is to improve the prompts used in the [Manubot AI Editor]([url](https://github.com/manubot/manubot-ai-editor)), which is a tool for [Manubot](https://manubot.org) that uses AI to help authors revise their manuscripts automatically

Under-the-hood, it uses:

- promptfoo for test configuration, running evaluations, and presenting comparisons.
- Ollama for managing local models.
- Python for basic scripting and coordination.

## Setup

### Install software requirements

1. Install [Miniconda](https://docs.conda.io/en/latest/miniconda.html).
1. Create conda environment:
   ```bash
   conda env create -f environment.yml
   conda activate manubot-ai-editor-evals
   ```
1. Install the last tested [promptfoo](https://promptfoo.dev/) version:
   ```bash
   npm install -g promptfoo@0.47.0
   ```
1. Install this package in editable mode:
   ```bash
   pip install -e .
   ```
1. Install [Ollama](https://ollama.ai/). The latest version we tested is [v0.1.32](https://github.com/ollama/ollama/releases/tag/v0.1.32), which in Linux (amd64) you can install with:
   ```bash
   sudo curl -L https://github.com/ollama/ollama/releases/download/v0.1.32/ollama-linux-amd64 -o /usr/bin/ollama
   sudo chmod +x /usr/bin/ollama
   ```

### Start required processes

1. Activate the conda environment if haven't already:
   ```bash
   conda activate manubot-ai-editor-evals
   ```
1. Start Ollama in a different terminal (no need to activate the conda environment), if not [already running automatically](https://github.com/jmorganca/ollama/issues/707):
   ```bash
   ollama serve
   ```

### Select models

promptfoo supports a [large selection of models from different providers](https://www.promptfoo.dev/docs/providers).
This tool lists a handful of select models in `src/run.py`, focusing on OpenAI ChatGPT and local models with [Ollama](https://ollama.ai/library).

This list is what is used when running the script commands below.
To add other models from promptfoo/Ollama, include their [ids](https://www.promptfoo.dev/docs/providers/ollama) here.
To select specific models for a pull/eval/view, you can comment/uncomment their entries.

### Pull local models

Before you can run models locally, you have to pull them with Ollama.

```bash
python src/run.py --pull
```

### Configure access to remote models

Provide an API key for the service you wish to use as an environment variable:

In **.env** file:

```
API_KEY_NAME="API_KEY_VALUE"
```

or in CLI:

```bash
export API_KEY_NAME="API_KEY_VALUE"
```

| Service   | API_KEY_NAME        |
| --------- | ------------------- |
| OpenAI    | OPENAI_API_KEY      |
| Replicate | REPLICATE_API_TOKEN |

(Per [promptfoo docs](https://www.promptfoo.dev/docs/providers))

## Evaluations

Evaluations are organized into folders by manuscript section.
For example, for the `abstract` and the `introduction` sections, the structure could be:

```bash
├── abstract
│   ├── cases
│   │   └── phenoplier
│   │       ├── inputs
│   │       ├── outputs
│   │       └── promptfooconfig.yaml
│   └── prompts
│       ├── baseline.txt
│       └── candidate.txt
├── introduction
│   ├── ...
```

Under each section, there are two subfolders: 1) `cases` and 2) `prompts`.

A case corresponds to text from an existing manuscript (journal article, preprint, etc.) for testing.
In the above example, `phenoplier` corresponds to [this journal article](https://doi.org/10.1038/s41467-023-41057-4).
A case contains a [promptfoo](https://promptfoo.dev/) configuration file (`promptfooconfig.yaml`) with test cases and assertions, and an `outputs` folder with the results of the evaluations across different models.

The `prompts` folder contains the prompts to be evaluated for this manuscript section.
At the moment, we are using 1) a candidate prompt containing a complex set of instructions and 2) a baseline prompt containing more basic instructions to compare the candidate prompt against.

## Usage

First, move to the directory of the section and case of interest.
Then run the `src/run.py` script from there.
For example, for the `abstract` section and the `phenoplier` case:

```bash
cd abstract/cases/phenoplier/
python ../../../src/run.py
```

### Run evaluation

Running the script without flags runs your evaluations.

```bash
python ../../../src/run.py
```

By default, all queries to the models are cached in `src/cache/*.db` (SQLite) for
faster and cheaper subsequent runs.

### Visualize results

To explore the results of your evaluations across *all* models in a web UI table, run:

```bash
python ../../../src/run.py --view
```

If you are interested only in a specific model such as `gpt-3.5-turbo-0613`, run:

```bash
promptfoo view outputs/gpt-3.5-turbo-0125/
```

[See more here](https://www.promptfoo.dev/docs/usage/web-ui).

### Misc

If you need to clear `promptfoo`'s cache, you can run:

```bash
promptfoo cache clear
```

## Advanced

### SQLite cache

In case the cache files located in `src/cache/*.db` (SQLite) need to be updated, you
can open the `.db` file with `sqlite3`:

```bash
sqlite3 src/cache/llm_cache-rep0.db
```

#### Updating cached queries

You can run queries to update the cache, such as:

```sql
-- Update the model name for a specific prompt
UPDATE full_llm_cache
SET llm = replace(llm, 'mixtral-8x22-fix', 'mixtral:8x22b-instruct-v0.1-q5_1' )
WHERE llm LIKE '%mixtral-8x22%';
```

#### Deleting old entries

To delete certain entries (such as old/previous models not used anymore):

```sql
DELETE FROM full_llm_cache
WHERE llm LIKE "%('model', 'mixtral:8x22b-instruct-v0.1-q4_1')%";
```

#### Vacuuming

From the terminal:

```bash
sqlite3 src/cache/llm_cache-rep0.db "VACUUM;"
```
