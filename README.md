# Prompt evaluations for the Manubot AI Editor

This repository contains code and frameworks used to evaluate a set of prompts used by
the Manubot AI Editor to revise scientific manuscripts.

## Setup

### Install software requirements
1. Install [Miniconda](https://docs.conda.io/en/latest/miniconda.html).
1. Create conda environment:
    ```bash
    conda env create -f environment.yml
    conda activate manubot-ai-editor-evals
    ```
1. Install [promptfoo](https://promptfoo.dev/):
   ```bash
   npm install -g promptfoo
   ```
1. Install this package in editable mode (needs to be done only once):

    ```bash
    pip install -e .
    ```
1. (In a different terminal) Install [Ollama](https://ollama.ai/) and start Ollama server:
   ```bash
   ollama serve
   ```

### Pull local large language models

In Ollama, you need to pull the models that you want to use (from the [library](https://ollama.ai/library)).
In the current configuration (check out `src/run.py`), several models are used.
To pull one model in Ollama, you run:

```bash
ollama pull mistral:7b-instruct-fp16
```

### Configure access to remote large language models

OpenAI:
```bash
export OPENAI_API_KEY="YOUR_API_KEY"
```

Replicate:
```bash
export REPLICATE_API_TOKEN="YOUR_API_KEY"
```

## Evaluations

### Structure

Evaluations are organized in folders by manuscript section.
For example, for the `abstract` and the `introduction` sections, the structure could be:

```bash
├── abstract
│   ├── cases
│   │   └── phenoplier
│   │       ├── outputs
│   │       │   ├── gpt-4.json
│   │       │   ├── mistral-7b-instruct-fp16.json
│   │       └── promptfooconfig.yaml
│   └── prompts
│       ├── baseline.txt
│       └── candidate.txt
├── introduction
│   ├── cases
│   │   └── phenoplier
│   │       ├── outputs
│   │       │   ├── gpt-4.json
│   │       │   ├── mistral-7b-instruct-fp16.json
│   │       └── promptfooconfig.yaml
│   └── prompts
│       ├── baseline.txt
│       └── candidate.txt
```

Under each section, there are two subfolders: 1) `cases` and 2) `prompts`.
A case corresponds to text from an existing manuscript (journal article, preprint, etc) for testing;
in the above example, `phenoplier` corresponds to [this journal article](https://doi.org/10.1038/s41467-023-41057-4).
A case contains a [promptfoo](https://promptfoo.dev/) configuration file (`promptfooconfig.yaml`) with test cases and asserts, and an `outputs` folder with the results of the evaluations across different models.
The `prompts` folder contains the prompts to be evaluated for this manuscript section;
right now we are using a baseline prompt (which includes a basic intruction to revise a text) and a candidate prompt (which includes a more complex set of instructions).

### Run

Make sure your conda environment is activated:

```bash
conda activate manubot-ai-editor-evals
```

To run the tests for a specific section and case, move to the corresponding directory and run the `run.py` script.
For example, for the `abstract` section and the `phenoplier` case:

```bash
cd abstract/cases/phenoplier/
python ../../../src/run.py
```

The script `src/run.py` runs `promptfoo eval` internally.

If for any reason you need to clear `promptfoo`'s cache, you can use:
```bash
promptfoo cache clear
```

## Visualize results

TODO: this needs more explanation to visualize the results for a specific model.

```bash
promptfoo view
```
