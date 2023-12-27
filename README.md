# Prompt evaluations for the Manubot AI Editor

The Manubot AI Editor is a tool for [Manubot](https://manubot.org) that uses AI to help authors revise their manuscripts.
This repository contains code used to evaluate the effectiveness of the prompts (instructions to the AI) used in the tool.

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
   ```
1. Install the last tested [promptfoo](https://promptfoo.dev/) version:
   ```bash
   npm install -g promptfoo@0.33.3
   ```
1. Install this package in editable mode (only needs to be done once):
   ```bash
   pip install -e .
   ```
1. Install [Ollama](https://ollama.ai/).

### Pull local large language models

Before you can run models locally, you have to pull them with Ollama.
See the [library of models](https://ollama.ai/library) Ollama has available.

Make sure the Ollama server is running (e.g., `ollama serve`).

The current configuration of this tool (see `src/run.py`) lists a handful of models that can be used in test cases.
To pull all of them, run:

```bash
# activate the environment if you didn't
# conda activate manubot-ai-editor-evals

# download models
python src/run.py --download-models
```

### Configure access to remote large language models

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

### Structure

Evaluations are organized into folders by manuscript section.
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
│   ├── ...
```

Under each section, there are two subfolders: 1) `cases` and 2) `prompts`.

A case corresponds to text from an existing manuscript (journal article, preprint, etc.) for testing.
In the above example, `phenoplier` corresponds to [this journal article](https://doi.org/10.1038/s41467-023-41057-4).
A case contains a [promptfoo](https://promptfoo.dev/) configuration file (`promptfooconfig.yaml`) with test cases and assertions, and an `outputs` folder with the results of the evaluations across different models.

The `prompts` folder contains the prompts to be evaluated for this manuscript section.
At the moment, we are using 1) a candidate prompt containing a complex set of instructions and 2) a baseline prompt containing more basic instructions to compare the candidate prompt against.

### Run

Activate the conda environment:

```bash
conda activate manubot-ai-editor-evals
```

Start Ollama server, if not [already running automatically](https://github.com/jmorganca/ollama/issues/707):

```bash
ollama serve
```

Move to the directory of a specific section and case, then run the `run.py` script from there.
For example, for the `abstract` section and the `phenoplier` case:

```bash
cd abstract/cases/phenoplier/
python ../../../src/run.py
```

This will run `promptfoo eval` for you as appropriate.

If you need to clear `promptfoo`'s cache, you can run:

```bash
promptfoo cache clear
```

## Visualize results

promptfoo provides a convenient way to view and compare results:

```bash
promptfoo view
```

[See more here](https://www.promptfoo.dev/docs/usage/web-ui).

If you run the command above, promptfoo will read the latest run result set.
The way I found to specify which a specific result set (such as those for Mistral or GPT-4) is to copy a result set in a directory and run `promptfoo view` with that directory as argument.

First, setup the directory with the results you want to visualize:

```bash
# create directory for results
mkdir -p /tmp/promptfoo/output

# copy one result set as the "latest" in that directory
cp outputs/gpt-4-1106-preview.json /tmp/promptfoo/output/latest.json
```

Then you run `promptfoo view` with that directory as argument (you can run this only once, and `promptfoo` will automatically refresh the page as you copy a new result set to the directory above).

```bash
promptfoo view /tmp/promptfoo/
```
