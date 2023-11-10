# Prompt evaluations for the Manubot AI Editor

This repository contains code and frameworks used to evaluate a set of prompts used by
the Manubot AI Editor to revise scientific manuscripts.

## Setup

### Install software requirements
1. Install [Miniconda](https://docs.conda.io/en/latest/miniconda.html).
1. Create conda environment:
    ```bash
    conda env create -f environment.yml
    ```
1. Install [Ollama](https://ollama.ai/).
1. Start Ollama server:
   ```bash
   ollama serve
   ```
1. Install [promptfoo](https://promptfoo.dev/):
   ```bash
   npm install -g promptfoo
   ```
1. Install this package in editable mode (needs to be done only once):

    ```bash
    pip install -e .
    ```

### Pull local large language models

In Ollama, you need to pull the models that you want to use:

 ```bash
# Mistral 7b is fast and seems to perform relatively well.
# Here we pull the fp16 version (the largest in size).
ollama pull mistral:7b-instruct-fp16

# We could also pull smaller models, e.g.:
# ollama pull mistral:7b-instruct-q5_K_M

# TODO: add https://huggingface.co/TheBloke/ShiningValiant-1.2-GGUF
#  This is the model I was trying and seems to work pretty well.
#  It is possible to download the gguf file and import it locally into the Ollama
#  server (see https://github.com/jmorganca/ollama#import-from-gguf).
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

## Run evaluations

```bash
# cache results and use 1 job at a time
promptfoo eval -j 1

# do not cache results; useful when testing prompts
promptfoo eval --no-cache -j 1

# clear cache
promptfoo cache clear
```

By default, the evaluations are run locally using open source models.
To use a different local model or a remote 

## Visualize results
```bash
promptfoo view
```
