# Prompt evaluations for the Manubot AI Editor

This repository contains code and frameworks used to evaluate a set of prompts used by
the Manubot AI Editor to revise scientific manuscripts.

## Setup
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
1. Pull the desired language model:
    ```bash
    # Xwin-LM seems to perform well (not tested yet)
    ollama pull xwinlm:70b-v0.1-q6_K
    
    # TODO: add https://huggingface.co/TheBloke/ShiningValiant-1.2-GGUF
    #  This is the model I was trying and seems to work pretty well.
    #  It is possible to download the gguf file and import it locally into the Ollama
    #  server (see https://github.com/jmorganca/ollama#import-from-gguf).
    ```
1. Install [promptfoo](https://promptfoo.dev/):
   ```bash
   npm install -g promptfoo
   ```
