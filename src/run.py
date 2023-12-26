import argparse
import subprocess
from dotenv import load_dotenv

load_dotenv()

# model names and repetitions
models = {
    "ollama:starling-lm:7b-alpha-fp16": 5,
    "ollama:mistral:7b-instruct-fp16": 5,
    "ollama:mistral:7b-instruct-v0.2-fp16": 5,
    "ollama:mixtral:8x7b-instruct-v0.1-q8_0": 5,
    "ollama:mixtral:8x7b-instruct-v0.1-q5_1": 5,
    "ollama:deepseek-llm:7b-chat-fp16": 5,
    "ollama:neural-chat:7b-v3.1-fp16": 5,
    "ollama:openchat:7b-v3.5-fp16": 5,
    "ollama:deepseek-llm:67b-chat-q5_1": 5,
    "ollama:alfred:40b-1023-q5_1": 5,
    "ollama:solar:10.7b-instruct-v1-q8_0": 5,
    "ollama:phi:2.7b-chat-v2-fp16": 5,
    "openai:gpt-3.5-turbo": 5,
    "openai:gpt-4": 5,
    "openai:gpt-4-1106-preview": 5,
}

# Set up argument parser
parser = argparse.ArgumentParser(description="Run model evaluations and optionally download models.")
parser.add_argument('--download-models', action='store_true', help='Download specified models using ollama pull')
args = parser.parse_args()

if args.download_models:
    for model in models.keys():
        model_prefix, model_name = model.split(":", maxsplit=1)

        if model_prefix != "ollama":
            continue

        command = f"ollama pull {model_name}"
        print(command)
        subprocess.run(command, shell=True)
else:
    for model, repeat in models.items():
        model_name = "".join(model.split(":", 1)[1:]).replace(":", "-")
        command = f"promptfoo eval -j 1 --no-cache --repeat {repeat} --providers {model} -o outputs/{model_name}.html -o outputs/{model_name}.csv -o outputs/{model_name}.txt -o outputs/{model_name}.json -o outputs/{model_name}.yaml"
        print(command)
        subprocess.run(command, shell=True)
