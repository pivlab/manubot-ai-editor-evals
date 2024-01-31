import re
import argparse
import subprocess
import time
from dotenv import load_dotenv


# model names and repetitions
models = {
    # "ollama:alfred:40b-1023-q5_1": 5,
    # "ollama:deepseek-llm:67b-chat-q5_1": 5,
    # "ollama:deepseek-llm:7b-chat-fp16": 5,
    # "ollama:llama2:13b-chat-q8_0": 5,
    # "ollama:llama2:70b-chat-q5_1": 5,
    # "ollama:llama2:7b-chat-fp16": 1,
    # "ollama:mistral:7b-instruct-fp16": 5,
    # "ollama:mistral:7b-instruct-v0.2-fp16": 5,
    # "ollama:mixtral:8x7b-instruct-v0.1-q5_1": 5,
    # "ollama:mixtral:8x7b-instruct-v0.1-q8_0": 5,
    # "ollama:neural-chat:7b-v3.1-fp16": 5,
    # "ollama:openchat:7b-v3.5-fp16": 5,
    # "ollama:phi:2.7b-chat-v2-fp16": 5,
    # "ollama:solar:10.7b-instruct-v1-q8_0": 5,
    # "ollama:starling-lm:7b-alpha-fp16": 5,
    # "openai:gpt-3.5-turbo-0613": 5,
    # "openai:gpt-3.5-turbo-1106": 5,
    "openai:gpt-4-0613": 1,
    # "openai:gpt-4-1106-preview": 5,
}


load_dotenv()


# extract parts of model name
def split_model(model):
    id, repeat = model
    prefix, name = id.split(":", maxsplit=1)
    path = name.replace(":", "-")
    return {
        "id": id,
        "prefix": prefix,
        "name": name,
        "path": path,
        "repeat": repeat,
    }


models = list(map(split_model, models.items()))


# parse arguments
parser = argparse.ArgumentParser(description="Run evaluations across different models")
parser.add_argument(
    "--pull",
    action="store_true",
    help="Download models with Ollama",
)
parser.add_argument(
    "--view",
    action="store_true",
    help="View evaluation results",
)
args = parser.parse_args()


# run in pull mode
if args.pull:
    for model in models:
        if model["prefix"] != "ollama":
            continue
        command = f"ollama pull {model['name']}"
        print(command)
        subprocess.run(command, shell=True)


# run in view mode
elif args.view:
    port = 8000
    processes = []

    # open preview
    for model in models:
        command = f"promptfoo view --yes --port {port} ./outputs/{model['path']}"
        port += 1
        print(command)
        processes.append(subprocess.Popen(command, shell=True))

    # stop previews and free up ports after webpages open
    time.sleep(2)
    for process in processes:
        process.terminate()

# run in evaluation mode
else:
    for model in models:
        command = f"""
        promptfoo eval \
        -j 1 \
        --no-cache \
        --repeat {model['repeat']} \
        --providers {model['id']} \
        -o outputs/{model['path']}/output/latest.html \
        -o outputs/{model['path']}/output/latest.csv \
        -o outputs/{model['path']}/output/latest.txt \
        -o outputs/{model['path']}/output/latest.json \
        -o outputs/{model['path']}/output/latest.yaml
        """.strip()
        print(re.sub(r"\s{2,}", " \\\n  ", command))
        subprocess.run(command, shell=True)
