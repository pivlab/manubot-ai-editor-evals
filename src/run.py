import re
import argparse
import subprocess
import time
from pathlib import Path
from dotenv import load_dotenv


# model names and repetitions
models = {
    "ollama=mistral=7b-instruct-fp16": 5,
    "ollama=mixtral=8x7b-instruct-v0.1-q8_0": 5,
    "ollama=starling-lm=7b-alpha-fp16": 5,
    "openai=gpt-3.5-turbo-0125": 5,
    "openai=gpt-4-0613": 5,
    "openai=gpt-4-0125-preview": 5,
}


load_dotenv()


# extract parts of model name
def split_model(model):
    id, repeat = model
    prefix, name = id.split("=", maxsplit=1)
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

script_dir = Path(__file__).parent.resolve()

# run in pull mode
if args.pull:
    for model in models:
        if model["prefix"] != "ollama":
            continue
        model_name = model["name"].replace("=", ":")
        command = f"ollama pull {model_name}"
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
        # the maximum concurrent API calls should be one for local models, and could
        #  be increased (or use the default value) for remote models
        max_concurrent_arg = "-j 1"
        if model["prefix"] == "openai":
            max_concurrent_arg = ""

        provider = f"exec:python {str(script_dir)}/llm.py --model {model['id']}"

        # --verbose \
        command = f"""
        promptfoo eval \
            {max_concurrent_arg} \
            --no-cache \
            --repeat {model['repeat']} \
            --providers "{provider}" \
            -o outputs/{model['path']}/output/latest.html \
            -o outputs/{model['path']}/output/latest.csv \
            -o outputs/{model['path']}/output/latest.txt \
            -o outputs/{model['path']}/output/latest.json \
            -o outputs/{model['path']}/output/latest.yaml
        """.strip()
        print(re.sub(r"\s{2,}", " \\\n  ", command))
        subprocess.run(command, shell=True)
