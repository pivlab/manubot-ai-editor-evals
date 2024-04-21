import re
import argparse
import subprocess
import time
from pathlib import Path
from dotenv import load_dotenv


# model names and repetitions
N_REPS = 5
models = {
    "ollama=mistral=7b-instruct-fp16": N_REPS,
    "ollama=mistral=7b-instruct-v0.2-fp16": N_REPS,
    "ollama=mixtral=8x7b-instruct-v0.1-q8_0": N_REPS,
    "ollama=mixtral=8x22b-instruct-v0.1-q4_1": N_REPS,
    "ollama=starling-lm=7b-alpha-fp16": N_REPS,
    "ollama=starling-lm=7b-beta-fp16": N_REPS,
    "ollama=llama2=7b-chat-fp16": N_REPS,
    "ollama=gemma=2b-instruct-fp16": N_REPS,
    "openai=gpt-3.5-turbo-0125": N_REPS,  # gpt-3.5-turbo points to this one
    "openai=gpt-4-0613": N_REPS,  # gpt-4 points to this one
    "openai=gpt-4-turbo-2024-04-09": N_REPS,  # gpt-4-turbo points to this one
    "anthropic=claude-3-opus-20240229": N_REPS,
    "anthropic=claude-3-sonnet-20240229": N_REPS,
    "anthropic=claude-3-haiku-20240307": N_REPS,
}


load_dotenv()


# extract parts of model name
def split_model(model):
    id, repeat = model
    prefix, name = id.split("=", maxsplit=1)
    path = name.replace("=", ":").replace(":", "-")
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
parser.add_argument(
    "-j", "--max-concurrency",
    type=int,
    help="Max concurrency for promptfoo eval. Only set to a value greater than 1 if the"
         " cache is only read, not written to.",
)
parser.add_argument(
    "--update-cache",
    action=argparse.BooleanOptionalAction,
    help="Forces the LangChain to skip any cached response, hit the model, and then "
         "update the cache",
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
    # before starting, remove temporary repetition file in cache dir if exists
    script_dir = Path(__file__).parent.resolve()
    cache_dir = script_dir / "cache"
    repeat_runs_file = cache_dir / "repeat_runs.pkl"
    repeat_runs_file.unlink(missing_ok=True)

    for model in models:
        # the maximum concurrent API calls is one by default (needed for custom caching
        # to work properly)
        max_concurrent_arg = "-j 1"
        # if model["prefix"] == "openai":
        #     max_concurrent_arg = ""
        if args.max_concurrency is not None:
            max_concurrent_arg = f"-j {args.max_concurrency}"

        update_cache_arg = ""
        if args.update_cache:
            update_cache_arg = "--update-cache"

        provider = f"exec:python {str(script_dir)}/llm.py {update_cache_arg} --model {model['id']}"

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
