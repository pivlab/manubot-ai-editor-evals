import argparse
from pathlib import Path
import fcntl

import pandas as pd
from langchain.chains import LLMChain
from langchain.prompts import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
)
from langchain_community.chat_models import ChatOllama
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from langchain.cache import SQLiteCache
from langchain.globals import set_llm_cache


class WriteOnlyCache(SQLiteCache):
    def lookup(self, prompt: str, llm_string: str):
        return None


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="LLM")
    parser.add_argument(
        "--model",
        type=str,
        help="Model",
    )
    parser.add_argument(
        "prompt",
        type=str,
        help="TODO",
    )
    parser.add_argument(
        "config",
        type=str,
        nargs="?",
        help="TODO",
    )
    parser.add_argument(
        "vars",
        type=str,
        nargs="?",
        help="TODO",
    )
    parser.add_argument(
        "--update-cache",
        action=argparse.BooleanOptionalAction,
        help="Forces the LangChain to skip any cached response, hit the model, and then"
        " update the cache",
    )
    args = parser.parse_args()

    # setup the cache
    script_dir = Path(__file__).parent.resolve()
    cache_dir = script_dir / "cache"
    cache_dir.mkdir(exist_ok=True, parents=True)

    REP_NUM = 0
    repeat_runs_file = cache_dir / "repeat_runs.pkl"
    with open(repeat_runs_file.with_suffix(".lock"), "w+") as g:
        fcntl.flock(g, fcntl.LOCK_EX)

        if not repeat_runs_file.exists():
            repeat_info = pd.DataFrame(columns=["model", "prompt", "repeat"])
        else:
            repeat_info = pd.read_pickle(repeat_runs_file)

        run_info = repeat_info[
            (repeat_info["model"] == args.model)
            & (repeat_info["prompt"] == args.prompt)
        ]
        if run_info.shape[0] > 0:
            REP_NUM = run_info["repeat"].max() + 1

        repeat_info = pd.concat(
            [
                repeat_info,
                pd.DataFrame(
                    {
                        "model": [args.model],
                        "prompt": [args.prompt],
                        "repeat": [REP_NUM],
                    }
                ),
            ],
            ignore_index=True,
        )
        repeat_info.to_pickle(repeat_runs_file)

        fcntl.flock(g, fcntl.LOCK_UN)

    default_cache_file = cache_dir / f"llm_cache-rep{REP_NUM}.db"

    cache_obj = None
    if args.update_cache:
        cache_obj = WriteOnlyCache(database_path=str(default_cache_file))
    else:
        cache_obj = SQLiteCache(database_path=str(default_cache_file))
    set_llm_cache(cache_obj)

    # process arguments
    model = args.model
    q = args.prompt

    model_type, model = model.split("=", 1)
    model = model.replace("=", ":")

    TEMPERATURE = 0.5
    MAX_TOKENS = 1024

    if model_type == "ollama":
        llm = ChatOllama(
            model=model,
            temperature=TEMPERATURE,
            max_tokens=MAX_TOKENS,
        )
    elif model_type == "openai":
        llm = ChatOpenAI(
            model_name=model,
            temperature=TEMPERATURE,
            max_tokens=MAX_TOKENS,
        )
    elif model_type == "anthropic":
        llm = ChatAnthropic(
            model=model,
            temperature=TEMPERATURE,
            max_tokens=MAX_TOKENS,
        )

    prompt = ChatPromptTemplate(
        messages=[
            # SystemMessagePromptTemplate.from_template(system_message),
            HumanMessagePromptTemplate.from_template("{question}"),
        ]
    )

    llm_chain = LLMChain(llm=llm, prompt=prompt)

    output = llm_chain.invoke(q)["text"]

    # fix for starling models
    output = output.replace("<|end_of_turn|>", "")

    print(output)
