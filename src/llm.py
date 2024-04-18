import argparse
from pathlib import Path

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
    args = parser.parse_args()

    # setup the cache
    script_dir = Path(__file__).parent.resolve()
    cache_dir = script_dir / "cache"
    cache_dir.mkdir(exist_ok=True, parents=True)

    REP_NUM = 0
    repeat_runs_file = cache_dir / "repeat_runs.pkl"
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
    
    repeat_info = pd.concat([
        repeat_info,
        pd.DataFrame({"model": [args.model], "prompt": [args.prompt], "repeat": [REP_NUM]}),
    ], ignore_index=True)
    repeat_info.to_pickle(repeat_runs_file)
    
    default_cache_file = cache_dir / f"llm_cache-rep{REP_NUM}.db"
    set_llm_cache(SQLiteCache(database_path=str(default_cache_file)))

    # process arguments
    model = args.model
    q = args.prompt

    model_type, model = model.split("=", 1)
    model = model.replace("=", ":")

    TEMPERATURE=0.5
    MAX_TOKENS=1024

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

    # q = sys.argv[1]
    output = llm_chain.invoke(q)["text"]
    print(output)
