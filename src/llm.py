import argparse

from langchain.chains import LLMChain
from langchain.prompts import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
)
from langchain_community.chat_models import ChatOllama
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic


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
