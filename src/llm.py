import argparse

from langchain.chains import LLMChain
from langchain.prompts import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
)
from langchain_community.chat_models import ChatOllama
from langchain_openai import ChatOpenAI


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

    if model_type == "ollama":
        llm = ChatOllama(
            model=model,
            temperature=0,
            max_tokens=1024,
        )
    elif model_type == "openai":
        llm = ChatOpenAI(
            model_name=model,
            temperature=0,
            max_tokens=1024,
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
