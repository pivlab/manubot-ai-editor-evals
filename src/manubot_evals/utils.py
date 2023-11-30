import json
from langchain.chat_models import ChatOllama, ChatOpenAI
from langchain.schema.messages import HumanMessage, SystemMessage


def eval_rubric(
    output: str,
    rubric: str,
    # model: str = "mistral:7b-instruct-fp16",
    # model: str = "orca2:13b-q8_0",
    model: str = "openai:gpt-4",
    # model: str = "openai:gpt-4-1106-preview",
    max_attemps: int = 3,
) -> str:
    if model.startswith("openai:"):
        chat = ChatOpenAI(model_name=model[7:], temperature=0.1, max_tokens=250, model_kwargs={"top_p": 1.0})
    else:
        chat = ChatOllama(model=model, cache=False)

    # ruff: noqa: E501
    system_message = """You are a professional copyeditor specializing in structuring different sections of research articles. You are grading output according to a user-specified rubric. If the statement in the rubric is true, then the output passes the test with score=1.0. You respond with a JSON object with this structure: {"pass": boolean, "score": float, "reason": string}.

Examples:

Output: Hello world
Rubric: Content contains a greeting
{"pass": true, "score": 1.0, "reason": "the content contains the word 'world'"}

Output: Avast ye swabs, repel the invaders!
Rubric: Does not speak like a pirate
{"pass": false, "score": 0.0, "reason": "'avast ye' is a common pirate term"}
    """.strip()

    human_message = f"Output: {output}\nRubric: {rubric}"

    messages = [
        SystemMessage(content=system_message),
        HumanMessage(content=human_message),
    ]

    t = None
    count = 0
    while t is None and count < max_attemps:
        try:
            t = chat.invoke(messages)
            json.loads(t.content)
        except json.JSONDecodeError:
            t = None
        finally:
            count += 1

    assert (
        t is not None
    ), f"Failed to get a response from the chatbot ({count} attempts)"

    return t.content.strip()
