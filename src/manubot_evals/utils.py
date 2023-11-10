import json
from langchain.chat_models import ChatOllama
from langchain.schema.messages import HumanMessage, SystemMessage


def eval_rubric(
    output: str,
    rubric: str,
    ollama_model: str = "llama2:7b-chat-fp16",
    max_attemps: int = 3,
) -> str:
    chat = ChatOllama(
        model=ollama_model,
        temperature=0.7,
        top_k=40,
        top_p=0.1,
    )

    # ruff: noqa: E501
    system_message = """You are grading output according to a user-specified rubric. If the statement in the rubric is true, then the output passes the test. You respond with a JSON object with this structure: {"pass": boolean, "score": float, "reason": string}.

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

    return t.content
