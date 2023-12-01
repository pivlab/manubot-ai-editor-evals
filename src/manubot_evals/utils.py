import json
from pathlib import Path
import tempfile

from langchain.globals import set_llm_cache
from langchain.cache import SQLiteCache

set_llm_cache(
    SQLiteCache(database_path=str(Path(tempfile.gettempdir()) / "langchain.db"))
)

from langchain.chains import LLMChain
from langchain.chat_models import ChatOllama, ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.prompts import (
    ChatPromptTemplate,
    MessagesPlaceholder,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)
from langchain.prompts import PromptTemplate


def eval_rubric(
    text: str,
    title: str,
    rubric: str,
    model_name: str = "openai:gpt-4",
    max_attemps: int = 1,
) -> str:
    if model_name.startswith("openai:"):
        model = ChatOpenAI(
            model_name=model_name[7:],
            temperature=0.2,
            max_tokens=250,
            model_kwargs={
                "top_p": 1.0,
            },
        )
    else:
        model = ChatOllama(model=model_name, temperature=0.2, max_tokens=250, top_p=1.0)

    summary_prompt = PromptTemplate.from_template("Summarize this abstract: {text}")
    context_prompt = PromptTemplate.from_template(
        "What is the context of this research (i.e., the broader field of research)?"
    )
    research_gap_prompt = PromptTemplate.from_template(
        "What is the research gap that this manuscript will answer?"
    )
    proposed_method_prompt = PromptTemplate.from_template(
        "What is the proposed method or approach to answer the research gap?"
    )
    results_prompt = PromptTemplate.from_template(
        "What are the concrete results obtained using the proposed method or approach?"
    )
    conclusion_prompt = PromptTemplate.from_template(
        "What is the conclusion that interprets the results to answer the research gap?"
    )
    conclusion2_prompt = PromptTemplate.from_template(
        "How does the conclusion move the broader field forward?"
    )
    rubric_prompt = PromptTemplate.from_template(
        'You are grading this text according to the rubric below. If the statement in the rubric is true, then the text passes the test. You respond with a JSON object with this structure: {"pass": boolean, "score": float, "reason": string}.\n\nRubric: {{ rubric }}\nJSON: ',
        template_format="jinja2",
    )

    messages = [
        summary_prompt,
        context_prompt,
        research_gap_prompt,
        proposed_method_prompt,
        results_prompt,
        conclusion_prompt,
        conclusion2_prompt,
        # rubric_mode_prompt,
        rubric_prompt,
    ]

    prompt = ChatPromptTemplate(
        messages=[
            SystemMessagePromptTemplate.from_template(
                "You are a scientist with copyediting skills, and you are having a conversation with a human interested in understanding a scientific manuscript titled '{title}'."
            ).format(title=title),
            # The `variable_name` here is what must align with memory
            MessagesPlaceholder(variable_name="chat_history"),
            HumanMessagePromptTemplate.from_template("{question}"),
        ]
    )

    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
    conversation = LLMChain(
        llm=model,
        prompt=prompt,
        verbose=True,
        memory=memory,
    )

    t = None
    count = 0
    while t is None and count < max_attemps:
        responses = []

        try:
            for m in messages:
                # FIXME: add json format here
                print(m.input_variables)
                inputs = {
                    "text": text,
                    "rubric": rubric,
                    "title": title,
                }

                question_value = m.format(**{k: inputs[k] for k in m.input_variables})

                if model_name in ("", "openai:gpt-4-1106-preview"):
                    if "JSON" in question_value:
                        model.model_kwargs["response_format"] = {"type": "json_object"}
                    elif "response_format" in model.model_kwargs:
                        del model.model_kwargs["response_format"]

                r = conversation.__call__({"question": question_value})
                responses.append(r["text"])

            print(responses)

            t = responses[-1]
            json.loads(t)
        except json.JSONDecodeError:
            t = None
        finally:
            count += 1

    assert (
        t is not None
    ), f"Failed to get a response from the chatbot ({count} attempts)"

    return t.strip()
