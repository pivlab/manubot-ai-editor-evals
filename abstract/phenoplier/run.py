import subprocess

models = {
#  "ollama:starling-lm:7b-alpha-fp16": 1,
#  "ollama:mistral:7b-instruct-fp16": 1,
#  "ollama:deepseek-llm:7b-chat-fp16": 1,
#  "ollama:neural-chat:7b-v3.1-fp16": 1,
#  "ollama:openchat:7b-v3.5-fp16": 1,
#  "ollama:deepseek-llm:67b-chat-q5_0": 1,
#  "ollama:alfred:40b-1023-q5_1": 1,
  "openai:gpt-3.5-turbo": 1,
  "openai:gpt-4": 1,
}

for model, model_n_rep in models.items():
    model_name = "".join(model.split(":")[1:]).replace(":", "-")
    command = f"promptfoo eval -j 1 --no-cache --repeat {model_n_rep} --providers {model} -o outputs/{model_name}.html -o outputs/{model_name}.csv -o outputs/{model_name}.txt -o outputs/{model_name}.json -o outputs/{model_name}.yaml"
    print(command)
    subprocess.run(command, shell=True)

