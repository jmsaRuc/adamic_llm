import json
import os
import jsonlines
from tenacity import (
    retry,
    stop_after_attempt,
    wait_fixed,
    wait_random,
)
import groq
from openrouter import OpenRouter
import requests
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv("benchmark.env")


client = OpenAI(
    base_url=os.getenv("OPENAI_API_BASE", "https://api.openai.com/v1"),
    api_key=os.getenv("OPENAI_API_KEY"),
)

client_groq = groq.Groq()

client_openrouter = OpenRouter(
    api_key=os.getenv("OPENROUTER_API_KEY"),
)


# These are the basic functions to call ChatGPT
def before_retry_fn(retry_state):
    if retry_state.attempt_number > 1:
        print(
            f"Retrying API call. Attempt #{retry_state.attempt_number}, {retry_state}"
        )


@retry(
    wait=wait_fixed(5) + wait_random(0, 5),
    stop=stop_after_attempt(1),
    before=before_retry_fn,
)
def groq_completion_with_backoff(**kwargs):
    try:
        response = client_groq.chat.completions.create(**kwargs)

    except groq.BadRequestError as e:
        if "tool_use_failed" in str(e):
            kwargs["messages"][-1]["content"] += "\n[WARNING]\nDONT USE ANY TOOLS"
            response2 = client_groq.chat.completions.create(**kwargs)
            return response2
        else:
            raise groq.BadRequestError(f"GROQ API call failed: {e}")
    return response


@retry(
    wait=wait_fixed(5) + wait_random(0, 5),
    stop=stop_after_attempt(1),
    before=before_retry_fn,
)
def openrouter_completion_with_backoff(**kwargs):
    return client_openrouter.chat.send(**kwargs)


@retry(
    wait=wait_fixed(5) + wait_random(0, 5),
    stop=stop_after_attempt(6),
    before=before_retry_fn,
)
def completion_with_backoff(**kwargs):
    return client.chat.completions.create(**kwargs)


def get_completion_messages(
    messages,
    model="gpt-3.5-turbo-1106",
    temperature=0,
    max_tokens=8192,
    model_loaded=None,
    tokenizer=None,
    model_type="openai",
):

    if model_type == "openai":
        response = completion_with_backoff(
            model=model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        name = (
            response.choices[0].message.name
            if hasattr(response.choices[0].message, "name")
            else None
        )
        completion_tokens = (
            response.usage.completion_tokens
            if hasattr(response.usage, "completion_tokens")
            else None
        )
        prompt_tokens = (
            response.usage.prompt_tokens
            if hasattr(response.usage, "prompt_tokens")
            else None
        )
        return (
            response.choices[0].message.content,
            name,
            completion_tokens,
            prompt_tokens,
        )
    elif model_type == "groq":
        response = groq_completion_with_backoff(
            model=model,
            messages=messages,
            temperature=temperature,
            max_completion_tokens=max_tokens,
            stream=False,
            tools=None,
            tool_choice="none",
            disable_tool_validation=True,
            reasoning_effort="high" if "gpt-oss" in model else None,
        )
        name = (
            response.choices[0].message.name
            if hasattr(response.choices[0].message, "name")
            else None
        )
        completion_tokens = (
            response.usage.completion_tokens
            if hasattr(response.usage, "completion_tokens")
            else None
        )
        prompt_tokens = (
            response.usage.prompt_tokens
            if hasattr(response.usage, "prompt_tokens")
            else None
        )
        return (
            response.choices[0].message.content,
            name,
            completion_tokens,
            prompt_tokens,
        )
    elif model_type == "open-router":
        response = openrouter_completion_with_backoff(
            model=model,
            messages=messages,
            temperature=temperature,
            max_completion_tokens=max_tokens,
            reasoning={"effort": "xhigh"},
        )
        name = (
            response.choices[0].message.name
            if hasattr(response.choices[0].message, "name")
            else None
        )
        completion_tokens = (
            response.usage.completion_tokens
            if hasattr(response.usage, "completion_tokens")
            else None
        )
        prompt_tokens = (
            response.usage.prompt_tokens
            if hasattr(response.usage, "prompt_tokens")
            else None
        )
        return (
            response.choices[0].message.content,
            name,
            completion_tokens,
            prompt_tokens,
        )
    elif model_type == "together":
        response = query_together_model(
            messages, model=model, max_tokens=max_tokens, temperature=temperature
        )
        return response
    elif "mis" in model and model_loaded != None and tokenizer != None:
        encodeds = tokenizer.apply_chat_template(messages, return_tensors="pt")
        model_inputs = encodeds.to("cuda")
        generated_ids = model_loaded.generate(
            model_inputs,
            max_new_tokens=max_tokens,
            pad_token_id=tokenizer.pad_token_id,
            do_sample=True if temperature > 0 else False,
            temperature=temperature,
        )
        decoded = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
        return decoded[0].split("[/INST] ")[-1]
    else:
        raise NotImplementedError


def get_completion(
    prompt,
    model="gpt-3.5-turbo",
    temperature=0,
    max_tokens=8192,
    model_loaded=None,
    tokenizer=None,
):
    messages = [{"role": "user", "content": prompt}]
    response, name_key = get_completion_messages(
        messages,
        model=model,
        temperature=temperature,
        max_tokens=max_tokens,
        model_loaded=model_loaded,
        tokenizer=tokenizer,
    )
    return response, name_key


@retry(
    wait=wait_fixed(5) + wait_random(0, 5),
    stop=stop_after_attempt(5),
    before=before_retry_fn,
)
def query_together_model(
    messages,
    model: str = "Qwen/Qwen1.5-72B-Chat",
    max_tokens: int = 128,
    temperature: float = 0,
):
    endpoint = "https://api.together.xyz/v1/chat/completions"
    api_key = os.getenv("TOGETHER_API_KEY")
    headers = {
        "Authorization": f"Bearer {api_key}",
    }
    try:
        res = requests.post(
            endpoint,
            json={
                "model": model,
                "max_tokens": max_tokens,
                "request_type": "language-model-inference",
                "temperature": temperature,
                "messages": messages,  # [{"role": "user", "content": prompt}]
            },
            headers=headers,
        )
        output = res.json()["choices"][0]["message"]["content"]
    except Exception as e:
        res_json = res.json()
        if "error" in res_json and res_json["error"]["message"].startswith(
            "Input validation error: `inputs`"
        ):
            output = "the question is too long"
        else:
            raise e
    return output


@retry(
    wait=wait_fixed(5) + wait_random(0, 5),
    stop=stop_after_attempt(3),
    before=before_retry_fn,
)
def query_openrouter_model(
    messages, model="openai/gpt-4o-mini", max_tokens=8192, temperature=0
):

    client = OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=os.getenv("OPENROUTER_API_KEY"),
    )
    try:
        completions = client.chat.completions.create(
            extra_headers={},
            model=model,
            messages=messages,
            max_tokens=max_tokens,
            n=1,
            stop=None,
            temperature=temperature,
            reasoning_effort="xhigh",
        )
        output = completions.choices[0].message.content.strip()
        completion_tokens = completions.usage.completion_tokens
        prompt_tokens = completions.usage.prompt_tokens
    except Exception as e:
        print("[ERROR]", e)
        output = "CAUTION: Problematic output!"
        completion_tokens = 0
        prompt_tokens = 0

    return output, completion_tokens, prompt_tokens


def parallel_query_chatgpt_model(args):
    return get_completion(*args)


class Agent:
    def __init__(
        self,
        name,
        system_prompt="",
        model="gpt-3.5-turbo-1106",
        temperature=0,
        max_tokens=8192,
        model_loaded=None,
        tokenizer=None,
        model_type="openai",
    ):
        self.name = name
        self.system_prompt = system_prompt
        self.model = model
        self.model_type = model_type
        self.model_loaded = model_loaded
        self.tokenizer = tokenizer
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.messages = (
            []
            if system_prompt == ""
            else [{"role": "system", "content": system_prompt}]
        )

    def respond(self, prompt, model=None):
        self.messages.append({"role": "user", "content": prompt})
        response, name_key, completion_tokens, prompt_tokens = get_completion_messages(
            self.messages,
            model=self.model if model == None else model,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            model_loaded=self.model_loaded,
            tokenizer=self.tokenizer,
            model_type=self.model_type,
        )
        self.messages.append(
            {"role": "assistant", "content": response, "name": name_key}
            if name_key
            else {"role": "assistant", "content": response}
        )
        return response, completion_tokens, prompt_tokens

    def respond_messages(self, messages, model=None):
        self.messages = messages
        response = get_completion_messages(
            self.messages,
            model=self.model if model == None else model,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            model_loaded=self.model_loaded,
            tokenizer=self.tokenizer,
            model_type=self.model_type,
        )
        self.messages.append({"role": "assistant", "content": response})
        return response

    def change_max_tokens(self, max_tokens):
        self.max_tokens = max_tokens

    def reset(self):
        self.messages = (
            []
            if self.system_prompt == ""
            else [{"role": "system", "content": self.system_prompt}]
        )


# Helper functions
def read_line_from_txt(file_path):
    with open(file_path, "r") as f:
        lines = f.readlines()
    return [l.strip() for l in lines]


# write a list of strings to a txt file
def write_list_to_txt(list_str, file_path):
    with open(file_path, "w") as f:
        for s in list_str:
            f.write(s + "\n")


# create a folder if it doesn't exist
def create_folder_if_not_exist(folder_path):
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)


def jsonl_to_list(path):
    with jsonlines.open(path) as reader:
        dataset = [obj for obj in reader]
    return dataset


def list_to_jsonl(dataset, path):
    with jsonlines.open(path, mode="w") as writer:
        writer.write_all(dataset)


def print_json(ex):
    print(json.dumps(ex, indent=2, ensure_ascii=False))


def make_dir_if_not_exist(dir_path):
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)


def prepara_model(model, precision="fp32"):
    model_loaded = None
    tokenizer = None
    return model_loaded, tokenizer


def prompt_to_messages(role, prompt, messages=[]):
    # role: user, assistant, system
    message = {"role": role, "content": prompt}
    messages.append(message)
    return messages


def messages_to_prompt(messages, tokenizer):
    if tokenizer.chat_template:
        prompt = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
    else:
        prompt = " ".join([f"{m['content']}" for m in messages])
    return prompt


def prompt_to_chatprompt(prompt, tokenizer):
    messages = prompt_to_messages("user", prompt, messages=[])
    prompt = messages_to_prompt(messages, tokenizer)
    return prompt


# testing functions
def _test_completion():
    model = "gpt-3.5-turbo"

    model_loaded = None
    tokenizer = None

    res = get_completion(
        "Qustion: what is 10 * 5 -4?\n Answer: Let's think step by step.",
        model=model,
        temperature=0,
        max_tokens=8192,
        model_loaded=model_loaded,
        tokenizer=tokenizer,
    )
    print(res)


def _test_agent():
    model = "gpt-3.5-turbo-1106"
    model = "zero-one-ai/Yi-34B-Chat"
    model_type = "together"

    model_loaded = None
    tokenizer = None

    agent = Agent(
        name="test",
        model=model,
        model_loaded=model_loaded,
        tokenizer=tokenizer,
        model_type=model_type,
    )
    res = agent.respond("Qustion: what is the captital of China?\n Answer:")
    res2 = agent.respond("Answer in Chinese.")
    # print(res)
    print_json(agent.messages)


def _test_together():
    prompt = "Qustion: what is a good time to go to sleed and get up?\n Answer:"
    model = "Qwen/Qwen1.5-72B-Chat"
    # model = "zero-one-ai/Yi-34B-Chat"
    # res = query_together_model(os.getenv("TOGETHER_API_KEY"), prompt, model=model, max_tokens=1024, temperature=0)
    res = query_openrouter_model(
        prompt, model="openai/gpt-4o-mini", max_tokens=64, temperature=0
    )
    print(res)


def test_openrouter():
    api_key = os.environ["OPENROUTER_API_KEY"]
    messages = [
        {"role": "user", "content": "What is the capital of France and Germany?"}
    ]
    list_models = [
        "anthropic/claude-3.5-sonnet",
        "openai/gpt-4o-2024-08-06",
        "google/gemini-pro-1.5",
        "google/gemini-flash-1.5",
        "openai/gpt-4o-mini",
        "anthropic/claude-3-haiku",
        "qwen/qwen-2.5-72b-instruct",
        "meta-llama/llama-3.1-70b-instruct",
        "meta-llama/llama-3.1-405b-instruct",
    ]

    list_models = ["openai/gpt-4o-mini"]
    for model in list_models:
        output = query_openrouter_model(messages=messages, model=model)
        print(f"model: {output}")


def _test_groq():
    model = "openai/gpt-oss-120b"

    res = get_completion(
        "Qustion: what is 10 * 5 -4?\n Answer: Let's think step by step.",
        model=model,
        temperature=0,
        max_tokens=1024,
    )
    print(res)


if __name__ == "__main__":
    # _test()
    # _test_completion()
    # _test_agent()
    # _test_translator()
    # _test_vllm()
    # _test_together()
    # test_openrouter()
    _test_groq()
