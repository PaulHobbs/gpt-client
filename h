#!/usr/bin/python3
"""
This script utilizes OpenAI's GPT-3 API to create a chatbot that can engage in
conversations with users.

Install dependencies with:

$ pip install openai prompt_toolkit

You can obtain an API key by creating an account on the OpenAI website. Once
you have your API key, save it to ~/.OPENAI-KEY.

This chatbot can be customized by adjusting the system prompt, _SYSTEM, below.
You can pass --critize to automatically ask the chatbot to criticize and expand
on each of its responses.
"""

from typing import Iterable, Callable, Any

import openai
import sys
import os
import functools
from prompt_toolkit import PromptSession
from prompt_toolkit.history import FileHistory

SESSION = None 
_QUITS = frozenset(("q", "quit", "exit", "bye"))

_MODELS = {
    "3": "gpt-3.5-turbo",
    "4": "gpt-4-1106-preview",
}

_OPENAI_KEY = '~/.OPENAI-KEY'

# System prompts are more expensive. Just use a user prompt for now.
def system_prompt(p):
    return {"role": "user", "content": p.strip()}

_SYSTEM = system_prompt("You are a rational, wise, careful, and deliberate thinker.")

def main():
    import argparse

    parser = argparse.ArgumentParser(description="A chatGPT cli interface")
    parser.add_argument(
        "prompt",
        nargs="*",
        type=str,
        help="Prompt. If not specified, uses readline for back-and-forth.",
    )
    parser.add_argument(
        "-v", dest="version", default="4", type=str, help="chatGPT model to use."
    )
    parser.add_argument(
        "--max_tokens", type=int, default=512, help="maximum chat response size."
    )
    parser.add_argument(
        "--criticise",
        dest="raw",
        action="store_false",
        default=True,
        help="use criticism.",
    )
    parser.add_argument(
        '--system',
        action='store_true',
        help='Do not provide system text.',
    )
    args = parser.parse_args()
    if args.system:
        global _SYSTEM
        _SYSTEM = _SYSTEM7

    openai.api_key = open(os.path.expanduser(_OPENAI_KEY)).read().strip()
    model = _MODELS[args.version]

    if not args.prompt:
        global SESSION
        SESSION = PromptSession(history=FileHistory(os.path.expanduser("~/.chat")))
        try:
            back_and_forth(
                functools.partial(post, max_tokens=args.max_tokens, model=model),
                raw=args.raw,
            )
        except (EOFError, KeyboardInterrupt):
            return
    else:
        post([message(' '.join(args.prompt))], args.max_tokens, model, raw=args.raw)


def message(prompt, role="user"):
    return {"role": role, "content": prompt}


def post(messages, max_tokens, model, raw=False) -> list[str]:
    go = lambda ms=messages: stream(
        openai.ChatCompletion.create(
            model=model,
            messages=[_SYSTEM] + ms,
            max_tokens=max_tokens,
            temperature=0.3,
            stream=True,
        )
    )
    if raw:
        return [go()]

    more = []
    more.append(go())
    more.append(
        message(
            "Checking the answer for internal consistency, criticiziing it, and providing an improved answer if needed:",
            role="assistant",
        )
    )
    print("\n----- reflect -----")
    more.append(go(messages + more))
    print("\n----- extend -----")
    more.append(message("Any other ideas? Provide examples and keep it short."))
    more.append(go(messages + more))
    return more


def prompt():
    i = SESSION.prompt(">> ", multiline=True)
    if i in _QUITS:
        raise EOFError
    return i


def back_and_forth(post: Callable, raw: bool = False):
    messages = []
    while True:
        m = prompt()
        messages.append(message(m))
        messages.extend(post(messages, raw=raw))
        # raw = True
        print("\n")


def stream(events: Iterable[Any]) -> str:
    response = []
    for event in events:
        event_text = event["choices"][0]["delta"].get("content", "")
        response.append(event_text)
        sys.stdout.write(event_text)
        sys.stdout.flush()
    return message("".join(response), role="assistant")


if __name__ == "__main__":
    main()
    print("\n\n")
