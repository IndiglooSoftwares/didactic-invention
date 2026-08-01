"""Entry point for the Agentic Mentor CLI."""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

from src.agent import AgenticMentor
from src.agent.memory import ConversationMemory
from src.config import MentorConfig


def _make_llm_client(config: MentorConfig):
    """Instantiate the appropriate LLM client from config.

    Extend this function to add support for more providers.

    Args:
        config: Resolved :class:`MentorConfig`.

    Returns:
        An LLM client with ``complete_with_messages`` and ``stream`` methods.
    """
    provider = config.llm.provider.lower()

    if provider == "openai":
        try:
            from openai import OpenAI
        except ImportError as exc:
            print(
                "OpenAI package not installed. Run: pip install openai",
                file=sys.stderr,
            )
            raise exc

        client = OpenAI(
            api_key=config.llm.api_key,
            base_url=config.llm.base_url,
        )

        class _OpenAIAdapter:
            def complete_with_messages(self, messages):
                resp = client.chat.completions.create(
                    model=config.llm.model,
                    messages=messages,
                    temperature=config.llm.temperature,
                    max_tokens=config.llm.max_tokens,
                )
                return resp.choices[0].message.content

            def complete(self, prompt: str) -> str:
                return self.complete_with_messages(
                    [{"role": "user", "content": prompt}]
                )

            def stream(self, messages):
                for chunk in client.chat.completions.create(
                    model=config.llm.model,
                    messages=messages,
                    temperature=config.llm.temperature,
                    max_tokens=config.llm.max_tokens,
                    stream=True,
                ):
                    delta = chunk.choices[0].delta
                    if delta.content:
                        yield delta.content

        return _OpenAIAdapter()

    raise NotImplementedError(
        f"Provider '{provider}' is not implemented yet. "
        "Add it to main.py or supply a custom client."
    )


def _load_memory(session_file: str | None, max_history: int) -> ConversationMemory:
    if session_file and Path(session_file).exists():
        with open(session_file) as f:
            data = json.load(f)
        return ConversationMemory.from_dict(data, max_history=max_history)
    return ConversationMemory(max_history=max_history)


def _save_memory(memory: ConversationMemory, session_file: str | None) -> None:
    if session_file:
        Path(session_file).parent.mkdir(parents=True, exist_ok=True)
        with open(session_file, "w") as f:
            json.dump(memory.to_dict(), f, indent=2)


def run_cli() -> None:
    """Start an interactive CLI session with the Agentic Mentor."""
    config = MentorConfig.from_env()
    memory = _load_memory(config.session_file, config.max_history)
    llm_client = _make_llm_client(config)
    mentor = AgenticMentor(llm_client=llm_client, memory=memory)

    print("╔══════════════════════════════════════╗")
    print("║       Agentic Mentor — CLI           ║")
    print("╚══════════════════════════════════════╝")
    print("Type your message and press Enter.")
    print("Commands: /assess <topic> | /quiz <topic> | /explain <topic>")
    print("          /resources <topic> | /end | /quit\n")

    try:
        while True:
            user_input = input("You: ").strip()
            if not user_input:
                continue

            # --- CLI commands ---
            if user_input.lower() in ("/quit", "/exit"):
                print("Goodbye! Keep learning. 🚀")
                break

            if user_input.lower() == "/end":
                summary = mentor.end_session()
                print(f"\nMentor:\n{summary}\n")
                _save_memory(mentor.memory, config.session_file)
                continue

            if user_input.startswith("/assess "):
                topic = user_input[8:].strip()
                response = mentor.start_assessment(topic)
            elif user_input.startswith("/explain "):
                topic = user_input[9:].strip()
                response = mentor.explain(topic)
            elif user_input.startswith("/quiz "):
                topic = user_input[6:].strip()
                response = mentor.quiz(topic)
            elif user_input.startswith("/resources "):
                topic = user_input[11:].strip()
                response = mentor.suggest_resources(topic)
            else:
                response = mentor.chat(user_input)

            print(f"\nMentor:\n{response}\n")
            _save_memory(mentor.memory, config.session_file)

    except (KeyboardInterrupt, EOFError):
        print("\nSession interrupted. Progress saved.")
        _save_memory(mentor.memory, config.session_file)


if __name__ == "__main__":
    run_cli()
