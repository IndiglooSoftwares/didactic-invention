"""Core Agentic Mentor implementation."""
from __future__ import annotations

from typing import Any, Dict, Iterator, List, Optional

from .memory import ConversationMemory
from .prompts import ASSESSMENT_PROMPT, SYSTEM_PROMPT
from .tools import MentorTools


class AgenticMentor:
    """An autonomous, tool-using AI mentor.

    The mentor maintains a conversation with a learner, uses its tool-set to
    explain concepts, quiz the learner, evaluate answers, and suggest resources
    — all while tracking progress in a :class:`~agent.memory.ConversationMemory`.

    Quick-start::

        from src.agent import AgenticMentor
        from my_llm_client import OpenAIClient  # or any compatible client

        mentor = AgenticMentor(llm_client=OpenAIClient())
        response = mentor.chat("I want to learn about recursion")
        print(response)

    Args:
        llm_client: Any object with a ``complete(prompt: str) -> str`` method.
                    For streaming, it may also expose
                    ``stream(prompt: str) -> Iterator[str]``.
        memory: Optional pre-built :class:`ConversationMemory`.  Useful when
                resuming a previous session.
        max_history: Maximum number of messages kept in the sliding window.
    """

    def __init__(
        self,
        llm_client: Any,
        memory: Optional[ConversationMemory] = None,
        max_history: int = 20,
    ) -> None:
        self._llm = llm_client
        self.memory = memory or ConversationMemory(max_history=max_history)
        self.tools = MentorTools(llm_client)
        self._system_message = SYSTEM_PROMPT

    # ------------------------------------------------------------------
    # Primary interaction methods
    # ------------------------------------------------------------------

    def chat(self, user_message: str) -> str:
        """Send a message and get the mentor's response.

        The mentor may decide to invoke one or more tools internally before
        returning its final reply.

        Args:
            user_message: The learner's input.

        Returns:
            The mentor's response as a plain string.
        """
        self.memory.add_message("user", user_message)

        # Build the full prompt: system prompt + conversation history
        messages = self._build_messages()

        # Ask the LLM to decide what to do next
        raw_response = self._llm.complete_with_messages(messages)

        # Optionally parse tool calls from the response (provider-specific)
        response = self._maybe_execute_tools(raw_response)

        self.memory.add_message("assistant", response)
        return response

    def stream_chat(self, user_message: str) -> Iterator[str]:
        """Stream the mentor's response token by token.

        Yields successive chunks of the response.  Accumulates the full
        response in memory once streaming is complete.

        Args:
            user_message: The learner's input.

        Yields:
            String chunks of the response.
        """
        self.memory.add_message("user", user_message)
        messages = self._build_messages()

        full_response: List[str] = []
        for chunk in self._llm.stream(messages):
            full_response.append(chunk)
            yield chunk

        self.memory.add_message("assistant", "".join(full_response))

    def start_assessment(self, topic: str) -> str:
        """Kick off a structured assessment for *topic*.

        Args:
            topic: The subject the learner wants to study.

        Returns:
            The mentor's opening assessment question(s).
        """
        prompt = ASSESSMENT_PROMPT.format(topic=topic)
        self.memory.add_message("user", f"I want to learn about: {topic}")
        self.memory.mark_topic_covered(topic)
        messages = self._build_messages(override_user_content=prompt)
        response = self._llm.complete_with_messages(messages)
        self.memory.add_message("assistant", response)
        return response

    def explain(self, topic: str) -> str:
        """Directly request a concept explanation.

        Args:
            topic: The concept to explain.

        Returns:
            A structured explanation.
        """
        result = self.tools.call(
            "explain_concept",
            topic=topic,
            level=self.memory.learner_profile.level,
        )
        if result.success:
            self.memory.add_message("assistant", result.output)
            self.memory.mark_topic_covered(topic)
            return result.output
        raise RuntimeError(result.error)

    def quiz(self, topic: str, num_questions: int = 3) -> str:
        """Generate a quiz on *topic* for the current learner level.

        Args:
            topic: The subject to quiz on.
            num_questions: Number of questions (1–10).

        Returns:
            The quiz as a formatted string.
        """
        result = self.tools.call(
            "generate_quiz",
            topic=topic,
            level=self.memory.learner_profile.level,
            num_questions=num_questions,
        )
        if result.success:
            self.memory.add_message("assistant", result.output)
            return result.output
        raise RuntimeError(result.error)

    def evaluate(
        self, question: str, correct_answer: str, learner_answer: str
    ) -> str:
        """Evaluate a learner's answer and return feedback.

        Args:
            question: The original question.
            correct_answer: The expected answer.
            learner_answer: What the learner said.

        Returns:
            Constructive feedback.
        """
        result = self.tools.call(
            "evaluate_answer",
            question=question,
            correct_answer=correct_answer,
            learner_answer=learner_answer,
            level=self.memory.learner_profile.level,
        )
        if result.success:
            self.memory.add_message("assistant", result.output)
            return result.output
        raise RuntimeError(result.error)

    def suggest_resources(self, topic: str) -> str:
        """Get resource suggestions for *topic*.

        Args:
            topic: The subject to find resources for.

        Returns:
            Curated resource suggestions.
        """
        result = self.tools.call(
            "suggest_resources",
            topic=topic,
            level=self.memory.learner_profile.level,
        )
        if result.success:
            self.memory.add_message("assistant", result.output)
            return result.output
        raise RuntimeError(result.error)

    def end_session(self) -> str:
        """Summarise the session and persist progress.

        Returns:
            A session summary with next-step recommendations.
        """
        profile = self.memory.learner_profile
        result = self.tools.call(
            "summarize_session",
            topics=profile.topics_covered,
            concepts=profile.mastered_concepts,
        )
        if result.success:
            self.memory.session_summary = result.output
            self.memory.increment_session()
            self.memory.clear_history()
            return result.output
        raise RuntimeError(result.error)

    # ------------------------------------------------------------------
    # Helper methods
    # ------------------------------------------------------------------

    def _build_messages(
        self, override_user_content: Optional[str] = None
    ) -> List[Dict[str, str]]:
        """Combine system prompt, optional session summary, and history."""
        messages: List[Dict[str, str]] = [
            {"role": "system", "content": self._system_message}
        ]

        # Inject a condensed summary of past sessions if available
        if self.memory.session_summary:
            messages.append(
                {
                    "role": "system",
                    "content": (
                        "## Previous session summary\n"
                        + self.memory.session_summary
                    ),
                }
            )

        history = self.memory.get_history()

        if override_user_content and history:
            # Replace the last user message with the override content
            history = history[:-1] + [
                {"role": "user", "content": override_user_content}
            ]
        elif override_user_content:
            history = [{"role": "user", "content": override_user_content}]

        messages.extend(history)
        return messages

    def _maybe_execute_tools(self, raw_response: str) -> str:
        """Parse and execute any tool calls embedded in *raw_response*.

        This is a minimal hook.  Replace or extend this method to implement
        your provider's function-calling / tool-use protocol.

        Args:
            raw_response: The raw text returned by the LLM.

        Returns:
            The final response string after executing any tool calls.
        """
        # Default: return response as-is.
        # Override this method to handle structured tool-call formats
        # (e.g. OpenAI function calling, Anthropic tool use, etc.).
        return raw_response
