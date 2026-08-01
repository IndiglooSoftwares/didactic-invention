"""Tools available to the Agentic Mentor."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional


@dataclass
class ToolResult:
    """Standardised return type for all mentor tools."""

    tool_name: str
    success: bool
    output: Any
    error: Optional[str] = None


class MentorTools:
    """Collection of tools the Agentic Mentor can invoke.

    Each method returns a :class:`ToolResult` so the agent can handle
    errors uniformly.  The actual LLM calls are delegated to the
    ``llm_client`` passed at construction time, which keeps the tools
    decoupled from any specific model provider.
    """

    def __init__(self, llm_client: Any) -> None:
        """
        Args:
            llm_client: Any object that exposes a ``complete(prompt: str) -> str``
                        method.  Swap in your preferred provider (OpenAI, Anthropic,
                        a local model, a mock for testing, etc.).
        """
        self._llm = llm_client
        self._registry: Dict[str, Callable] = {
            "explain_concept": self.explain_concept,
            "generate_quiz": self.generate_quiz,
            "evaluate_answer": self.evaluate_answer,
            "suggest_resources": self.suggest_resources,
            "summarize_session": self.summarize_session,
        }

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @property
    def available_tools(self) -> List[str]:
        """Names of all registered tools."""
        return list(self._registry.keys())

    def call(self, tool_name: str, **kwargs) -> ToolResult:
        """Dispatch a tool call by name."""
        if tool_name not in self._registry:
            return ToolResult(
                tool_name=tool_name,
                success=False,
                output=None,
                error=f"Unknown tool '{tool_name}'. Available: {self.available_tools}",
            )
        try:
            result = self._registry[tool_name](**kwargs)
            return ToolResult(tool_name=tool_name, success=True, output=result)
        except Exception as exc:  # noqa: BLE001
            return ToolResult(
                tool_name=tool_name, success=False, output=None, error=str(exc)
            )

    # ------------------------------------------------------------------
    # Individual tools
    # ------------------------------------------------------------------

    def explain_concept(self, topic: str, level: str = "beginner") -> str:
        """Return a structured explanation of *topic* for the given *level*.

        Args:
            topic: The concept to explain (e.g. "recursion").
            level: Learner level — "beginner", "intermediate", or "advanced".

        Returns:
            A formatted explanation string.
        """
        from .prompts import EXPLANATION_PROMPT

        prompt = EXPLANATION_PROMPT.format(topic=topic, level=level)
        return self._llm.complete(prompt)

    def generate_quiz(
        self, topic: str, level: str = "beginner", num_questions: int = 3
    ) -> str:
        """Generate a short quiz on *topic*.

        Args:
            topic: The subject to quiz the learner on.
            level: Learner level.
            num_questions: How many questions to include (1–10).

        Returns:
            A formatted quiz string.
        """
        from .prompts import QUIZ_PROMPT

        num_questions = max(1, min(num_questions, 10))
        prompt = QUIZ_PROMPT.format(
            topic=topic, level=level, num_questions=num_questions
        )
        return self._llm.complete(prompt)

    def evaluate_answer(
        self,
        question: str,
        correct_answer: str,
        learner_answer: str,
        level: str = "beginner",
    ) -> str:
        """Evaluate *learner_answer* and return constructive feedback.

        Args:
            question: The original question posed to the learner.
            correct_answer: The expected correct answer.
            learner_answer: What the learner responded.
            level: Learner level.

        Returns:
            Feedback as a string.
        """
        from .prompts import FEEDBACK_PROMPT

        prompt = FEEDBACK_PROMPT.format(
            question=question,
            correct_answer=correct_answer,
            learner_answer=learner_answer,
            level=level,
        )
        return self._llm.complete(prompt)

    def suggest_resources(self, topic: str, level: str = "beginner") -> str:
        """Suggest curated learning resources for *topic*.

        Args:
            topic: The subject the learner wants to study.
            level: Learner level.

        Returns:
            A list of suggested resources as a formatted string.
        """
        prompt = (
            f"Suggest 5 high-quality, freely available learning resources for the "
            f"topic '{topic}' suitable for a {level} learner. "
            f"Include: name, URL (if applicable), format (video/article/book/course), "
            f"and a one-sentence description of why it is useful."
        )
        return self._llm.complete(prompt)

    def summarize_session(
        self,
        topics: List[str],
        concepts: List[str],
        performance: str = "",
    ) -> str:
        """Generate a session summary with next-step recommendations.

        Args:
            topics: Topics covered during the session.
            concepts: Key concepts the learner encountered.
            performance: Optional free-text description of how the learner did.

        Returns:
            A session summary string.
        """
        from .prompts import SESSION_SUMMARY_PROMPT

        prompt = SESSION_SUMMARY_PROMPT.format(
            topics=", ".join(topics) if topics else "none",
            concepts=", ".join(concepts) if concepts else "none",
            performance=performance or "not recorded",
        )
        return self._llm.complete(prompt)
