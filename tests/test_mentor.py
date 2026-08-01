"""Tests for AgenticMentor."""
import pytest

from src.agent.mentor import AgenticMentor
from src.agent.memory import ConversationMemory


class MockLLMClient:
    """Minimal LLM stub."""

    def complete(self, prompt: str) -> str:
        return f"[mock: {prompt[:30]}]"

    def complete_with_messages(self, messages) -> str:
        last = messages[-1]["content"] if messages else ""
        return f"[mock response to: {last[:30]}]"

    def stream(self, messages):
        yield "[mock "
        yield "streamed"
        yield " response]"


@pytest.fixture()
def mentor():
    return AgenticMentor(llm_client=MockLLMClient())


class TestAgenticMentor:
    def test_chat_adds_messages_to_memory(self, mentor):
        response = mentor.chat("What is recursion?")
        history = mentor.memory.get_history()
        assert len(history) == 2
        assert history[0]["role"] == "user"
        assert history[1]["role"] == "assistant"
        assert isinstance(response, str)

    def test_stream_chat_yields_chunks_and_stores_response(self, mentor):
        chunks = list(mentor.stream_chat("Explain loops"))
        assert len(chunks) > 0
        history = mentor.memory.get_history()
        # user + assistant
        assert len(history) == 2
        full = "".join(chunks)
        assert history[1]["content"] == full

    def test_start_assessment_marks_topic_covered(self, mentor):
        mentor.start_assessment("Python")
        assert "Python" in mentor.memory.learner_profile.topics_covered

    def test_explain_marks_topic_covered(self, mentor):
        mentor.explain("closures")
        assert "closures" in mentor.memory.learner_profile.topics_covered

    def test_quiz_returns_string(self, mentor):
        result = mentor.quiz("sorting algorithms", num_questions=2)
        assert isinstance(result, str)

    def test_evaluate_returns_feedback(self, mentor):
        feedback = mentor.evaluate(
            question="What does DRY stand for?",
            correct_answer="Don't Repeat Yourself",
            learner_answer="Don't Repeat Yourself",
        )
        assert isinstance(feedback, str)

    def test_suggest_resources_returns_string(self, mentor):
        result = mentor.suggest_resources("data structures")
        assert isinstance(result, str)

    def test_end_session_clears_history_and_increments_count(self, mentor):
        mentor.chat("hello")
        mentor.end_session()
        assert mentor.memory.get_history() == []
        assert mentor.memory.learner_profile.session_count == 1

    def test_multiple_sessions_accumulate_count(self, mentor):
        for _ in range(3):
            mentor.chat("question")
            mentor.end_session()
        assert mentor.memory.learner_profile.session_count == 3

    def test_build_messages_includes_system_prompt(self, mentor):
        mentor.memory.add_message("user", "hi")
        messages = mentor._build_messages()
        assert messages[0]["role"] == "system"
        assert len(messages) >= 2

    def test_build_messages_injects_session_summary(self, mentor):
        mentor.memory.session_summary = "Covered Python basics."
        mentor.memory.add_message("user", "continue")
        messages = mentor._build_messages()
        roles = [m["role"] for m in messages]
        assert roles.count("system") == 2

    def test_learner_level_respected_in_tools(self):
        """Tools should use the learner's current level."""
        captured = {}

        class CapturingLLM:
            def complete(self, prompt: str) -> str:
                captured["prompt"] = prompt
                return "ok"

            def complete_with_messages(self, messages):
                return "ok"

            def stream(self, messages):
                yield "ok"

        m = AgenticMentor(llm_client=CapturingLLM())
        m.memory.update_level("advanced")
        m.explain("metaclasses")
        assert "advanced" in captured.get("prompt", "")

    def test_custom_memory_is_used(self):
        mem = ConversationMemory(max_history=5)
        mem.learner_profile.name = "Alice"
        m = AgenticMentor(llm_client=MockLLMClient(), memory=mem)
        assert m.memory.learner_profile.name == "Alice"
