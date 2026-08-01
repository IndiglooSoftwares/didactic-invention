"""Tests for MentorTools using a mock LLM client."""
import pytest

from src.agent.tools import MentorTools, ToolResult


class MockLLMClient:
    """Minimal LLM stub that echoes back a canned response."""

    def complete(self, prompt: str) -> str:
        return f"[mock response for: {prompt[:40]}]"

    def complete_with_messages(self, messages) -> str:
        return "[mock chat response]"

    def stream(self, messages):
        yield "[mock"
        yield " stream"
        yield "]"


@pytest.fixture()
def tools():
    return MentorTools(llm_client=MockLLMClient())


class TestMentorTools:
    def test_available_tools_lists_all_tools(self, tools):
        expected = {
            "explain_concept",
            "generate_quiz",
            "evaluate_answer",
            "suggest_resources",
            "summarize_session",
        }
        assert set(tools.available_tools) == expected

    def test_call_unknown_tool_returns_failure(self, tools):
        result = tools.call("nonexistent_tool")
        assert isinstance(result, ToolResult)
        assert result.success is False
        assert "Unknown tool" in result.error

    def test_explain_concept_returns_tool_result(self, tools):
        result = tools.call("explain_concept", topic="recursion", level="beginner")
        assert result.success is True
        assert isinstance(result.output, str)
        assert len(result.output) > 0

    def test_generate_quiz_clamps_num_questions(self, tools):
        # num_questions=0 should be clamped to 1; the LLM still returns a string
        result = tools.call(
            "generate_quiz", topic="loops", level="beginner", num_questions=0
        )
        assert result.success is True

    def test_generate_quiz_max_clamped_at_10(self, tools):
        result = tools.call(
            "generate_quiz", topic="loops", level="beginner", num_questions=100
        )
        assert result.success is True

    def test_evaluate_answer_success(self, tools):
        result = tools.call(
            "evaluate_answer",
            question="What is 2+2?",
            correct_answer="4",
            learner_answer="4",
            level="beginner",
        )
        assert result.success is True
        assert isinstance(result.output, str)

    def test_suggest_resources_success(self, tools):
        result = tools.call(
            "suggest_resources", topic="machine learning", level="intermediate"
        )
        assert result.success is True

    def test_summarize_session_with_empty_lists(self, tools):
        result = tools.call("summarize_session", topics=[], concepts=[])
        assert result.success is True

    def test_summarize_session_with_data(self, tools):
        result = tools.call(
            "summarize_session",
            topics=["Python", "functions"],
            concepts=["closures", "decorators"],
            performance="answered 4/5 quiz questions correctly",
        )
        assert result.success is True
        assert isinstance(result.output, str)

    def test_tool_error_is_captured_in_result(self):
        """If the LLM client raises, ToolResult.success should be False."""

        class BrokenLLM:
            def complete(self, prompt):
                raise ConnectionError("network error")

        broken_tools = MentorTools(llm_client=BrokenLLM())
        result = broken_tools.call("explain_concept", topic="anything")
        assert result.success is False
        assert "network error" in result.error
