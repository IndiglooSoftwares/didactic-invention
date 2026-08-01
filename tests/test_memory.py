"""Tests for ConversationMemory."""
import pytest

from src.agent.memory import ConversationMemory, LearnerProfile, Message


class TestMessage:
    def test_to_dict_contains_role_and_content(self):
        msg = Message(role="user", content="Hello")
        d = msg.to_dict()
        assert d == {"role": "user", "content": "Hello"}


class TestConversationMemory:
    def test_add_and_get_history(self):
        mem = ConversationMemory()
        mem.add_message("user", "hi")
        mem.add_message("assistant", "hello")
        history = mem.get_history()
        assert len(history) == 2
        assert history[0] == {"role": "user", "content": "hi"}
        assert history[1] == {"role": "assistant", "content": "hello"}

    def test_sliding_window_trims_oldest_messages(self):
        mem = ConversationMemory(max_history=3)
        for i in range(5):
            mem.add_message("user", f"msg {i}")
        history = mem.get_history()
        assert len(history) == 3
        assert history[0]["content"] == "msg 2"

    def test_clear_history_keeps_profile(self):
        mem = ConversationMemory()
        mem.add_message("user", "hello")
        mem.learner_profile.name = "Alice"
        mem.clear_history()
        assert mem.get_history() == []
        assert mem.learner_profile.name == "Alice"

    def test_update_level_valid(self):
        mem = ConversationMemory()
        mem.update_level("intermediate")
        assert mem.learner_profile.level == "intermediate"

    def test_update_level_invalid_raises(self):
        mem = ConversationMemory()
        with pytest.raises(ValueError):
            mem.update_level("expert")

    def test_mark_topic_covered_no_duplicates(self):
        mem = ConversationMemory()
        mem.mark_topic_covered("recursion")
        mem.mark_topic_covered("recursion")
        assert mem.learner_profile.topics_covered.count("recursion") == 1

    def test_mark_concept_mastered_removes_from_weak_areas(self):
        mem = ConversationMemory()
        mem.flag_weak_area("pointers")
        mem.mark_concept_mastered("pointers")
        assert "pointers" not in mem.learner_profile.weak_areas
        assert "pointers" in mem.learner_profile.mastered_concepts

    def test_increment_session(self):
        mem = ConversationMemory()
        assert mem.learner_profile.session_count == 0
        mem.increment_session()
        mem.increment_session()
        assert mem.learner_profile.session_count == 2

    def test_serialisation_round_trip(self):
        mem = ConversationMemory(max_history=10)
        mem.add_message("user", "teach me Python")
        mem.add_message("assistant", "Sure!")
        mem.learner_profile.name = "Bob"
        mem.update_level("advanced")
        mem.mark_topic_covered("functions")
        mem.flag_weak_area("decorators")
        mem.session_summary = "Covered functions."

        data = mem.to_dict()
        restored = ConversationMemory.from_dict(data, max_history=10)

        assert restored.get_history() == mem.get_history()
        assert restored.learner_profile.name == "Bob"
        assert restored.learner_profile.level == "advanced"
        assert "functions" in restored.learner_profile.topics_covered
        assert "decorators" in restored.learner_profile.weak_areas
        assert restored.session_summary == "Covered functions."
