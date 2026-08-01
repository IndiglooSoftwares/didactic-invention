"""Memory management for the Agentic Mentor."""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import List, Optional


@dataclass
class Message:
    """A single message in the conversation."""

    role: str  # "user", "assistant", or "system"
    content: str
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def to_dict(self) -> dict:
        return {"role": self.role, "content": self.content}


@dataclass
class LearnerProfile:
    """Persistent profile tracking the learner's progress."""

    name: str = "Learner"
    level: str = "beginner"  # beginner | intermediate | advanced
    topics_covered: List[str] = field(default_factory=list)
    mastered_concepts: List[str] = field(default_factory=list)
    weak_areas: List[str] = field(default_factory=list)
    session_count: int = 0


class ConversationMemory:
    """Manages conversation history and learner context.

    Keeps a sliding window of recent messages (to stay within token limits)
    while maintaining a separate long-term summary of past sessions.
    """

    def __init__(self, max_history: int = 20) -> None:
        self.max_history = max_history
        self._messages: List[Message] = []
        self.learner_profile = LearnerProfile()
        self.session_summary: str = ""

    # ------------------------------------------------------------------
    # Message management
    # ------------------------------------------------------------------

    def add_message(self, role: str, content: str) -> None:
        """Append a message and trim history to the sliding window."""
        self._messages.append(Message(role=role, content=content))
        if len(self._messages) > self.max_history:
            self._messages = self._messages[-self.max_history :]

    def get_history(self) -> List[dict]:
        """Return the conversation history as a list of role/content dicts."""
        return [m.to_dict() for m in self._messages]

    def clear_history(self) -> None:
        """Clear current conversation history (keeps learner profile)."""
        self._messages = []

    # ------------------------------------------------------------------
    # Learner profile helpers
    # ------------------------------------------------------------------

    def update_level(self, level: str) -> None:
        """Update the learner's assessed knowledge level."""
        valid = {"beginner", "intermediate", "advanced"}
        if level not in valid:
            raise ValueError(f"level must be one of {valid}")
        self.learner_profile.level = level

    def mark_topic_covered(self, topic: str) -> None:
        if topic not in self.learner_profile.topics_covered:
            self.learner_profile.topics_covered.append(topic)

    def mark_concept_mastered(self, concept: str) -> None:
        if concept not in self.learner_profile.mastered_concepts:
            self.learner_profile.mastered_concepts.append(concept)
        if concept in self.learner_profile.weak_areas:
            self.learner_profile.weak_areas.remove(concept)

    def flag_weak_area(self, concept: str) -> None:
        if concept not in self.learner_profile.weak_areas:
            self.learner_profile.weak_areas.append(concept)

    def increment_session(self) -> None:
        self.learner_profile.session_count += 1

    # ------------------------------------------------------------------
    # Serialisation helpers (for persistence between sessions)
    # ------------------------------------------------------------------

    def to_dict(self) -> dict:
        return {
            "messages": [m.to_dict() for m in self._messages],
            "learner_profile": {
                "name": self.learner_profile.name,
                "level": self.learner_profile.level,
                "topics_covered": self.learner_profile.topics_covered,
                "mastered_concepts": self.learner_profile.mastered_concepts,
                "weak_areas": self.learner_profile.weak_areas,
                "session_count": self.learner_profile.session_count,
            },
            "session_summary": self.session_summary,
        }

    @classmethod
    def from_dict(cls, data: dict, max_history: int = 20) -> "ConversationMemory":
        memory = cls(max_history=max_history)
        for msg in data.get("messages", []):
            memory.add_message(msg["role"], msg["content"])
        profile_data = data.get("learner_profile", {})
        memory.learner_profile.name = profile_data.get("name", "Learner")
        memory.learner_profile.level = profile_data.get("level", "beginner")
        memory.learner_profile.topics_covered = profile_data.get("topics_covered", [])
        memory.learner_profile.mastered_concepts = profile_data.get(
            "mastered_concepts", []
        )
        memory.learner_profile.weak_areas = profile_data.get("weak_areas", [])
        memory.learner_profile.session_count = profile_data.get("session_count", 0)
        memory.session_summary = data.get("session_summary", "")
        return memory
