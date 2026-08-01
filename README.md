# didactic-invention

An **Agentic Mentor** — an autonomous, tool-using AI tutor that adapts to each learner's pace and knowledge level.

## Features

| Feature | Description |
|---|---|
| **Adaptive assessment** | Opens every topic with a diagnostic conversation to calibrate depth |
| **Structured explanations** | One-sentence summary → why it matters → core idea → example → pitfalls |
| **Socratic quizzing** | Multiple formats (MC, T/F, short answer) targeting known misconceptions |
| **Answer evaluation** | Constructive, level-appropriate feedback on every learner response |
| **Resource suggestions** | Curated free resources matched to the learner's level |
| **Session memory** | Sliding-window conversation history + persistent learner profile |
| **Session summaries** | End-of-session recap with progress notes and next steps |
| **Streaming support** | Opt-in token-by-token streaming for low-latency UX |
| **Provider-agnostic** | Swap in OpenAI, Anthropic, Ollama, or any custom LLM client |

## Project layout

```
.
├── main.py                  # CLI entry point
├── requirements.txt
├── .env.example             # Environment variable reference
├── src/
│   ├── config.py            # LLMConfig + MentorConfig (env-driven)
│   └── agent/
│       ├── mentor.py        # AgenticMentor — the core agent class
│       ├── tools.py         # MentorTools — explain / quiz / evaluate / resources / summarise
│       ├── memory.py        # ConversationMemory + LearnerProfile
│       └── prompts.py       # System & tool prompts
└── tests/
    ├── test_mentor.py
    ├── test_tools.py
    ├── test_memory.py
    └── test_config.py
```

## Quick start

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Configure environment

```bash
cp .env.example .env
# Edit .env and set LLM_API_KEY (and any other settings you need)
```

### 3. Run the CLI

```bash
python main.py
```

#### CLI commands

| Command | Description |
|---|---|
| `/assess <topic>` | Start a structured assessment for *topic* |
| `/explain <topic>` | Get a detailed explanation of *topic* |
| `/quiz <topic>` | Take a short quiz on *topic* |
| `/resources <topic>` | Get resource suggestions for *topic* |
| `/end` | Summarise the session and save progress |
| `/quit` | Exit the CLI |

Any other input is treated as a free-form chat message.

## Using the Python API

```python
from src.agent import AgenticMentor
from src.config import MentorConfig

# Build config from environment variables
config = MentorConfig.from_env()

# Bring your own LLM client (must expose .complete_with_messages and .complete)
from openai import OpenAI
raw_client = OpenAI(api_key=config.llm.api_key)

class MyAdapter:
    def complete_with_messages(self, messages):
        resp = raw_client.chat.completions.create(
            model=config.llm.model, messages=messages
        )
        return resp.choices[0].message.content

    def complete(self, prompt):
        return self.complete_with_messages([{"role": "user", "content": prompt}])

    def stream(self, messages):
        for chunk in raw_client.chat.completions.create(
            model=config.llm.model, messages=messages, stream=True
        ):
            delta = chunk.choices[0].delta
            if delta.content:
                yield delta.content

mentor = AgenticMentor(llm_client=MyAdapter(), max_history=config.max_history)

# Free-form chat
print(mentor.chat("I want to learn about recursion"))

# Structured flows
print(mentor.start_assessment("Python decorators"))
print(mentor.explain("closures"))
print(mentor.quiz("sorting algorithms", num_questions=3))
print(mentor.evaluate("What is O(n log n)?", "log-linear time", "log n time"))
print(mentor.suggest_resources("machine learning"))
print(mentor.end_session())
```

## Adding a new LLM provider

1. Create an adapter class with `complete(prompt) -> str`, `complete_with_messages(messages) -> str`, and `stream(messages) -> Iterator[str]` methods.
2. Register it in the `_make_llm_client` factory in `main.py` (or pass it directly to `AgenticMentor`).

## Running tests

```bash
python -m pytest tests/ -v
```

## Environment variables

| Variable | Default | Description |
|---|---|---|
| `LLM_PROVIDER` | `openai` | Provider: `openai` \| `anthropic` \| `ollama` \| `custom` |
| `LLM_MODEL` | `gpt-4o` | Model identifier |
| `LLM_API_KEY` | — | API key for the provider |
| `LLM_BASE_URL` | — | Optional base URL override |
| `LLM_TEMP` | `0.7` | Sampling temperature (0.0–2.0) |
| `LLM_MAX_TOKENS` | `2048` | Max tokens per response |
| `LLM_STREAM` | `false` | Enable streaming (`true`/`false`) |
| `MAX_HISTORY` | `20` | Conversation window size |
| `SESSION_FILE` | — | Path for JSON session persistence |
| `LOG_LEVEL` | `INFO` | Logging level |

## License

See [LICENSE](LICENSE).
