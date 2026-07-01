# AI Study Assistant — Project Report
**Course:** B5779 – Applied Machine Learning, University of Bologna, 2025/2026  
**Authors:** Fatemeh Bozorgi  
**Date:** June 2026

---

## 1. Introduction

This project develops an **AI-powered Study Assistant** for the Applied Machine Learning course at the University of Bologna. The system allows students to interact with lecture materials through natural language: asking for explanations, generating practice quizzes, or requesting concise summaries. The assistant adapts to each student's knowledge gaps over time and actively checks its own responses for factual accuracy.

The system is built on four progressively advanced layers:

| Phase | Focus |
|-------|-------|
| 1 | Core multi-agent LangGraph pipeline with RAG |
| 2 | Student model memory and adaptive questioning |
| 3 | Critic agent and self-reflection loop |
| 4 | MCP tool integration and multi-agent scaling (planned) |

---

## 2. System Architecture

### 2.1 Technology Stack

| Component | Technology |
|-----------|-----------|
| Agent orchestration | LangGraph (StateGraph) |
| LLM backbone | Hermes 3 8B via Ollama (local) |
| Vector store | ChromaDB + nomic-embed-text embeddings |
| Document ingestion | LangChain PDF/Text loaders |
| Frontend | Streamlit |
| Conversation persistence | JSON files (UUID-keyed) |

### 2.2 Pipeline Overview

The system is structured as a directed graph of specialised nodes. The full pipeline is:

```
User Input
    │
    ▼
[Router]          ← keyword-based fast pre-classification
    │
    ▼
[Planner]         ← LLM classifies intent → mode, difficulty, RAG flag
    │
    ├─ greeting     → [Greeting Node] ──────────────────────────► [Memory] → END
    │
    ├─ course_info  → [Course Info Node] ← course_profile only ─┐
    │                  (no lecture routing, no RAG)              │
    │                                                            │
    ▼                                                            │
[Lecture Router]  ← identifies which lecture(s) are relevant     │
    │                                                            │
    ▼                                                            │
[RAG Node]        ← retrieves top-k relevant chunks from ChromaDB │
    │                                                            │
    ├─ explain / mixed → [Tutor Agent]                           │
    ├─ quiz            → [Quiz Agent]                            │
    └─ summarize       → [Summarize Agent]                       │
                              │                                  │
                              ▼                                  │
                        [Synthesizer]   ← formats output by mode │
                              │◄─────────────────────────────────┘
                              ▼
                        [Critic Agent]  ← hallucination check + self-reflection
                              │
                              ▼
                        [Memory Node]   ← updates chat history + weak topics
                              │
                              ▼
                            END
```

### 2.3 Shared State (TutorState)

All nodes communicate through a single typed state dictionary (`TutorState`), which carries:

```python
class TutorState(TypedDict):
    question: str
    route: str
    context: str               # joined RAG chunks
    answer: str
    course_profile: Dict
    chat_history: List[Dict]   # full conversation history
    lecture_route: Dict        # lecture IDs identified by LLM
    retrieved_chunks: List[str]
    plan: dict                 # mode, difficulty, needs_rag, focus, response_style
    student_memory: Dict       # persistent cross-session profile
    weak_topics: List[str]     # topics the student struggled with
    critic_passed: bool        # Phase 3 hallucination verdict
    reflection_count: int      # prevents infinite self-reflection loops
```

---

## 3. Phase 1 — Core Multi-Agent Pipeline

### 3.1 Router Node

A lightweight keyword-based classifier provides a fast first-pass decision. It routes to `quiz`, `summarize`, or `tutor` based on trigger words in the question. This avoids LLM latency for unambiguous intents.

> **Note (found during evaluation, §7):** `router_node` (`Agent/router.py`) is fully implemented but not currently registered as a node in `Agent/graph.py` — the compiled graph's entry point is `planner`, not `router`. The keyword pre-classification described here does not run in the current build; the Planner Node (§3.2) makes the routing decision alone. Either wire `router_node` in ahead of the planner (to skip the LLM call for unambiguous keyword matches) or remove it — leaving it disconnected risks the next reader assuming it's active.

### 3.2 Planner Node

The LLM-based planner performs intent classification and produces a structured plan:

```json
{
  "mode": "explain|quiz|summarize|mixed|greeting",
  "needs_rag": true,
  "difficulty": "beginner|intermediate|advanced",
  "focus": ["gradient descent", "CNN"],
  "response_style": "step-by-step"
}
```

The plan drives all downstream routing decisions. Difficulty and focus are passed to the specialist agents to shape their outputs.

### 3.3 Lecture Router Node

A second LLM call maps the question to specific lecture IDs from the course:

```json
{ "lecture_ids": [3, 4], "confidence": 0.85, "needs_general_course": false }
```

These IDs are used to filter the ChromaDB vector store, ensuring the RAG retrieval is scoped to the most relevant lecture material rather than searching across all five lectures.

### 3.4 RAG Node

The RAG node uses the lecture filter (or falls back to global search) to retrieve the top-5 most relevant text chunks from ChromaDB. The chunks are stored in `retrieved_chunks` and joined into `context`, which all downstream agents can read. If `needs_rag` is `false` (e.g., greetings), retrieval is skipped.

### 3.5 Specialist Agents

**Tutor Agent** — generates step-by-step explanations using the retrieved context. Constrained by a strict system prompt to avoid hallucination: if information is absent from the context, the tutor must say so explicitly.

**Quiz Agent** — generates multiple-choice questions in structured JSON format, which the synthesizer then renders for display. Difficulty is driven by the plan.

**Summarize Agent** — produces concise summaries of the retrieved lecture content.

### 3.6 Synthesizer Node

Attempts to parse the agent output as structured JSON. If successful, it applies mode-specific formatting (quiz layout, explanation headers, summary bullets). Plain-text answers pass through unchanged.

---

## 4. Phase 2 — Student Model Memory and Adaptive Questioning

### 4.1 Motivation

A static tutor applies the same strategy regardless of what a student already knows. The goal of Phase 2 is to build a lightweight **student model** — a persistent profile of the student's knowledge gaps — and use it to personalise quiz difficulty and topic selection.

### 4.2 Weak Topic Tracking

After every quiz session, the Memory Node calls the LLM to extract a concise topic label (e.g., `"gradient descent"`, `"lstm sequence modeling"`) from the quiz question. This label is appended to `weak_topics` if it is not already present.

```python
# memory.py (excerpt)
if plan.get("mode") == "quiz":
    topic = _extract_topic(state["question"], plan.get("focus", []))
    if topic and topic not in weak_topics:
        weak_topics = weak_topics + [topic]
```

The `student_memory` dict tracks:
- `weak_topics` — list of topic labels
- `quiz_topics_seen` — deduplicated set of all topics seen
- `total_quiz_sessions` — session counter

### 4.3 Adaptive Questioning

The Quiz Agent reads `weak_topics` and `total_quiz_sessions` from the state. Two adaptations are applied:

1. **Difficulty escalation** — after the first session, `beginner` is escalated to `intermediate`, and `intermediate` to `advanced`. This mirrors spaced repetition: topics the student has already encountered get progressively harder questions.

2. **Topic steering** — the prompt instructs the LLM to prioritise questions on the identified weak topics, forcing targeted practice rather than random sampling.

```python
# quiz.py (excerpt)
if weak_topics and total_sessions > 0:
    if difficulty == "beginner":
        difficulty = "intermediate"
    ...
adaptive_note = f"Student's previously identified weak topics: {', '.join(weak_topics)}. "
                "Prioritise questions that address these areas."
```

### 4.4 Persistence

`weak_topics` and `student_memory` are stored alongside conversation messages in a per-conversation JSON file and restored when the student re-opens an old conversation. The Streamlit sidebar also displays the current weak topic list as a live progress indicator.

---

## 5. Phase 3 — Critic Agent and Self-Reflection

### 5.1 Motivation

Retrieval-augmented generation reduces hallucination, but does not eliminate it. The LLM can still interpolate between documents, confuse lectures, or generate plausible-sounding but incorrect statements. Phase 3 adds a dedicated **Critic Agent** that independently verifies the tutor's output against the retrieved context.

### 5.2 Critic Node

After the Synthesizer formats the answer, the Critic Node calls the LLM with the following structured task:

> "Compare the answer against the context. Decide if the answer is GROUNDED (supported) or HALLUCINATED (contains unsupported claims). If hallucinated, produce a corrected answer."

The critic returns a JSON verdict:

```json
{
  "grounded": false,
  "issues": ["Claim X is not found in the context", "Lecture 3 does not discuss Y"],
  "revised_answer": "..."
}
```

Quiz answers are excluded from hallucination checking because they are structured JSON output with clearly defined correct answers.

### 5.3 Self-Reflection Loop

When the critic flags an answer as hallucinated, a **self-reflection** pass is triggered. The Critic Node invokes the LLM a second time with:

- The list of identified issues
- The original (flawed) answer
- The full retrieved context

The LLM is instructed to produce a corrected answer that stays strictly within the context and explicitly acknowledges any missing information.

To prevent runaway loops, `reflection_count` in the state limits reflection to **one revision**. If the answer is still flagged after one revision, a disclaimer is appended rather than looping again:

```
---
*Note: Some parts of this answer could not be fully verified against the lecture materials.*
```

This design choice prioritises system stability over exhaustive correction.

### 5.4 Impact

The critic adds one LLM call per non-quiz answer. In exchange:
- Clear contradictions with the lecture slides are caught and corrected
- The student receives a transparent signal when the system is uncertain
- The `critic_passed` flag in state enables future logging and analysis of hallucination rates

---

## 6. Data and Retrieval

### 6.1 Lecture Corpus

Five lectures from the Applied Machine Learning course were ingested:

| Lecture | Topics |
|---------|--------|
| 1 | ML basics, regression, logistic regression, bias-variance tradeoff |
| 2 | Neural network representations, forward propagation |
| 3 | MLP, backpropagation, gradient descent |
| 4 | CNN, convolution, pooling, feature maps |
| 6 | RNN, LSTM, sequence modeling |

### 6.2 Ingestion Pipeline

PDFs are loaded with `PyPDFLoader`, split into 800-character chunks with 100-character overlap, and embedded with `nomic-embed-text` via Ollama. Each chunk is tagged with its lecture number (`metadata["lecture"]`) to enable filtered retrieval.

### 6.3 Retrieval Strategy

The Lecture Router maps questions to specific lecture IDs using the LLM. The RAG Node then runs a similarity search filtered to those IDs, returning the top-5 most relevant chunks. This two-stage strategy (coarse lecture routing → fine chunk retrieval) reduces noise from cross-lecture interference.

---

## 7. Evaluation

Sections 3–6 describe the system's design; this section measures it. Two evaluations were run against the live system (Hermes 3 8B via Ollama, the populated ChromaDB store over Lectures 1, 2, 3, 4, 6): an **end-to-end pipeline evaluation** and an **isolated critic evaluation**. Code and data live in `eval/`; raw per-item outputs are saved to `eval/pipeline_eval_results.json` and `eval/critic_eval_results.json` for inspection.

### 7.1 Method

**Pipeline evaluation** (`eval/run_pipeline_eval.py`, `eval/dataset.jsonl`, n=19): a labelled set of questions is run through the full graph (`app.invoke`) exactly as the Streamlit UI would call it. The set covers two to three questions per lecture (mixing `explain`/`quiz`/`summarize` modes), one greeting, two **course-info** questions ("what is this course about", "how many credits / who teaches it"), and four **out-of-scope** questions on topics the corpus does not cover (transformers/attention, SVMs, reinforcement learning, exam logistics — Lecture 5 is absent from `data/raw`, and none of the ingested lectures touch these topics). For each case we check: planner mode accuracy, lecture-routing accuracy, keyword grounding (whether expected terms appear in the final answer), out-of-scope refusal rate (via lexical markers such as "could not find"), critic-pass rate, and latency.

**Critic evaluation** (`eval/run_critic_eval.py`, `eval/critic_cases.jsonl`, n=12): `critic_node` is called directly with hand-written `(context, answer)` pairs, bypassing retrieval and generation entirely. Six pairs are faithful paraphrases of the context (gold label: grounded); six contain a clearly fabricated or contradictory claim (gold label: hallucinated). This isolates the critic's classification accuracy from retrieval quality, which the pipeline evaluation above conflates.

This is a small, single-run, single-seed evaluation — it establishes rough behavioural bounds and surfaces qualitative failure modes, not statistically robust performance claims. The critic cases are also deliberately clear-cut (obvious contradictions rather than subtle misattributions), so its 100% score below should be read as "the critic catches blatant hallucinations," not "the critic is a reliable hallucination detector in general."

### 7.2 Results

| Metric | Result |
|---|---|
| Planner mode accuracy (n=19) | 100% |
| Lecture routing accuracy, in-scope (n=12) | 100% |
| Keyword grounding rate (n=10) | 100% |
| Out-of-scope refusal rate (n=4) | 75% (lexical marker match only, see 7.3) |
| Critic-pass rate (first pass, n=19) | 79% |
| Avg. end-to-end latency | 3.5s/question |
| Critic hallucination-detection accuracy (n=12) | 100% (P=1.00, R=1.00, F1=1.00) |

### 7.3 Qualitative Findings

The first evaluation run surfaced a real, user-facing bug: **the system could not answer meta-questions about the course itself** (e.g. "What is this course about?", "How many credits is it worth?"). Tracing it back:

1. The planner had no dedicated mode for course-level questions — it classified them as `explain`/`summarize` with `needs_rag=true`, so they were routed through **lecture retrieval** instead of the structured `data/course_profile.json` (which already contained the course name, code, credits, teachers, and topic list — the data existed, it just wasn't reachable from that code path).
2. The Lecture Router then forced a guess at which lecture(s) were relevant to a question that isn't about any specific lecture, and the RAG node pulled in unrelated chunks (e.g. a slide about TensorFlow Playground exercises).
3. Only `tutor_node` ever received `course_profile` in its prompt; `summarize_node` and `quiz_node` never did (and a second, unused `TUTOR_SYSTEM_PROMPT` in `prompts/tutor_prompt.py` — never imported by `tutor.py` — refers to an "OFFICIAL COURSE INFORMATION" block that no node actually populates). So depending on which specialist the planner picked, the model either had no course data at all, or had it available but buried behind irrelevant lecture text, and answered "I do not have enough information" or fabricated a plausible-sounding but wrong description from the irrelevant chunks.

**Fix:** added a dedicated `course_info` mode (`Agent/planner.py`) and a new `course_info_node` (`Agent/nodes/course_info.py`) that answers directly from `course_profile`, with `needs_rag=false` for this mode so it skips the Lecture Router and RAG node entirely (mirroring the existing `greeting` bypass). It still flows through `synthesizer → critic → memory`, with `context` set to the course-profile JSON so the critic has something concrete to check the answer against. After the fix, both course-info test questions are answered correctly and completely (course name, code, credits, teachers, and topic list all present, `critic_passed=True`), and planner mode accuracy rose from 94% to 100% because a previously-ambiguous exam-logistics question is now correctly classified as `course_info` too.

While fixing this, two more findings came from the eval harness itself, not the app:

- **The eval harness had the same bug it was measuring.** `run_pipeline_eval.py` originally initialised `course_profile: {}` instead of calling `load_course_profile()` — so the first eval run "confirmed" the course-info failure for the wrong reason (empty test fixture, not just a routing bug). This is a reminder that a harness which doesn't mirror production initialization exactly can produce misleading passes/failures; it's now fixed to call the same loader `App/streamlit_app.py` uses.
- **The lexical refusal/keyword checks are a blunt instrument.** Several correct, honest answers ("the JSON does not contain exam date information") didn't match the eval script's hardcoded refusal phrases until the marker list was extended. Any lexical proxy like this should be treated as a lower bound on correct behaviour, not a precise measurement — manual transcript spot-checks remain necessary.

Remaining findings from the original run still stand:

- **False refusals from imperfect lecture retrieval.** For two in-scope questions — "What is the bias-variance tradeoff?" (Lecture 1) and "What is gradient descent used for?" (Lecture 3) — the lecture router correctly identified the right lecture, but the top-5 similarity search returned chunks about a different subtopic (autodiff/backprop) and missed the section that actually answers the question. The tutor then correctly refused rather than hallucinating, but the student is left without an answer that does, in fact, exist in the corpus. This is a **retrieval recall** problem — the RAG design's main residual risk is under-answering, not over-answering. Increasing `k`, using smaller/overlapping chunks, or reranking would be the natural fix; it has not been implemented yet.
- **Self-reflection appears to correct genuine hallucinations, not just add disclaimers.** In the two false-refusal cases above, the *final* answer is an honest refusal, but `critic_passed=False` — meaning the critic rejected the tutor's original draft and the self-reflection pass rewrote it. Since the pipeline does not currently retain the pre-critic draft, the discarded first answer can't be inspected directly; logging it would let a future evaluation distinguish "critic corrected a real hallucination" from "critic was overly conservative on a correct answer."
- **The Lecture Router previously ran even when RAG was skipped** (e.g. on greetings), costing an unnecessary LLM call per turn. This is now fixed as a side effect of the `course_info` change: the planner's mode is checked immediately after planning, before the Lecture Router node, so `greeting` and `course_info` both bypass it entirely.

---

## 8. System Limitations

| Limitation | Description |
|------------|-------------|
| Single-model stack | All LLM calls use the same local Hermes 3 8B model. A larger or specialised model could improve planning and criticism quality. |
| Quiz self-evaluation | The system generates questions and checks explanations, but cannot evaluate whether a student's quiz answer is correct (no interactive quiz session). |
| Critic cost | Each answer now triggers two LLM calls (answer + critic). On a local 8B model this is manageable, but adds latency. |
| Weak topic labels | Topic extraction uses a short LLM prompt; the labels are sometimes imprecise or duplicate. A controlled vocabulary or embedding-based deduplication would improve reliability. |
| Phase 4 not implemented | MCP tool integration (web search, calculator, code execution) and multi-agent A2A collaboration were planned but are outside the scope of this project. |
| Retrieval recall (§7.3) | Top-5 similarity search within the routed lecture(s) sometimes misses the chunk that actually answers the question, causing false "not found" refusals on in-scope topics. Not yet fixed. |
| Dead prompt module | `prompts/tutor_prompt.py` defines a second, unused `TUTOR_SYSTEM_PROMPT` (richer than the one actually used in `Agent/nodes/tutor.py`) that is only imported by `summarize_node`, and references an "OFFICIAL COURSE INFORMATION" block that no node populates for it. Harmless today because `course_info` mode now handles course questions directly, but worth deleting or reconciling to avoid future confusion. |
| Evaluation scale | The evaluation in §7 uses n=19 (pipeline) and n=12 (critic) hand-written cases from a single run — sufficient to surface failure modes, not to bound error rates precisely. |

---

## 9. Conclusion

This project demonstrates how a multi-agent LangGraph architecture can deliver a coherent, adaptive tutoring experience grounded in course-specific lecture materials. The four-phase design allowed incremental complexity:

- **Phase 1** established the RAG-grounded, mode-aware pipeline with specialist agents for teaching, quizzing, and summarising.
- **Phase 2** introduced a student model that tracks weak topics across sessions and steers quiz difficulty and focus adaptively.
- **Phase 3** added a hallucination-checking critic with a one-pass self-reflection mechanism, making the system more honest and robust.

The fully local deployment (Ollama + ChromaDB) means the system can run without internet access or cloud API costs, making it practical for offline study environments. The Streamlit interface provides an accessible chat UI with conversation history and a live weak-topic tracker.

Future work should focus on interactive quiz evaluation (grading student answers and updating the weak-topic model accordingly), a larger backbone model, retrieval recall improvements (§7.3), and Phase 4 MCP tool integration for dynamic web search and code execution.

---

## Appendix: File Structure

```
study-assistant/
├── eval/
│   ├── dataset.jsonl               # labelled Q&A cases for pipeline eval
│   ├── critic_cases.jsonl          # grounded/hallucinated pairs for critic eval
│   ├── run_pipeline_eval.py        # end-to-end evaluation (Sec. 7)
│   └── run_critic_eval.py          # isolated critic evaluation (Sec. 7)
├── Agent/
│   ├── state.py             # TutorState TypedDict
│   ├── graph.py             # LangGraph pipeline definition
│   ├── planner.py           # LLM-based intent planner
│   ├── router.py            # keyword-based fast router
│   ├── lecture_router.py    # LLM-based lecture classifier
│   ├── course_metadata.py   # course constants
│   └── nodes/
│       ├── rag.py           # retrieval node
│       ├── tutor.py         # tutor agent
│       ├── quiz.py          # adaptive quiz agent
│       ├── summarize.py     # summarization agent
│       ├── course_info.py   # answers course-level questions from course_profile (§7.3)
│       ├── synthesizer.py   # output formatter
│       ├── critic.py        # hallucination checker + self-reflection (Phase 3)
│       └── memory.py        # chat history + weak topic tracker (Phase 2)
├── App/
│   ├── streamlit_app.py     # Streamlit chat UI
│   └── conversation_manager.py
├── rag/
│   ├── ingest.py            # PDF ingestion pipeline
│   ├── retriever.py         # ChromaDB wrapper
│   └── course_context.py
├── models/
│   └── llm.py               # Ollama LLM client
├── data/
│   ├── raw/                 # source PDFs (Lectures 1–4, 6)
│   └── chroma_db/           # persisted vector store
└── requirements.txt
```
