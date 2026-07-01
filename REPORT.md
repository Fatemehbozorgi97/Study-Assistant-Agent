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
[Router]          ← keyword-based fast pre-classification (implemented, not yet wired in — §3.1)
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

A lightweight keyword-based classifier was implemented to provide a fast first-pass decision, routing to `quiz`, `summarize`, or `tutor` based on trigger words in the question and avoiding LLM latency for unambiguous intents. In the current build this node is not yet wired into the compiled graph — the Planner Node (§3.2) currently makes the routing decision on its own — so it is kept in the codebase as a planned optimisation rather than an active component (see §8).

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

Sections 3–6 describe how the system was designed; this section describes how I tested whether it actually behaves that way. Design decisions like the critic agent or the two-stage retrieval strategy are easy to justify on paper, but the only way to know whether they hold up is to run the system on real questions and look closely at what comes back. I therefore built a small evaluation suite (`eval/`) and ran it against the live system — Hermes 3 8B through Ollama, with the ChromaDB store populated from Lectures 1, 2, 3, 4 and 6 — rather than relying on a handful of manual chat tests.

### 7.1 Method

I split the evaluation into two parts. The first is an **end-to-end pipeline test**: a set of 19 hand-written questions (`eval/dataset.jsonl`) covering two to three questions per lecture across the `explain`, `quiz` and `summarize` modes, a greeting, two questions about the course itself (e.g. "What is this course about?"), and four questions on topics the corpus does not cover at all (transformers/attention, SVMs, reinforcement learning, exam logistics — Lecture 5 was never ingested, and none of the five lectures I do have touch these topics). Each question is run through the full LangGraph pipeline exactly as the Streamlit app would call it, and I check the planner's mode classification, whether the lecture router picked the right lecture, whether the expected keywords show up in the final answer, whether out-of-scope questions get an honest refusal instead of a fabricated answer, whether the critic flags the answer, and how long the whole thing takes (`eval/run_pipeline_eval.py`).

The second part isolates the **critic agent** from the rest of the pipeline. Instead of relying on retrieval to produce a context, I wrote twelve `(context, answer)` pairs directly (`eval/critic_cases.jsonl`) — six where the answer is a faithful paraphrase of the context, and six where it contains an obviously fabricated or contradicted claim — and fed them straight into `critic_node`, bypassing the tutor and the vector store entirely (`eval/run_critic_eval.py`). This lets me measure whether the critic itself is doing its job, separately from whether retrieval gave it good material to check against.

I want to be upfront about the limits of this evaluation. Nineteen and twelve cases, run once, is enough to find failure modes and get a rough sense of where the system is solid and where it isn't — it is not enough to make a statistically confident claim about error rates. The critic test cases are also deliberately obvious (clear contradictions rather than subtle ones), so a perfect score there should be read as "the critic catches blatant hallucinations," not as proof that it would catch a more subtle one.

### 7.2 Results

| Metric | Result |
|---|---|
| Planner mode accuracy (n=19) | 100% |
| Lecture routing accuracy, in-scope (n=12) | 100% |
| Keyword grounding rate (n=10) | 100% |
| Out-of-scope refusal rate (n=4) | 75% (lower bound — see discussion below) |
| Critic-pass rate, first pass (n=19) | 79% |
| Avg. end-to-end latency | 3.5s per question |
| Critic hallucination-detection accuracy (n=12) | 100% (precision, recall and F1 all 1.00) |

### 7.3 Discussion

The most useful thing the first evaluation run did was surface a bug I hadn't noticed in manual testing: the assistant could not answer basic questions about the course itself, such as "What is this course about?" or "How many credits is it worth?", even though `data/course_profile.json` already contained the course name, code, credits, teachers and topic list. Manually testing this question afterwards in the running app confirmed it — the answer said it "could not find" this information in the lecture materials, even though the information was never really about the lectures at all.

Tracing through the pipeline showed why. The planner had no notion of a course-level question distinct from a lecture-level one, so it classified questions like these as `explain` or `summarize` with `needs_rag=true`, sending them through the lecture router and the RAG node instead of toward `course_profile`. The lecture router then had to guess which lecture a question with no lecture-specific content belonged to, and the retrieved chunks ended up being unrelated lecture snippets (in one run, a slide about a TensorFlow Playground exercise). To make things worse, only the tutor node's prompt actually included `course_profile` — the summarize and quiz nodes never received it — so depending on which specialist agent the planner happened to route to, the model either had no course information available at all, or had lecture noise crowding out the one useful fact it needed.

I fixed this by giving the planner a dedicated `course_info` mode and adding a new node, `course_info_node`, that answers directly from `course_profile` and skips the lecture router and RAG node completely, in the same way greetings already bypass retrieval. It still passes through the synthesizer and critic so its answers are checked for consistency, using the course profile itself as the grounding context. After the change, both course-info questions in the evaluation set are answered correctly and completely, and the exam-logistics question that used to be a borderline case is now classified correctly too — planner accuracy went from 94% to 100%. While tracking this down I also found that my own evaluation script had initialised `course_profile` as an empty dictionary rather than loading the real one, which meant an early test run "confirmed" the bug for a slightly different reason than the real one; correcting the evaluation script to load the profile the same way the Streamlit app does was necessary before the numbers in §7.2 could be trusted.

Two other things came out of this round of testing. First, on two in-scope questions ("What is the bias-variance tradeoff?", "What is gradient descent used for?") the lecture router picked the correct lecture, but the top-5 similarity search returned chunks about a related but different subtopic and missed the section that actually answers the question. The tutor correctly said it could not find the answer rather than guessing, so no hallucination reached the student, but the answer does exist somewhere in that lecture's slides and the student is left without it. This points to retrieval recall, not the critic or the anti-hallucination prompting, as the system's main remaining weak point — a larger `k`, smaller overlapping chunks, or a reranking step would be the natural next step, though I have not implemented this yet. Second, in reviewing the transcripts for these two cases, I noticed that although the *final* answer given to the student was an honest refusal, `critic_passed` was `False` for both — meaning the critic had rejected the tutor's first draft and the self-reflection pass rewrote it into the refusal. I don't currently log the discarded first draft anywhere, so I can't directly confirm what it originally said, but it is a reasonable sign that the self-reflection mechanism is doing more than adding a disclaimer — a natural follow-up would be to log the pre-critic draft so this can be checked directly in a future evaluation.

Fixing the course-info routing had a smaller side benefit: previously the lecture router ran on every question, including greetings, even though its output was never used when RAG was skipped. Moving the mode check to right after the planner (so `greeting` and `course_info` branch away before reaching the lecture router) removed this wasted call and is part of why average latency dropped from 4.8s to 3.5s per question between the two evaluation runs.

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
| Router node unused | `Agent/router.py` implements a keyword-based fast router (§3.1) but it was never wired into the compiled graph, so it currently has no effect on runtime behaviour. Left in place as a planned optimisation. |
| Evaluation scale | The evaluation in §7 uses 19 (pipeline) and 12 (critic) hand-written cases from a single run — enough to surface failure modes, not enough to bound error rates precisely. |

---

## 9. Conclusion

This project demonstrates how a multi-agent LangGraph architecture can deliver a coherent, adaptive tutoring experience grounded in course-specific lecture materials. The four-phase design allowed incremental complexity:

- **Phase 1** established the RAG-grounded, mode-aware pipeline with specialist agents for teaching, quizzing, and summarising.
- **Phase 2** introduced a student model that tracks weak topics across sessions and steers quiz difficulty and focus adaptively.
- **Phase 3** added a hallucination-checking critic with a one-pass self-reflection mechanism, making the system more honest and robust.

The evaluation in §7 was a useful check on these design claims rather than just a formality: testing the system on a small but deliberately varied set of questions found a genuine gap (course-level questions were not being answered correctly) that manual chat testing during development had missed, and fixing it also removed an unnecessary LLM call on every greeting. This suggests that even a small, targeted test set is worth building for a system like this, since agentic pipelines have enough moving parts that a bug in one node's inputs is not always obvious from the outside.

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
