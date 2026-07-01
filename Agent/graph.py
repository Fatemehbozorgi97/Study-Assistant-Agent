from langgraph.graph import StateGraph, END

from Agent.state import TutorState
from Agent.planner import planner_node
from Agent.lecture_router import lecture_router_node

from Agent.nodes.rag import rag_node
from Agent.nodes.greeting import greeting_node
from Agent.nodes.course_info import course_info_node
from Agent.nodes.tutor import tutor_node
from Agent.nodes.quiz import quiz_node
from Agent.nodes.summarize import summarize_node
from Agent.nodes.synthesizer import synthesizer_node
from Agent.nodes.critic import critic_node
from Agent.nodes.memory import memory_node


graph = StateGraph(TutorState)

# ─── core routing ─────────────────────────────────────────────
graph.add_node("planner", planner_node)
graph.add_node("lecture_router", lecture_router_node)

# ─── Phase 1: specialist agents ───────────────────────────────
graph.add_node("greeting", greeting_node)
graph.add_node("course_info", course_info_node)
graph.add_node("rag", rag_node)
graph.add_node("tutor", tutor_node)
graph.add_node("quiz", quiz_node)
graph.add_node("summarize", summarize_node)
graph.add_node("synthesizer", synthesizer_node)

# ─── Phase 3: critic + self-reflection ────────────────────────
graph.add_node("critic", critic_node)

# ─── Phase 2: student model memory ────────────────────────────
graph.add_node("memory", memory_node)


# ─── edges ────────────────────────────────────────────────────

graph.set_entry_point("planner")

# Greetings and course-info questions don't need lecture routing or RAG —
# they're answered without touching the vector store, so branch before
# lecture_router rather than after it.
graph.add_conditional_edges(
    "planner",
    lambda s: s["plan"]["mode"],
    {
        "explain":     "lecture_router",
        "quiz":        "lecture_router",
        "summarize":   "lecture_router",
        "mixed":       "lecture_router",
        "greeting":    "greeting",
        "course_info": "course_info",
    }
)

graph.add_conditional_edges(
    "lecture_router",
    lambda s: s["plan"]["mode"],
    {
        "explain":   "rag",
        "quiz":      "rag",
        "summarize": "rag",
        "mixed":     "rag",
    }
)

# greeting bypasses RAG/agents — goes straight to memory
graph.add_edge("greeting", "memory")

# course_info bypasses RAG/lecture routing but still runs through the
# synthesizer/critic so its factual claims are checked against course_profile
graph.add_edge("course_info", "synthesizer")

# After RAG, dispatch to the appropriate specialist agent
graph.add_conditional_edges(
    "rag",
    lambda s: s["plan"]["mode"],
    {
        "explain":   "tutor",
        "quiz":      "quiz",
        "summarize": "summarize",
        "mixed":     "tutor",
    }
)

# All agents flow into synthesizer, then critic, then memory
graph.add_edge("tutor",     "synthesizer")
graph.add_edge("quiz",      "synthesizer")
graph.add_edge("summarize", "synthesizer")

graph.add_edge("synthesizer", "critic")
graph.add_edge("critic",      "memory")
graph.add_edge("memory",      END)


app = graph.compile()
