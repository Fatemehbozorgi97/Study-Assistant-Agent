from typing import TypedDict, List, Dict, Any, Optional


class TutorState(TypedDict):
    question: str

    context: str
    answer: str

    # course awareness
    course_profile: Dict[str, Any]

    # memory system
    chat_history: List[Dict[str, str]]
    lecture_route: Dict[str, Any]
    retrieved_chunks: List[str]

    plan: dict

    # Phase 2: student model memory
    student_memory: Dict[str, Any]   # persisted across sessions in Streamlit
    weak_topics: List[str]           # topics the student struggles with

    # Phase 3: critic / self-reflection
    critic_passed: bool
    reflection_count: int            # guard against infinite self-reflection loops