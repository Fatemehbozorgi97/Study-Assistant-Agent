import streamlit as st
import sys
import os
import json
# -----------------------------------
# fix imports
# -----------------------------------

sys.path.append(
    os.path.dirname(os.path.dirname(__file__))
)

from App.conversation_manager import (
    create_conversation,
    save_conversation,
    load_conversation,
    list_conversations
)

from Agent.graph import app
from rag.course_context import load_course_profile


# -----------------------------------
# page config
# -----------------------------------

st.set_page_config(
    page_title="AI Study Assistant",
    page_icon="📚",
    layout="wide"
)

st.title("📚 AI Study Assistant (LangGraph Tutor)")


# ===================================
# SIDEBAR
# ===================================

st.sidebar.title("💬 Conversations")


# -----------------------------------
# initialize conversation
# -----------------------------------

if "current_conversation" not in st.session_state:

    conv_id = create_conversation()

    st.session_state.current_conversation = conv_id

    st.session_state.messages = []

    st.session_state.student_memory = {}

    st.session_state.weak_topics = []


# -----------------------------------
# new conversation button
# -----------------------------------

if st.sidebar.button("+ New Chat"):

    conv_id = create_conversation()

    st.session_state.current_conversation = conv_id

    st.session_state.messages = []

    st.session_state.student_memory = {}

    st.session_state.weak_topics = []

    st.rerun()


# -----------------------------------
# list previous conversations
# -----------------------------------

conversations = list_conversations()

for conv in conversations:

    if st.sidebar.button(
        conv["title"],
        key=conv["id"]
    ):

        data = load_conversation(conv["id"])

        st.session_state.current_conversation = conv["id"]

        st.session_state.messages = data.get("messages", [])

        st.session_state.student_memory = data.get("student_memory", {})

        st.session_state.weak_topics = data.get("weak_topics", [])

        st.rerun()


# ===================================
# RENDER CHAT HISTORY
# ===================================

for msg in st.session_state.messages:

    with st.chat_message(msg["role"]):

        st.markdown(msg["content"])


# ===================================
# USER INPUT
# ===================================

user_input = st.chat_input(
    "Ask your study question..."
)


if user_input:

    # -----------------------------------
    # add user message
    # -----------------------------------

    st.session_state.messages.append({
        "role": "user",
        "content": user_input
    })

    with st.chat_message("user"):

        st.markdown(user_input)

    # -----------------------------------
    # run LangGraph
    # -----------------------------------

    with st.spinner("Thinking..."):

        result = app.invoke({
            "question": user_input,
            "context": "",
            "answer": "",
            "retrieved_chunks": [],
            "plan": {},
            "lecture_route": {},
            "course_profile": load_course_profile(),
            "chat_history": st.session_state.messages,
            "student_memory": st.session_state.student_memory,
            "weak_topics": st.session_state.weak_topics,
            "critic_passed": True,
            "reflection_count": 0,
        })

    answer = result["answer"]

    # -----------------------------------
    # add assistant message
    # -----------------------------------

    st.session_state.messages.append({
        "role": "assistant",
        "content": answer
    })

    with st.chat_message("assistant"):

        st.markdown(answer)

    # -----------------------------------
    # update student model memory
    # -----------------------------------

    st.session_state.student_memory = result.get("student_memory", {})
    st.session_state.weak_topics = result.get("weak_topics", [])

    # show weak topic tracker in sidebar
    if st.session_state.weak_topics:
        st.sidebar.markdown("---")
        st.sidebar.markdown("**Your weak topics:**")
        for t in st.session_state.weak_topics:
            st.sidebar.markdown(f"- {t}")

    # -----------------------------------
    # save conversation
    # -----------------------------------

    conversation_data = {
        "id": st.session_state.current_conversation,
        "title": (
            st.session_state.messages[0]["content"][:40]
            if st.session_state.messages
            else "New Conversation"
        ),
        "messages": st.session_state.messages,
        "student_memory": st.session_state.student_memory,
        "weak_topics": st.session_state.weak_topics,
    }

    save_conversation(
        st.session_state.current_conversation,
        conversation_data
    )

