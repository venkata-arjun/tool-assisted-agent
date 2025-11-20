# app.py

# --- 0. PACKAGE IMPORTS & SETUP ---

from fastapi import FastAPI
from pydantic import BaseModel
import os
import re
from typing import List, Dict, Any, Annotated
import asyncio
import uuid
from datetime import datetime

# LangChain & LangGraph Imports
from langchain_groq import ChatGroq
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import StateGraph, START, END
from langchain_core.messages import HumanMessage, AIMessage
from typing import TypedDict
import operator


# --- 1. CONFIGURATION & INITIALIZATION ---

# Better API Key handling
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
if not GROQ_API_KEY:
    print("Warning: GROQ_API_KEY environment variable not set.")
    print("Please set it using: set GROQ_API_KEY=your_key_here")

# LLM SETUP
try:
    llm = ChatGroq(
        model_name="llama-3.3-70b-versatile", temperature=0.2, api_key=GROQ_API_KEY
    )
    print("LLM Initialized successfully.")
except Exception as e:
    print(f"Error initializing LLM: {e}")
    llm = None


# Define State
class ChatState(TypedDict):
    messages: Annotated[List[Any], operator.add]
    user_input: str
    response: str
    thread_id: str


# --- 2. HELPER FUNCTIONS ---


def get_recent_history(messages: List) -> str:
    """Returns the latest conversation as a simple text string for prompts"""
    try:
        if not messages:
            return "No conversation history yet."

        lines = []
        for m in messages:
            if isinstance(m, HumanMessage):
                lines.append(f"USER: {m.content}")
            elif isinstance(m, AIMessage):
                lines.append(f"BOT: {m.content}")

        return "\n".join(lines[-10:])  # Last 10 messages
    except Exception as e:
        print(f"Error loading memory: {e}")
        return "No conversation history yet."


def grade_from_score(score: float) -> str:
    """Convert numeric marks to letter grade (Indian-style)"""
    if score >= 90:
        return "A+"
    elif score >= 80:
        return "A"
    elif score >= 70:
        return "B"
    elif score >= 60:
        return "C"
    elif score >= 50:
        return "D"
    elif score >= 40:
        return "E"
    else:
        return "F"


# --- 3. TOOL FUNCTIONS ---


def positive_prompt_tool(state: ChatState) -> Dict[str, Any]:
    """Handle positive/constructive messages"""
    try:
        history = get_recent_history(state["messages"])
        user_input = state["user_input"]

        prompt = f"""You are a warm, natural human assistant.
Previous conversation:
{history}

Current message: {user_input}
Reply briefly and encouragingly. Never repeat the user."""

        response = llm.invoke(prompt)
        return {
            "response": response.content,
            "messages": [AIMessage(content=response.content)],
        }
    except Exception as e:
        print(f"Error in positive_prompt_tool: {e}")
        return {
            "response": "I'm having trouble responding right now. Please try again.",
            "messages": [
                AIMessage(
                    content="I'm having trouble responding right now. Please try again."
                )
            ],
        }


def negative_prompt_tool(state: ChatState) -> Dict[str, Any]:
    """Handle complaints, worries, or negative messages"""
    try:
        history = get_recent_history(state["messages"])
        user_input = state["user_input"]

        prompt = f"""You are a calm, practical assistant.
Previous conversation:
{history}

Current message: {user_input}
Give short, helpful advice or empathy without repeating the user."""

        response = llm.invoke(prompt)
        return {
            "response": response.content,
            "messages": [AIMessage(content=response.content)],
        }
    except Exception as e:
        print(f"Error in negative_prompt_tool: {e}")
        return {
            "response": "I'm having trouble responding right now. Please try again.",
            "messages": [
                AIMessage(
                    content="I'm having trouble responding right now. Please try again."
                )
            ],
        }


def self_harm_safety_tool(state: ChatState) -> Dict[str, Any]:
    """Handle self-harm related messages with emergency contacts"""
    safety_response = (
        "I can't help with self-harm. If you're in danger, please call emergency services immediately. "
        "In India → Aasra: +91 9820466726. Reach out to someone you trust right now."
    )
    return {
        "response": safety_response,
        "messages": [AIMessage(content=safety_response)],
    }


def student_marks_tool(state: ChatState) -> Dict[str, Any]:
    """Handle academic queries about marks, grades, and averages"""
    try:
        history = get_recent_history(state["messages"])
        user_input = state["user_input"]

        prompt = f"""
You are an expert academic helper that remembers everything from the conversation.

Full conversation so far:
{history}

Current user message: {user_input}

Tasks:
• Extract & remember all marks/subjects mentioned (update if user corrects).
• Handle follow-ups naturally ("what was the average?", "add 5 marks to Alice", "who passed?" etc.).
• Use the grading system: A+ (90+), A (80-89), B (70-79), C (60-69), D (50-59), E (40-49), F (<40).
• Always reply in short, natural English (example: "Alice: 92 → A+ | Bob: 78 → B | Average: 85.0").

If nothing in memory yet and no marks found → politely ask for them.
"""
        response = llm.invoke(prompt)
        return {
            "response": response.content,
            "messages": [AIMessage(content=response.content)],
        }
    except Exception as e:
        print(f"Error in student_marks_tool: {e}")
        error_msg = "I'm having trouble processing academic queries right now. Please try again."
        return {"response": error_msg, "messages": [AIMessage(content=error_msg)]}


# --- 4. LANGGRAPH WORKFLOW SETUP ---


def router(state: ChatState) -> str:
    """Route the conversation to the appropriate tool"""
    user_input = state["user_input"].lower()

    # 1. Safety block (instant response - highest priority)
    safety_phrases = [
        "end his life",
        "end my life",
        "kill myself",
        "suicide",
        "want to die",
        "self harm",
        "self-harm",
        "harm myself",
        "end it all",
        "don't want to live",
    ]

    if any(phrase in user_input for phrase in safety_phrases):
        return "safety"

    # 2. Academic / marks keywords or numbers
    academic_keywords = [
        "score",
        "mark",
        "grade",
        "average",
        "total",
        "biology",
        "physics",
        "maths",
        "chemistry",
        "science",
        "english",
        "history",
        "geography",
        "economics",
        "accountancy",
        "what was",
        "how much",
        "again",
        "list",
        "add ",
        "got in",
        "scored in",
        "subject",
        "exam",
        "test",
        "result",
        "percentage",
        "pass",
        "fail",
        "rank",
    ]

    if any(word in user_input for word in academic_keywords) or re.search(
        r"\d", state["user_input"]
    ):
        return "academic"

    # 3. Emotional messages
    positive_words = [
        "happy",
        "amazing",
        "great",
        "beautiful",
        "wow",
        "love",
        "excited",
        "good",
        "awesome",
        "fantastic",
        "wonderful",
        "excellent",
        "perfect",
        "brilliant",
    ]

    negative_words = [
        "sad",
        "angry",
        "upset",
        "worried",
        "anxious",
        "stressed",
        "frustrated",
        "annoyed",
        "depressed",
        "tired",
        "exhausted",
        "bored",
        "lonely",
        "scared",
    ]

    if any(pos in user_input for pos in positive_words):
        return "positive"
    elif any(neg in user_input for neg in negative_words):
        return "negative"
    else:
        # Default to positive response for neutral messages
        return "positive"


# Create the workflow
workflow = StateGraph(ChatState)

# Define nodes
workflow.add_node("router", lambda state: state)  # Just pass through for routing
workflow.add_node("positive", positive_prompt_tool)
workflow.add_node("negative", negative_prompt_tool)
workflow.add_node("academic", student_marks_tool)
workflow.add_node("safety", self_harm_safety_tool)

# Define edges - FIXED: Use the router function directly in conditional edges
workflow.add_conditional_edges(
    "router",
    router,  # Direct function reference
    {
        "positive": "positive",
        "negative": "negative",
        "academic": "academic",
        "safety": "safety",
    },
)

workflow.add_edge("positive", END)
workflow.add_edge("negative", END)
workflow.add_edge("academic", END)
workflow.add_edge("safety", END)

workflow.set_entry_point("router")

# Create checkpointer (replaces memory)
checkpointer = InMemorySaver()

# Compile the graph
graph = workflow.compile(checkpointer=checkpointer)
print("LangGraph workflow initialized with memory checkpointer.")


# --- 5. FASTAPI SETUP ---

app = FastAPI(
    title="LangGraph Chat Agent API",
    description="A smart chat agent using LangGraph with conversation memory",
    version="1.0.0",
)


# Pydantic Models for Request/Response
class ChatRequest(BaseModel):
    query: str
    user_name: str = "User"
    thread_id: str = "default"


class ChatResponse(BaseModel):
    response: str
    history: List[dict]
    thread_id: str


class ThreadResponse(BaseModel):
    thread_id: str
    message: str


# Thread management
active_threads: Dict[str, Dict] = {}


@app.post("/chat", response_model=ChatResponse, tags=["Chat"])
async def chat_endpoint(request: ChatRequest):
    """
    Handles a user chat message and maintains conversation history using LangGraph.
    """
    user_input = request.query.strip()
    thread_id = request.thread_id

    if not user_input:
        return ChatResponse(
            response="Please provide a non-empty message.",
            history=[],
            thread_id=thread_id,
        )

    try:
        # Prepare input for the graph
        input_data = {
            "user_input": user_input,
            "messages": [HumanMessage(content=user_input)],
            "response": "",
            "thread_id": thread_id,
        }

        config = {"configurable": {"thread_id": thread_id}}

        # Invoke the graph
        result = graph.invoke(input_data, config)

        # Get conversation history
        thread_config = {"configurable": {"thread_id": thread_id}}
        snapshot = checkpointer.get_tuple(thread_config)

        history_dicts = []
        if snapshot and hasattr(snapshot, "checkpoint") and snapshot.checkpoint:
            for msg in snapshot.checkpoint.get("messages", []):
                if isinstance(msg, HumanMessage):
                    history_dicts.append({"role": "user", "content": msg.content})
                elif isinstance(msg, AIMessage):
                    history_dicts.append({"role": "bot", "content": msg.content})

        return ChatResponse(
            response=result["response"], history=history_dicts, thread_id=thread_id
        )

    except Exception as e:
        error_msg = f"Sorry, I encountered an error: {str(e)}"
        print(f"Error in chat_endpoint: {e}")
        return ChatResponse(response=error_msg, history=[], thread_id=thread_id)


@app.post("/threads/create", response_model=ThreadResponse, tags=["Threads"])
async def create_thread():
    """Create a new conversation thread"""
    thread_id = str(uuid.uuid4())
    active_threads[thread_id] = {
        "created_at": datetime.now().isoformat(),
        "message_count": 0,
    }
    return ThreadResponse(
        thread_id=thread_id, message="New conversation thread created"
    )


@app.delete("/threads/{thread_id}", tags=["Threads"])
async def delete_thread(thread_id: str):
    """Delete a conversation thread"""
    try:
        # Clear from checkpointer
        config = {"configurable": {"thread_id": thread_id}}
        checkpointer.clear_tuple(config)

        # Remove from active threads
        if thread_id in active_threads:
            del active_threads[thread_id]

        return {"status": "success", "message": f"Thread {thread_id} deleted"}
    except Exception as e:
        return {"status": "error", "message": str(e)}


@app.get("/threads", tags=["Threads"])
async def list_threads():
    """List all active conversation threads"""
    return {"threads": active_threads, "total_threads": len(active_threads)}


@app.get("/health", tags=["Health"])
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "service": "LangGraph Chat API",
        "active_threads": len(active_threads),
        "memory_backend": "InMemorySaver",
    }


@app.get("/", tags=["Root"])
async def root():
    """Root endpoint with API information"""
    return {
        "message": "LangGraph Chat Agent API",
        "version": "1.0.0",
        "endpoints": {
            "chat": "/chat (POST)",
            "create_thread": "/threads/create (POST)",
            "list_threads": "/threads (GET)",
            "delete_thread": "/threads/{thread_id} (DELETE)",
            "health": "/health (GET)",
            "docs": "/docs (GET)",
        },
        "features": [
            "Academic grade tracking and analysis",
            "Emotional support for positive/negative messages",
            "Safety protocols for self-harm concerns",
            "LangGraph conversation memory with checkpoints",
            "Multi-thread conversation support",
        ],
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
