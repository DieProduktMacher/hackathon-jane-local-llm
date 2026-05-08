"""
FastAPI backend that exposes the LangGraph medical history intake agent.

Endpoints
─────────
POST /sessions
    Start a new intake session.
    Body: { "disease": "<condition name>" }
    Returns the first clinical question, or an error if the disease is unknown.

POST /sessions/{session_id}/messages
    Send the doctor's answer; receive the next question or the final summary.

GET  /sessions/{session_id}/status
    Check whether a session is still active.

GET  /diseases
    List all diseases available in the CSV database.

GET  /health
    Liveness probe.

Usage example (curl)
─────────────────────
    # 1. Start a session with the disease name
    RESP=$(curl -s -X POST http://localhost:8000/sessions \
                -H 'Content-Type: application/json' \
                -d '{"disease": "Migraine"}')
    SESSION=$(echo $RESP | jq -r .session_id)
    echo $RESP | jq -r .message   # → [1/5] Where exactly is the pain located?

    # 2. Keep answering until done=true
    curl -s -X POST http://localhost:8000/sessions/$SESSION/messages \
         -H 'Content-Type: application/json' \
         -d '{"message": "Right side of my head"}'
"""

import logging
import sys
import uuid
from pathlib import Path
from typing import Optional

# Ensure the backend package directory is on sys.path so that
# `from agent import ...` works regardless of the working directory.
sys.path.insert(0, str(Path(__file__).parent))

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from agent import (
    DISEASE_DATA,
    _SUPPORTED_DISEASES,
    get_pending_interrupt,
    session_reply,
    session_start,
)

# ── App setup ──────────────────────────────────────────────────────────────────

app = FastAPI(
    title="Medical History Intake Agent",
    description=(
        "A LangGraph-powered agent that guides a doctor through structured "
        "patient intake questions for a given disease."
    ),
    version="2.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

logger = logging.getLogger(__name__)

# Track which sessions have been started (guards against unknown IDs)
_active_sessions: set[str] = set()


# ── Pydantic models ────────────────────────────────────────────────────────────

class SessionCreateRequest(BaseModel):
    disease: Optional[str] = None


class SessionCreatedResponse(BaseModel):
    session_id: str
    message: str
    done: bool = False


class MessageRequest(BaseModel):
    message: str


class MessageResponse(BaseModel):
    message: str
    done: bool


class SessionStatusResponse(BaseModel):
    session_id: str
    active: bool
    pending_question: str | None


class DiseaseInfo(BaseModel):
    name: str
    questions_count: int
    symptoms: str
    tests: str
    treatments: str


class DiseasesResponse(BaseModel):
    diseases: list[DiseaseInfo]


# ── Endpoints ──────────────────────────────────────────────────────────────────

@app.post(
    "/sessions",
    response_model=SessionCreatedResponse,
    summary="Start a new intake session",
)
async def create_session(body: SessionCreateRequest) -> SessionCreatedResponse:
    """
    Creates a fresh LangGraph session and immediately processes the supplied
    disease name. Returns the first clinical question when the disease is
    recognised, or an error message (with done=true) when it is not.
    """
    if not body.disease.strip():
        raise HTTPException(
            status_code=422,
            detail=(
                f"Disease name is required. "
                f"Supported conditions: {_SUPPORTED_DISEASES}"
            ),
        )

    session_id = str(uuid.uuid4())
    try:
        message, done = session_start(session_id, body.disease)
    except Exception as exc:
        logger.exception("session_start raised an unexpected error")
        raise HTTPException(status_code=500, detail=f"Agent error: {exc}") from exc

    if not done:
        _active_sessions.add(session_id)

    return SessionCreatedResponse(session_id=session_id, message=message, done=done)


@app.post(
    "/sessions/{session_id}/messages",
    response_model=MessageResponse,
    summary="Send an answer and receive the next question or the summary",
)
async def post_message(session_id: str, body: MessageRequest) -> MessageResponse:
    """
    Resumes the LangGraph workflow with the doctor's answer.

    - While questions remain, returns the next question with `done=false`.
    - Once all questions are answered, returns the LLM-generated clinical
      summary with `done=true`.
    """
    if session_id not in _active_sessions:
        raise HTTPException(
            status_code=404,
            detail=f"Session '{session_id}' not found. Create one via POST /sessions.",
        )

    pending = get_pending_interrupt(session_id)
    if pending is None:
        raise HTTPException(
            status_code=409,
            detail="This session is already complete. Start a new one via POST /sessions.",
        )

    try:
        message, done = session_reply(session_id, body.message)
    except Exception as exc:
        logger.exception("session_reply raised an unexpected error")
        raise HTTPException(
            status_code=500,
            detail=f"Agent error: {exc}",
        ) from exc

    if done:
        _active_sessions.discard(session_id)

    return MessageResponse(message=message, done=done)


@app.get(
    "/sessions/{session_id}/status",
    response_model=SessionStatusResponse,
    summary="Check session status",
)
async def session_status(session_id: str) -> SessionStatusResponse:
    if session_id not in _active_sessions:
        raise HTTPException(status_code=404, detail=f"Session '{session_id}' not found.")
    pending = get_pending_interrupt(session_id)
    return SessionStatusResponse(
        session_id=session_id,
        active=pending is not None,
        pending_question=pending,
    )


@app.get(
    "/diseases",
    response_model=DiseasesResponse,
    summary="List all diseases in the database",
)
async def list_diseases() -> DiseasesResponse:
    return DiseasesResponse(
        diseases=[
            DiseaseInfo(
                name=v["name"],
                questions_count=len(v["questions"]),
                symptoms=v["symptoms"],
                tests=v["tests"],
                treatments=v["treatments"],
            )
            for v in DISEASE_DATA.values()
        ]
    )


@app.get("/health", summary="Liveness probe")
async def health() -> dict:
    return {"status": "ok"}
