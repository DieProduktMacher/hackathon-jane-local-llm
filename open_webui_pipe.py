"""
OpenWebUI Pipe — Medical History Intake Agent
─────────────────────────────────────────────
Routes messages from an OpenWebUI chat to the LangGraph-powered FastAPI
backend (backend/main.py).

Conversation flow inside OpenWebUI
───────────────────────────────────
  [User: Migraine]
  Assistant: [1/5] Where exactly is the pain located…?
  [User: Right side of my head]
  Assistant: [2/5] Is the pain throbbing or pulsating?
  …
  Assistant: ## Medical History Summary …   (done=true)

  The next message in the same chat starts a fresh session automatically.

Session mapping
───────────────
  OpenWebUI's chat_id  →  SessionInfo(session_id, last_user_msg,
                                       last_response, done)

  Avoiding spurious advancement of the backend session
  ─────────────────────────────────────────────────────
  OpenWebUI fires extra pipe() invocations per chat turn for internal
  tasks (title_generation, tags_generation, query_generation,
  follow_up_generation, function_calling, …) and for retries / streaming
  warm-up.  Each of those would otherwise advance the LangGraph session
  by one question, causing the agent to skip questions and end the
  interview prematurely.

  Two guards:

  1. Internal-task filter — if OpenWebUI tells us this call is a
     non-completion task we return immediately without touching the
     backend session.  OpenWebUI surfaces the task via __task__ kwarg
     or via metadata["task"].
  2. Last-message dedup — if the most recent user message we already
     processed for this chat shows up again (retry / streaming
     warm-up) we return the cached response.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, Optional

import httpx
from pydantic import BaseModel, Field


@dataclass
class _SessionInfo:
    session_id: str
    last_user_msg: str = ""
    last_response: str = ""
    done: bool = False


class Pipe:
    class Valves(BaseModel):
        FASTAPI_BASE_URL: str = Field(
            default="http://127.0.0.1:8000",
            description="Base URL of the Medical History FastAPI backend.",
        )
        CONNECT_TIMEOUT: float = Field(
            default=5.0, description="HTTP connect timeout (s)."
        )
        READ_TIMEOUT: Optional[float] = Field(
            default=120.0,
            description=(
                "HTTP read timeout (s). "
                "Set to None for no limit (needed if Ollama is slow)."
            ),
        )
        WRITE_TIMEOUT: float = Field(
            default=10.0, description="HTTP write timeout (s)."
        )
        POOL_TIMEOUT: float = Field(
            default=5.0, description="HTTP connection pool timeout (s)."
        )

    def __init__(self):
        self.valves = self.Valves()
        # Maps OpenWebUI chat_id → _SessionInfo
        self._sessions: dict[str, _SessionInfo] = {}

    # ── OpenWebUI registration ─────────────────────────────────────────────────

    def pipes(self):
        return [
            {
                "id": "medical_intake.pipe",
                "name": "Medical History Intake",
            }
        ]

    # ── Internal helpers ───────────────────────────────────────────────────────

    def _base(self) -> str:
        return (self.valves.FASTAPI_BASE_URL or "").rstrip("/")

    def _timeout(self) -> httpx.Timeout:
        return httpx.Timeout(
            connect=self.valves.CONNECT_TIMEOUT,
            read=self.valves.READ_TIMEOUT,
            write=self.valves.WRITE_TIMEOUT,
            pool=self.valves.POOL_TIMEOUT,
        )

    def _create_session(self, disease: str) -> tuple[str, str, bool]:
        """
        POST /sessions with the disease name.

        Returns (session_id, first_message, done).
        done=True means the disease was not recognised.
        """
        url = f"{self._base()}/sessions"
        with httpx.Client(timeout=self._timeout()) as client:
            resp = client.post(url, json={"disease": disease})
            resp.raise_for_status()
            data = resp.json()
            return data["session_id"], data["message"], data["done"]

    def _send_message(self, session_id: str, message: str) -> tuple[str, bool]:
        """POST /sessions/{id}/messages → (response_text, done)."""
        url = f"{self._base()}/sessions/{session_id}/messages"
        with httpx.Client(timeout=self._timeout()) as client:
            resp = client.post(url, json={"message": message})
            resp.raise_for_status()
            data = resp.json()
            return data["message"], data["done"]

    def _extract_last_user_message(self, body: Dict[str, Any]) -> str:
        """Pull the most recent user turn out of the OpenWebUI message list."""
        for m in reversed(body.get("messages") or []):
            if m.get("role") == "user":
                content = m.get("content")
                if isinstance(content, str):
                    return content.strip()
                if isinstance(content, list):
                    parts = [
                        p["text"]
                        for p in content
                        if isinstance(p, dict)
                        and p.get("type") == "text"
                        and isinstance(p.get("text"), str)
                    ]
                    if parts:
                        return "\n".join(parts).strip()
        if isinstance(body.get("prompt"), str):
            return body["prompt"].strip()
        return ""

    # ── Main entrypoint ────────────────────────────────────────────────────────

    # Tasks that are NOT user-driven chat completions and must not advance
    # the medical-interview session.
    _INTERNAL_TASKS = frozenset({
        "title_generation",
        "tags_generation",
        "query_generation",
        "follow_up_generation",
        "image_prompt_generation",
        "autocomplete_generation",
        "emoji_generation",
        "moa_response_generation",
        "function_calling",
    })

    def pipe(
        self,
        body: Dict[str, Any],
        __metadata__: Optional[dict] = None,
        __task__: Optional[str] = None,
        **_kwargs: Any,
    ) -> str:
        metadata = __metadata__ or {}
        chat_id: str = metadata.get("chat_id", "default")

        # ── Guard 1: skip OpenWebUI internal tasks ─────────────────────────────
        # title_generation / tags_generation / etc. fire extra pipe() calls per
        # chat turn — answering them would advance the medical interview by one
        # question per call, causing the user-visible "Q1 jumps to Q5" bug.
        task = (
            __task__
            or metadata.get("task")
            or (body.get("metadata") or {}).get("task")
        )
        if task and task in self._INTERNAL_TASKS:
            return ""

        try:
            user_msg = self._extract_last_user_message(body)
            if not user_msg:
                return (
                    "Please enter the patient's disease or condition to begin the intake.\n\n"
                    "Example: *Migraine*, *Hypertension*, *Asthma* …"
                )

            info = self._sessions.get(chat_id)

            # ── Guard 2: dedup retries / streaming warm-up ─────────────────────
            if info is not None and info.last_user_msg == user_msg:
                return info.last_response

            # ── No active session, OR previous session is finished ─────────────
            if info is None or info.done:
                session_id, message, done = self._create_session(user_msg)
                self._sessions[chat_id] = _SessionInfo(
                    session_id=session_id,
                    last_user_msg=user_msg,
                    last_response=message,
                    done=done,
                )
                return message

            # ── Active session: forward the answer to the current question ─────
            message, done = self._send_message(info.session_id, user_msg)
            self._sessions[chat_id] = _SessionInfo(
                session_id=info.session_id,
                last_user_msg=user_msg,
                last_response=message,
                done=done,
            )
            return message

        except httpx.ConnectError:
            return (
                f"⚠️  Cannot reach the medical agent at "
                f"`{self.valves.FASTAPI_BASE_URL}`. "
                f"Make sure the FastAPI server is running."
            )
        except httpx.HTTPStatusError as exc:
            # If the backend session is gone (server restart), clean up and
            # ask the user to re-enter the disease name.
            if exc.response.status_code in (404, 409):
                self._sessions.pop(chat_id, None)
                return (
                    "⚠️  Session expired (the server may have restarted). "
                    "Please enter the disease name again to start a new intake."
                )
            return (
                f"⚠️  Backend returned HTTP {exc.response.status_code}: "
                f"{exc.response.text}"
            )
        except Exception as exc:  # noqa: BLE001
            return f"⚠️  Unexpected error: {exc}"
