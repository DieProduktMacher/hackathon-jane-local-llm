"""
LangGraph-based medical history intake agent.

Workflow
────────
1. session_start(session_id, disease_input)
   • Initialises a LangGraph thread with the disease name already in state.
   • The first question shown to the doctor is ALWAYS the mandatory safety
     gate ("Does the patient have suicidal thoughts?"), independent of
     whether the disease is known.
2. session_reply(session_id, user_input)
   • Safety gate: an affirmative answer short-circuits the interview with a
     legal-refusal message and no further questions.
   • Otherwise, the disease is looked up in medical-questions.csv:
       – Known   → loads the structured question list for that condition.
       – Unknown → falls back to a single open-ended symptom question
                   ("(unknown disease) Which symptoms is the patient reporting?").
   • Returns the next question, or the final LLM summary when done.

Graph architecture
──────────────────
Each node has at most ONE interrupt() call, avoiding the multi-interrupt-in-a-loop
pitfall where LangGraph can skip questions when replaying the node from scratch.

  safety_gate ──► (yes) ──► safety_block ──► END
              └─► (no)  ──► load_disease_questions ──► ask_question ──► (loop) ──► summarize ──► END
"""

import csv
import os
import re
from pathlib import Path
from typing import Literal, Optional, TypedDict

from dotenv import load_dotenv
from langchain_ollama import ChatOllama
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, StateGraph
from langgraph.types import Command, interrupt

# Load .env from the backend directory so OLLAMA_MODEL etc. are picked up
load_dotenv(Path(__file__).parent / ".env")

# ── Configuration ──────────────────────────────────────────────────────────────

DATA_PATH = Path(__file__).parent / "medical-questions.csv"
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.2")


# ── Disease data ───────────────────────────────────────────────────────────────

def _parse_questions(raw: str) -> list[str]:
    """Split on the pipe character used as question separator in the CSV."""
    return [q.strip() for q in raw.strip().strip('"').split("|") if q.strip()]


def _load_disease_data() -> dict[str, dict]:
    data: dict[str, dict] = {}
    with open(DATA_PATH, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            name = row["Disease_Name"].strip()
            data[name.lower()] = {
                "name": name,
                "questions": _parse_questions(row["Key_Questions"]),
                "symptoms": row["Related_Symptoms"].strip(),
                "tests": row["Common_Tests"].strip(),
                "treatments": row["Typical_Treatments"].strip(),
            }
    return data


DISEASE_DATA: dict[str, dict] = _load_disease_data()

_SUPPORTED_DISEASES = ", ".join(v["name"] for v in DISEASE_DATA.values())


def find_disease(user_input: str) -> Optional[dict]:
    """Case-insensitive lookup with partial-match fallback."""
    normalized = user_input.strip().lower()
    if normalized in DISEASE_DATA:
        return DISEASE_DATA[normalized]
    for key, value in DISEASE_DATA.items():
        if key in normalized or normalized in key:
            return value
    return None


# ── LangGraph state ────────────────────────────────────────────────────────────

class State(TypedDict):
    disease_name: str
    disease_info: dict
    questions: list
    current_index: int
    answers: list       # [{"question": str, "answer": str}]
    summary: str
    safety_answer: str


def _fresh_state(disease_name: str = "") -> State:
    """Return a new state dict — never reuse a shared mutable instance."""
    return {
        "disease_name": disease_name,
        "disease_info": {},
        "questions": [],
        "current_index": 0,
        "answers": [],
        "summary": "",
        "safety_answer": "",
    }


# ── Graph nodes ────────────────────────────────────────────────────────────────

UNKNOWN_DISEASE_QUESTION = "(unknown disease) Which symptoms is the patient reporting?"

SAFETY_QUESTION = "Does the patient have suicidal thoughts?"

SAFETY_REFUSAL = (
    "⚠️  This intake assistant is not legally permitted to provide clinical "
    "guidance for a patient who has reported suicidal thoughts.\n\n"
    "Please ensure the patient is connected with appropriate emergency or "
    "psychiatric care immediately. In a crisis, contact local emergency "
    "services or a regional suicide-prevention hotline."
)

# Word-boundary match catches "yes", "Yes.", "yes, definitely", etc., without
# false-firing on "yesterday". A leading "no" / "not" overrides — handled below.
_YES_RE = re.compile(
    r"\b(?:yes|yeah|yep|yup|y|true|affirmative|positive|confirmed)\b",
    re.IGNORECASE,
)
_NO_PREFIX_RE = re.compile(r"^\s*(?:no\b|n\b|nope\b|nah\b|not\b|negative\b)", re.IGNORECASE)


def _is_affirmative(text: str) -> bool:
    """True if *text* reads as a yes-answer to a yes/no question."""
    if not text:
        return False
    if _NO_PREFIX_RE.search(text):
        return False
    return bool(_YES_RE.search(text))


def safety_gate_node(state: State) -> State:
    """
    Mandatory safety check shown as the FIRST question of every interview.
    Records the answer in state and lets ``_route_after_safety`` decide
    whether to block or proceed.
    """
    answer: str = interrupt(SAFETY_QUESTION)
    return {
        **state,
        "safety_answer": answer,
        "answers": state["answers"] + [{"question": SAFETY_QUESTION, "answer": answer}],
    }


def safety_block_node(state: State) -> State:
    """Short-circuit terminal node: write the legal refusal as the summary."""
    return {**state, "summary": SAFETY_REFUSAL}


def load_disease_questions_node(state: State) -> State:
    """
    Look up disease info and initialise the questions list.
    Runs once at the start of each session — no interrupt here.

    If the disease isn't in the CSV, we still continue the interview with a
    single open-ended fallback question so the doctor can capture symptoms
    and the LLM can produce a useful summary.
    """
    disease_info = find_disease(state["disease_name"])
    if not disease_info:
        return {
            **state,
            "disease_info": {"name": state["disease_name"].strip(), "unknown": True},
            "questions": [UNKNOWN_DISEASE_QUESTION],
            "current_index": 0,
            "answers": [],
        }
    return {
        **state,
        "disease_name": disease_info["name"],
        "disease_info": disease_info,
        "questions": list(disease_info["questions"]),
        "current_index": 0,
        "answers": [],
    }


def ask_question_node(state: State) -> State:
    """
    Ask a single clinical question via interrupt() and record the answer.

    Each invocation of this node has exactly ONE interrupt() call, so
    LangGraph never needs to replay multiple interrupts in a single pass.
    The node is re-entered for each question through the conditional loop edge.
    """
    questions: list[str] = state["questions"]
    idx: int = state["current_index"]
    total: int = len(questions)

    answer: str = interrupt(f"[{idx + 1}/{total}] {questions[idx]}")

    return {
        **state,
        "answers": state["answers"] + [{"question": questions[idx], "answer": answer}],
        "current_index": idx + 1,
    }


def summarize_node(state: State) -> State:
    """Generate the clinical summary via Ollama once all questions are answered."""
    disease_info = state["disease_info"]
    answers = state["answers"]
    is_unknown = disease_info.get("unknown", False)

    try:
        llm = ChatOllama(base_url=OLLAMA_BASE_URL, model=OLLAMA_MODEL, temperature=0)

        qa_block = "\n".join(
            f"  Q: {a['question']}\n  A: {a['answer']}" for a in answers
        )

        if is_unknown:
            patient_input = state["disease_name"].strip() or "(none provided)"
            prompt = (
                "You are a clinical documentation assistant helping a doctor record a "
                "patient intake interview.\n\n"
                f"The patient initially reported \"{patient_input}\", which is not a "
                "recognised condition in our structured-question database. The doctor "
                "asked one open-ended symptom question instead.\n\n"
                f"Patient response:\n{qa_block}\n\n"
                "Write a concise medical history summary the doctor can add to the "
                "patient record. Based on the symptoms reported, suggest plausible "
                "differential diagnoses, list relevant follow-up questions, and "
                "recommend next diagnostic steps. Make clear that no specific "
                "condition has been confirmed."
            )
        else:
            prompt = (
                "You are a clinical documentation assistant helping a doctor record a "
                "patient intake interview.\n\n"
                f"Diagnosed condition: {disease_info['name']}\n\n"
                f"Patient responses:\n{qa_block}\n\n"
                f"Related symptoms to watch: {disease_info['symptoms']}\n"
                f"Common diagnostic tests: {disease_info['tests']}\n"
                f"Typical treatments: {disease_info['treatments']}\n\n"
                "Write a concise medical history summary the doctor can add to the "
                "patient record. Highlight key positive findings, flag any concerning "
                "answers, and recommend next steps."
            )

        response = llm.invoke(prompt)
        summary = response.content
    except Exception as exc:  # noqa: BLE001
        summary = (
            f"⚠️  Could not generate AI summary: {exc}\n\n"
            "**Interview answers recorded:**\n"
            + "\n".join(
                f"- **{a['question']}**  \n  {a['answer']}" for a in answers
            )
        )

    return {**state, "summary": summary}


def _route_after_question(state: State) -> Literal["ask_question", "summarize"]:
    """Loop back for more questions, or proceed to summary when done."""
    if state["current_index"] < len(state["questions"]):
        return "ask_question"
    return "summarize"


def _route_after_safety(state: State) -> Literal["safety_block", "load_disease_questions"]:
    """Block the interview if the doctor reported suicidal thoughts."""
    if _is_affirmative(state.get("safety_answer", "")):
        return "safety_block"
    return "load_disease_questions"


# ── Build & compile graph ──────────────────────────────────────────────────────

_builder = StateGraph(State)
_builder.add_node("safety_gate", safety_gate_node)
_builder.add_node("safety_block", safety_block_node)
_builder.add_node("load_disease_questions", load_disease_questions_node)
_builder.add_node("ask_question", ask_question_node)
_builder.add_node("summarize", summarize_node)

_builder.set_entry_point("safety_gate")
_builder.add_conditional_edges("safety_gate", _route_after_safety)
_builder.add_edge("safety_block", END)
_builder.add_edge("load_disease_questions", "ask_question")
_builder.add_conditional_edges("ask_question", _route_after_question)
_builder.add_edge("summarize", END)

_memory = MemorySaver()
graph = _builder.compile(checkpointer=_memory)


# ── Visualisation ─────────────────────────────────────────────────────────────

def visualize_graph() -> str:
    """Return a Mermaid diagram string for the compiled graph."""
    return graph.get_graph().draw_mermaid()


# ── Internal helpers ───────────────────────────────────────────────────────────

def _config(session_id: str) -> dict:
    return {"configurable": {"thread_id": session_id}}


def get_pending_interrupt(session_id: str) -> Optional[str]:
    """Return the pending interrupt message, or None if the session is done."""
    snapshot = graph.get_state(_config(session_id))
    if snapshot.tasks and snapshot.tasks[0].interrupts:
        return str(snapshot.tasks[0].interrupts[0].value)
    return None


# ── Public session helpers ─────────────────────────────────────────────────────

def session_start(session_id: str, disease_input: str) -> tuple[str, bool]:
    """
    Initialise a new LangGraph thread with the disease name already in state,
    so only a single graph.invoke() is needed.

    Returns
    -------
    (message, done)
        message – first clinical question (structured for known diseases, or
                  the open-ended fallback question for unknown ones)
        done    – always False; the interview always has at least one question
    """
    graph.invoke(_fresh_state(disease_input), _config(session_id))

    pending = get_pending_interrupt(session_id)
    if pending is not None:
        return pending, False

    # Should not reach here — graph always interrupts on the first question.
    snapshot = graph.get_state(_config(session_id))
    return snapshot.values.get("summary", "Interview complete."), True


def session_reply(session_id: str, user_input: str) -> tuple[str, bool]:
    """
    Resume the graph with *user_input* (answer to the current question).

    Returns
    -------
    (message, done)
        message – next question or final summary
        done    – True when the interview is finished
    """
    graph.invoke(Command(resume=user_input), _config(session_id))

    pending = get_pending_interrupt(session_id)
    if pending is not None:
        return pending, False

    snapshot = graph.get_state(_config(session_id))
    summary = snapshot.values.get("summary", "Interview complete.")
    return summary, True


if __name__ == "__main__":
    print(visualize_graph())
