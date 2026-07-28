"""
FastAPI HTTP layer for query_lg (LangGraph implementation).

Runs side-by-side with query/server.py (V1) on a separate port, per the
original coexistence decision. Session IDs are namespaced with an
"lg-" prefix (see langgraph_migration session notes) so the two systems
stay cleanly attributable if you ever compare them side by side.

IMPORTANT SCOPE NOTE: memory reuses query.memory's actual functions
directly (same DynamoDB table, same rolling-summary compression) —
nothing here reimplements memory logic. See _format_memory_context()
below for the only new piece: shaping load_context()'s dict into the
single memory_context string GraphState carries. planner_node is the
only graph node that currently reads memory_context — agent_node and
synthesis do NOT get memory injected yet (V1 also feeds context_note
into sub-agents and synthesis; that parity is still open here).

Clarify / interrupt-resume: the graph can pause mid-run when the
planner emits a "clarify" sentinel (graph.py's sentinel_response node
calls interrupt()). /ask returns needs_clarification=True plus a
clarify_question in that case; the frontend then calls /resume with the
SAME session_id and the user's answer to continue that exact paused
run — NOT a new /ask call, which would start a fresh run instead of
resuming the interrupted one.

Checkpointer: persistent AsyncSqliteSaver, one file on disk
(checkpoints.sqlite next to this file, or QUERY_LG_DB_PATH if set).
Opened once at startup via FastAPI's lifespan and kept open for the
process lifetime — NOT graph.py's default in-memory MemorySaver, which
would lose every paused clarify() on restart and isn't safe to share
across workers. Run this with a single uvicorn worker (--workers 1);
SQLite + a single long-lived connection doesn't support multiple
worker processes safely.
"""


import os
import sys
import uuid
from contextlib import asynccontextmanager

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import aiosqlite
from fastapi import FastAPI, BackgroundTasks
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from langgraph.types import Command

from query import memory
from query_lg.graph import compile_graph, CHECKPOINT_SERDE
from query_lg.state import build_initial_state

STATIC_DIR = os.path.join(os.path.dirname(__file__), "static")
# Override with QUERY_LG_DB_PATH in prod to point at a mounted volume —
# otherwise the checkpoint DB lives inside the container's writable layer
# and every redeploy silently wipes all paused/resumable threads.
DB_PATH = os.environ.get(
    "QUERY_LG_DB_PATH",
    os.path.join(os.path.dirname(__file__), "checkpoints.sqlite"),
)


@asynccontextmanager
async def lifespan(app: FastAPI):
    conn = await aiosqlite.connect(DB_PATH)
    saver = AsyncSqliteSaver(conn, serde=CHECKPOINT_SERDE)
    await saver.setup()
    app.state.graph = compile_graph(checkpointer=saver)
    try:
        yield
    finally:
        await conn.close()


app = FastAPI(lifespan=lifespan)
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


# ── Request / Response models ─────────────────────────────────────────────────

class AskRequest(BaseModel):
    question: str
    session_id: str


class ResumeRequest(BaseModel):
    session_id: str
    run_id: str
    answer: str


class AskResponse(BaseModel):
    answer: str | None = None
    needs_clarification: bool = False
    clarify_question: str | None = None
    session_id: str
    run_id: str


# ── Static routes ─────────────────────────────────────────────────────────────

@app.get("/health")
async def health():
    return {"status": "ok"}


@app.get("/")
async def index():
    return FileResponse(os.path.join(STATIC_DIR, "index.html"))


@app.get("/chat/{session_id}")
async def chat_session(session_id: str):
    """Serve chat UI for a specific session — JS reads session_id from URL,
    same convention as V1's /chat/{session_id}."""
    return FileResponse(os.path.join(STATIC_DIR, "index.html"))


# ── Memory formatting ──────────────────────────────────────────────────────────
# NOTE ON SCOPE: this reuses query.memory's actual read/write functions
# directly (same DynamoDB table, same compression/summary logic) — it does
# NOT reimplement any memory logic. The only new code here is shaping
# memory.load_context()'s return dict into the single memory_context
# string GraphState carries, mirroring V1's query/planner.py plan()'s
# session_memory + recent-turns framing so the two systems stay
# comparable. session_ids are "lg-" prefixed (see langgraph_migration
# session notes) so this never collides with V1's own session rows in
# the same table.

def _truncate_turn(text: str, limit: int = 600) -> str:
    """Truncate at a line boundary, not mid-character, so a partial
    markdown table row doesn't get handed to the model as if it were
    complete. 600 chars (up from 200) is enough to reliably carry a
    comparison table's data rows, not just its title."""
    if len(text) <= limit:
        return text
    truncated = text[:limit]
    last_newline = truncated.rfind("\n")
    if last_newline > 0:
        truncated = truncated[:last_newline]
    return truncated + "\n...[truncated]"


def _format_memory_context(context: dict) -> str:
    parts = []
    if context.get("summary"):
        parts.append(f"<session_memory>\n{context['summary']}\n</session_memory>")
    recent = context.get("recent_turns") or []
    if recent:
        recent_text = "\n".join(
            f"{t['role'].upper()}: {_truncate_turn(str(t['content']))}" for t in recent[-4:]
        )
        parts.append(f"Conversation context:\n{recent_text}")
    return "\n\n".join(parts)


# ── Shared result interpretation ──────────────────────────────────────────────
def _interpret_result(result: dict, session_id: str, run_id: str) -> AskResponse:
    """Both /ask and /resume land here — the graph can pause on EITHER
    call (a resumed run can itself hit a second clarify), so this is not
    special-cased to only run after /ask."""
    interrupts = result.get("__interrupt__")
    if interrupts:
        payload = interrupts[0].value or {}
        return AskResponse(
            needs_clarification=True,
            clarify_question=payload.get("question", "Can you clarify?"),
            session_id=session_id,
            run_id=run_id,
        )
    return AskResponse(answer=result.get("final_answer"), session_id=session_id, run_id=run_id)


# ── Main endpoints ─────────────────────────────────────────────────────────────

@app.post("/ask", response_model=AskResponse)
async def ask(request: AskRequest, background_tasks: BackgroundTasks):
    context = await run_in_threadpool(memory.load_context, request.session_id)
    memory_context = _format_memory_context(context)
    # A FRESH thread_id per top-level question, NOT request.session_id.
    # GraphState's node_results/drafts/retry_counts use a dict-union
    # reducer (needed for parallel Send fan-in WITHIN one run) — the
    # checkpointer merges rather than clears those fields across
    # ainvoke() calls on the same thread_id. Reusing session_id as the
    # thread_id for an entire conversation meant turn 2 inherited turn
    # 1's node_results under the same node ids, dispatch saw them as
    # already-done, skipped re-running the agent, and synthesis handed
    # back turn 1's stale answer verbatim. run_id scopes the graph
    # thread to a SINGLE question (reused only across that question's
    # own clarify pause/resume); session_id stays the separate key for
    # query.memory's cross-turn conversation history.
    run_id = f"{request.session_id}:{uuid.uuid4().hex[:8]}"
    config = {
    "configurable": {"thread_id": run_id},
    "metadata": {"session_id": request.session_id, "run_id": run_id},
    "tags": ["lg"],
    }
    initial_state = build_initial_state(
        request.question, request.session_id, memory_context=memory_context
    )
    try:
        result = await app.state.graph.ainvoke(initial_state, config=config)
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})

    response = _interpret_result(result, request.session_id, run_id)

    background_tasks.add_task(memory.save_turn, request.session_id, "user", request.question)
    background_tasks.add_task(
        memory.save_turn, request.session_id, "assistant",
        response.clarify_question if response.needs_clarification else (response.answer or ""),
    )
    return response


@app.post("/resume", response_model=AskResponse)
async def resume(request: ResumeRequest, background_tasks: BackgroundTasks):
    config = {
    "configurable": {"thread_id": request.run_id},
    "metadata": {"session_id": request.session_id, "run_id": request.run_id},
    "tags": ["lg"],
    }
    try:
        result = await app.state.graph.ainvoke(
            Command(resume=request.answer), config=config
        )
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})

    response = _interpret_result(result, request.session_id, request.run_id)

    background_tasks.add_task(memory.save_turn, request.session_id, "user", request.answer)
    background_tasks.add_task(
        memory.save_turn, request.session_id, "assistant",
        response.clarify_question if response.needs_clarification else (response.answer or ""),
    )
    return response