"""
FastAPI HTTP layer for the financial research multi-agent platform.

Memory is wired: session_id in the request loads prior context and saves
turns after the response. Compression fires inside save_turn (via
BackgroundTasks) and doesn't block the HTTP response.
"""

import os
import sys
import uuid

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from fastapi import FastAPI, BackgroundTasks
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from query import admin as admin_module
from query import chart_agent
from query import clarification
from query.orchestrator import run as orchestrate
import query.memory as memory

STATIC_DIR = os.path.join(os.path.dirname(__file__), "static")

app = FastAPI()
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

# In-memory chart store — keyed by chart_id UUID
# { chart_id: { "ready": bool, "charts": list[spec] } }
# "charts" is always a list (empty list = nothing chartable, same role as the
# old chart=None convention but avoids the None special-case on the frontend).
chart_store: dict[str, dict] = {}


# ── Request / Response models ─────────────────────────────────────────────────

class AskRequest(BaseModel):
    question: str
    session_id: str | None = None


class AskResponse(BaseModel):
    answer: str
    chart_id: str | None = None


class EvalRunRequest(BaseModel):
    ids: list[str] | None = None
    tier: str = "all"
    include_retired: bool = False


# ── Static routes ─────────────────────────────────────────────────────────────

@app.get("/health")
async def health():
    return {"status": "ok"}


@app.get("/")
async def index():
    return FileResponse(os.path.join(STATIC_DIR, "index.html"))


@app.get("/admin")
async def admin_page():
    return FileResponse(os.path.join(STATIC_DIR, "admin.html"))


@app.get("/chat/{session_id}")
async def chat_session(session_id: str):
    """Serve chat UI for a specific session — JS reads session_id from URL."""
    return FileResponse(os.path.join(STATIC_DIR, "index.html"))


# ── Main ask endpoint ─────────────────────────────────────────────────────────

@app.post("/ask", response_model=AskResponse)
async def ask(request: AskRequest, background_tasks: BackgroundTasks):
    try:
        context = None
        if request.session_id:
            context = await run_in_threadpool(memory.load_context, request.session_id)

        # Check-and-clear pending clarification BEFORE calling orchestrate.
        # Shared with run_eval.py's multi-turn runner — see
        # clarification.resolve_pending_clarification()'s docstring for why
        # this logic lives there instead of being duplicated here.
        effective_question = await run_in_threadpool(
            clarification.resolve_pending_clarification,
            request.session_id, request.question,
        )

        answer, node_tool_calls, meta = await run_in_threadpool(
            orchestrate,
            effective_question,
            context=context,
            verbose=False,
            session_id=request.session_id,
        )
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})

    # Save turns after response (non-blocking).
    # Always save request.question (what the user typed), not effective_question
    # (the merged version) — raw turn history reflects literal user input.
    if request.session_id:
        background_tasks.add_task(
            memory.save_turn, request.session_id, "user", request.question
        )
        background_tasks.add_task(
            memory.save_turn, request.session_id, "assistant", answer
        )
        if meta.get("awaiting_clarification"):
            background_tasks.add_task(
                memory.set_pending_clarification,
                request.session_id, effective_question, answer,
            )

    # Chart building — skip entirely when we're just asking a clarifying
    # question (no tool data was fetched, node_tool_calls will be empty).
    if meta.get("awaiting_clarification"):
        return AskResponse(answer=answer, chart_id=None)

    chart_id = str(uuid.uuid4())
    chart_store[chart_id] = {"ready": False, "charts": []}
    background_tasks.add_task(
        _extract_and_store_chart, chart_id, request.question, answer, node_tool_calls
    )

    return AskResponse(answer=answer, chart_id=chart_id)


# ── Chart background task + poll endpoint ─────────────────────────────────────

async def _extract_and_store_chart(
    chart_id: str, question: str, answer: str, node_tool_calls: dict
):
    """Run chart building in a thread so it doesn't block the event loop."""
    try:
        specs = await run_in_threadpool(
            chart_agent.build_charts, question, answer, node_tool_calls
        )
        chart_store[chart_id] = {"ready": True, "charts": specs}
    except Exception as e:
        print(f"[Server] Chart building failed for {chart_id}: {e}")
        chart_store[chart_id] = {"ready": True, "charts": []}


@app.get("/chart/{chart_id}")
async def get_chart(chart_id: str):
    """
    Frontend polls this after receiving an answer.
    Returns {ready: bool, chart: spec | null}.
    chart=null means extraction complete but no chart warranted.
    """
    result = chart_store.get(chart_id)
    if result is None:
        return JSONResponse(content={"ready": False, "charts": []})
    return JSONResponse(content=result)


# ── Admin API endpoints ───────────────────────────────────────────────────────

@app.get("/admin/api/pipeline")
async def admin_pipeline():
    data = await run_in_threadpool(admin_module.get_pipeline_status)
    return JSONResponse(content=data)


@app.get("/admin/api/sessions")
async def admin_sessions(limit: int = 20):
    data = await run_in_threadpool(admin_module.get_sessions, limit)
    return JSONResponse(content=data)


@app.get("/admin/api/evals")
async def admin_evals():
    data = await run_in_threadpool(admin_module.get_evals)
    return JSONResponse(content=data)


@app.get("/admin/api/telemetry")
async def admin_telemetry(limit: int = 50):
    data = await run_in_threadpool(admin_module.get_telemetry, limit)
    return JSONResponse(content=data)


@app.get("/admin/api/sessions/{session_id}")
async def admin_session_detail(session_id: str):
    data = await run_in_threadpool(admin_module.get_session_detail, session_id)
    return JSONResponse(content=data)


@app.get("/admin/api/telemetry/detail")
async def admin_trace_detail(key: str):
    data = await run_in_threadpool(admin_module.get_trace_detail, key)
    return JSONResponse(content=data)


@app.post("/admin/api/pipeline/{source}/fire")
async def admin_fire_pipeline(source: str):
    data = await run_in_threadpool(admin_module.fire_pipeline, source)
    return JSONResponse(content=data)


# ── Eval runner endpoints ─────────────────────────────────────────────────────
# Specific paths (/questions, /status, /run) must come before the /{run_id}/records
# path-param route so FastAPI matches them as literals, not as run_id values.

@app.get("/admin/api/evals/questions")
async def admin_eval_questions():
    data = await run_in_threadpool(admin_module.list_eval_questions)
    return JSONResponse(content=data)


@app.post("/admin/api/evals/run")
async def admin_eval_run(request: EvalRunRequest):
    data = await run_in_threadpool(
        admin_module.trigger_eval_run,
        request.ids, request.tier, request.include_retired,
    )
    return JSONResponse(content=data)


@app.get("/admin/api/evals/status")
async def admin_eval_status():
    data = await run_in_threadpool(admin_module.get_eval_run_state)
    return JSONResponse(content=data)


@app.get("/admin/api/evals/{run_id}/records")
async def admin_eval_records(run_id: str):
    data = await run_in_threadpool(admin_module.get_eval_run_records, run_id)
    return JSONResponse(content=data)