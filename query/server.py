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
from query import chart_extractor
from query.orchestrator import run as orchestrate
import query.memory as memory

STATIC_DIR = os.path.join(os.path.dirname(__file__), "static")

app = FastAPI()
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

# In-memory chart store — keyed by chart_id UUID
# { chart_id: { "ready": bool, "chart": spec | None } }
chart_store: dict[str, dict] = {}


# ── Request / Response models ─────────────────────────────────────────────────

class AskRequest(BaseModel):
    question: str
    session_id: str | None = None


class AskResponse(BaseModel):
    answer: str
    chart_id: str | None = None


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

        answer = await run_in_threadpool(
            orchestrate,
            request.question,
            context=context,
            verbose=False,
            session_id=request.session_id,
        )
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})

    # Generate chart_id upfront — extraction runs async after response is sent
    chart_id = str(uuid.uuid4())
    chart_store[chart_id] = {"ready": False, "chart": None}

    # Save turns after response (non-blocking)
    if request.session_id:
        background_tasks.add_task(
            memory.save_turn, request.session_id, "user", request.question
        )
        background_tasks.add_task(
            memory.save_turn, request.session_id, "assistant", answer
        )

    # Chart extraction — async, doesn't block response
    background_tasks.add_task(
        _extract_and_store_chart, chart_id, request.question, answer
    )

    return AskResponse(answer=answer, chart_id=chart_id)


# ── Chart background task + poll endpoint ─────────────────────────────────────

async def _extract_and_store_chart(chart_id: str, question: str, answer: str):
    """Run chart extraction in a thread so it doesn't block the event loop."""
    try:
        spec = await run_in_threadpool(
            chart_extractor.extract_chart, question, answer
        )
        chart_store[chart_id] = {"ready": True, "chart": spec}
    except Exception as e:
        print(f"[Server] Chart extraction failed for {chart_id}: {e}")
        chart_store[chart_id] = {"ready": True, "chart": None}


@app.get("/chart/{chart_id}")
async def get_chart(chart_id: str):
    """
    Frontend polls this after receiving an answer.
    Returns {ready: bool, chart: spec | null}.
    chart=null means extraction complete but no chart warranted.
    """
    result = chart_store.get(chart_id)
    if result is None:
        return JSONResponse(content={"ready": False, "chart": None})
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