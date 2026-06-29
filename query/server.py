"""
FastAPI HTTP layer — thin end-to-end slice.

Serves the same planner -> executor orchestration query/agent.py's CLI
uses (query/orchestrator.py's run()), over HTTP. Single-turn only:
session_id is accepted on the request schema but ignored here, so the
request shape doesn't need to change again once memory.py is wired in.
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from fastapi import FastAPI
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from query.orchestrator import run as orchestrate

STATIC_DIR = os.path.join(os.path.dirname(__file__), "static")

app = FastAPI()
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


class AskRequest(BaseModel):
    question: str
    session_id: str | None = None  # accepted but unused in this task —
    # placeholder so the frontend's request shape doesn't change again
    # once memory.py's save_turn/load_turns are wired in here.


class AskResponse(BaseModel):
    answer: str


@app.get("/health")
async def health():
    return {"status": "ok"}


@app.get("/")
async def index():
    return FileResponse(os.path.join(STATIC_DIR, "index.html"))


@app.post("/ask", response_model=AskResponse)
async def ask(request: AskRequest):
    # orchestrate() is sync and calls asyncio.run() internally (see
    # orchestrator.py) -- it would raise if called directly from this
    # already-running event loop, so it's offloaded to a thread.
    try:
        answer = await run_in_threadpool(
            orchestrate,
            request.question,
            history=[],
            verbose=False,
            session_id=None,
        )
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})

    return AskResponse(answer=answer)
