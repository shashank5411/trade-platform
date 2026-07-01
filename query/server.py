"""
FastAPI HTTP layer for the financial research multi-agent platform.

Memory is wired: session_id in the request loads prior context and saves
turns after the response. Compression fires inside save_turn (via
BackgroundTasks) and doesn't block the HTTP response.
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from fastapi import FastAPI, BackgroundTasks
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from query.orchestrator import run as orchestrate
import query.memory as memory

STATIC_DIR = os.path.join(os.path.dirname(__file__), "static")

app = FastAPI()
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


class AskRequest(BaseModel):
    question: str
    session_id: str | None = None


class AskResponse(BaseModel):
    answer: str


@app.get("/health")
async def health():
    return {"status": "ok"}


@app.get("/")
async def index():
    return FileResponse(os.path.join(STATIC_DIR, "index.html"))


@app.post("/ask", response_model=AskResponse)
async def ask(request: AskRequest, background_tasks: BackgroundTasks):
    # orchestrate() is sync and calls asyncio.run() internally — offload to thread.
    try:
        context = None
        if request.session_id:
            context = await run_in_threadpool(memory.load_context, request.session_id)

        answer = await run_in_threadpool(
            orchestrate,
            request.question,
            context=context,
            verbose=False,
        )
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})

    # Save turns after the response is sent — compression fires here if needed.
    if request.session_id:
        background_tasks.add_task(
            memory.save_turn, request.session_id, "user", request.question
        )
        background_tasks.add_task(
            memory.save_turn, request.session_id, "assistant", answer
        )

    return AskResponse(answer=answer)
