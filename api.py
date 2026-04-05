import asyncio
import json
from contextlib import asynccontextmanager
from datetime import datetime, timedelta, time as dtime
from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

load_dotenv()

from memory_engine import ask
from proactive_engine import generate_weekly_digest, get_latest_digest

CACHE_DIR = Path("affinity_cache")
SYNC_LOG = CACHE_DIR / "sync_log.json"


def _seconds_until_next_2am() -> float:
    now = datetime.now()
    target = datetime.combine(now.date(), dtime(2, 0))
    if now >= target:
        target += timedelta(days=1)
    return (target - now).total_seconds()


async def nightly_sync_scheduler() -> None:
    while True:
        wait = _seconds_until_next_2am()
        print(f"Next Affinity sync scheduled in {wait / 3600:.1f} hours")
        await asyncio.sleep(wait)
        try:
            from sync_affinity import run_sync

            await asyncio.to_thread(run_sync)
            print("Nightly Affinity sync completed")
        except Exception as e:
            print(f"Nightly Affinity sync failed: {e}")

        if datetime.now().weekday() == 0:
            try:
                await asyncio.to_thread(generate_weekly_digest)
                print("Weekly digest generated")
            except Exception as e:
                print(f"Digest generation failed: {e}")


@asynccontextmanager
async def lifespan(app: FastAPI):
    asyncio.create_task(nightly_sync_scheduler())
    yield


app = FastAPI(lifespan=lifespan)
app.mount("/static", StaticFiles(directory="static"), name="static")


class Query(BaseModel):
    question: str


@app.get("/", response_class=HTMLResponse)
async def root():
    with open("static/index.html") as f:
        return f.read()


@app.post("/query")
async def query(body: Query):
    if not body.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty")
    try:
        result = ask(body.question, verbose=False)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/health")
async def health():
    return {"status": "ok"}


@app.get("/sync-status")
async def sync_status():
    if not SYNC_LOG.exists():
        return {"last_sync": None, "history": []}
    with open(SYNC_LOG, encoding="utf-8") as f:
        log = json.load(f)
    return {
        "last_sync": log[-1]["timestamp"] if log else None,
        "last_result": log[-1]["results"] if log else None,
        "history": log[-5:],
    }


@app.post("/sync")
async def trigger_sync():
    from sync_affinity import run_sync

    results = await asyncio.to_thread(run_sync)
    return {"status": "complete", "results": results}


@app.get("/digest")
async def get_digest():
    digest = get_latest_digest()
    if not digest:
        return {"digest": "No digest generated yet.", "generated_at": None}
    return digest


@app.post("/digest/generate")
async def generate_digest():
    result = await asyncio.to_thread(generate_weekly_digest)
    return result
