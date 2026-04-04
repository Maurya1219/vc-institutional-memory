import subprocess
from contextlib import asynccontextmanager
from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

load_dotenv()

_ROOT = Path(__file__).resolve().parent
_STATIC = _ROOT / "static"


@asynccontextmanager
async def lifespan(app: FastAPI):
    vs = _ROOT / "dvc_vectorstore" / "index.faiss"
    if not vs.is_file():
        print("Vector store not found — running ingestion...")
        subprocess.run(
            ["python3", "ingest_affinity.py"],
            cwd=str(_ROOT),
            check=True,
        )
        print("Ingestion complete.")
    yield


from memory_engine import ask

app = FastAPI(lifespan=lifespan)
app.mount("/static", StaticFiles(directory=str(_STATIC)), name="static")


class Query(BaseModel):
    question: str


@app.get("/", response_class=HTMLResponse)
async def root():
    index_path = _STATIC / "index.html"
    if not index_path.is_file():
        raise HTTPException(status_code=500, detail="static/index.html missing")
    return HTMLResponse(index_path.read_text(encoding="utf-8"))


@app.post("/query")
async def query(body: Query):
    if not body.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty")
    try:
        result = ask(body.question, verbose=False)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e


@app.get("/health")
async def health():
    return {"status": "ok"}
