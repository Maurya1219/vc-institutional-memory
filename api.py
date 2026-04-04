from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from memory_engine import ask

load_dotenv()

_ROOT = Path(__file__).resolve().parent
_STATIC = _ROOT / "static"

app = FastAPI()
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
