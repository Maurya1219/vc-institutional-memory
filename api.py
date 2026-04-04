import os
from contextlib import asynccontextmanager

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

load_dotenv()


@asynccontextmanager
async def lifespan(app: FastAPI):
    if not os.path.exists("dvc_vectorstore/index.faiss"):
        print("Vector store not found — running ingestion...")
        from ingest_affinity import main

        main()
        print("Ingestion complete.")
    else:
        print("Vector store found — skipping ingestion.")
    yield


from memory_engine import ask

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
