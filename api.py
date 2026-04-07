import asyncio
import json
import os
import urllib.parse
from contextlib import asynccontextmanager
from datetime import datetime, timedelta, time as dtime
from pathlib import Path

import msal
from dotenv import load_dotenv
from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from starlette.middleware.sessions import SessionMiddleware

load_dotenv()

AZURE_CLIENT_ID = os.getenv("AZURE_CLIENT_ID")
AZURE_CLIENT_SECRET = os.getenv("AZURE_CLIENT_SECRET")
AZURE_TENANT_ID = os.getenv("AZURE_TENANT_ID")
SESSION_SECRET = os.getenv("SESSION_SECRET", "changeme-use-a-real-secret")
ALLOWED_DOMAIN = "dallasvc.com"
REDIRECT_URI = os.getenv(
    "REDIRECT_URI",
    "https://web-production-f614a.up.railway.app/auth/callback",
)
POST_LOGOUT_REDIRECT_URI = os.getenv(
    "POST_LOGOUT_REDIRECT_URI",
    "https://web-production-f614a.up.railway.app",
)

SCOPES = ["User.Read", "Mail.Read"]

CACHE_DIR = Path("affinity_cache")
SYNC_LOG = CACHE_DIR / "sync_log.json"


class Query(BaseModel):
    question: str


class MeetingPrepRequest(BaseModel):
    company: str


class ScoreDealRequest(BaseModel):
    company: str
    description: str
    sector: str = ""
    stage: str = ""
    ask: str = ""
    extra_context: str = ""


def get_msal_app():
    return msal.ConfidentialClientApplication(
        AZURE_CLIENT_ID,
        authority=f"https://login.microsoftonline.com/{AZURE_TENANT_ID}",
        client_credential=AZURE_CLIENT_SECRET,
    )


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
                from proactive_engine import generate_weekly_digest

                await asyncio.to_thread(generate_weekly_digest)
                print("Weekly digest generated")
            except Exception as e:
                print(f"Digest generation failed: {e}")


@asynccontextmanager
async def lifespan(app: FastAPI):
    asyncio.create_task(nightly_sync_scheduler())
    yield


app = FastAPI(lifespan=lifespan)

app.add_middleware(
    SessionMiddleware,
    secret_key=SESSION_SECRET,
    max_age=86400 * 7,
    same_site="lax",
    https_only=os.getenv("RAILWAY_ENVIRONMENT") is not None
    or os.getenv("FORCE_HTTPS_COOKIES", "").lower() in ("1", "true", "yes"),
)

app.mount("/static", StaticFiles(directory="static"), name="static")


def require_auth(request: Request):
    user = request.session.get("user")
    if not user:
        raise HTTPException(status_code=401, detail="Not authenticated")
    return user


def _meeting_prep_sync(company: str) -> dict:
    from graph_query import answer_graph_query
    from meeting_prep import run_meeting_prep
    from memory_engine import search_with_notes

    def rag_context(name: str) -> str:
        return search_with_notes(f"everything about {name} notes history status")

    def graph_context(name: str) -> str:
        return answer_graph_query(f"who do we know at {name}")

    return run_meeting_prep(company, rag_context, graph_context)


def _score_deal_sync(body: ScoreDealRequest) -> dict:
    from deal_scorer import format_score, score_deal

    sc = score_deal(
        company_name=body.company,
        description=body.description,
        sector=body.sector,
        stage=body.stage,
        ask=body.ask,
        extra_context=body.extra_context,
    )
    formatted = format_score(sc, body.company)
    return {
        "company": body.company,
        "score": sc.model_dump(),
        "formatted": formatted,
        "generated_at": datetime.now().isoformat(),
    }


@app.get("/auth/login")
async def login(request: Request):
    msal_app = get_msal_app()
    auth_url = msal_app.get_authorization_request_url(
        scopes=SCOPES,
        redirect_uri=REDIRECT_URI,
        state="login",
    )
    return RedirectResponse(auth_url)


@app.get("/auth/callback")
async def auth_callback(request: Request, code: str | None = None, error: str | None = None):
    if error:
        return HTMLResponse(
            f"<h3>Login failed: {error}</h3><a href='/auth/login'>Try again</a>"
        )
    if not code:
        return RedirectResponse("/auth/login")

    msal_app = get_msal_app()
    result = msal_app.acquire_token_by_authorization_code(
        code,
        scopes=SCOPES,
        redirect_uri=REDIRECT_URI,
    )

    if "error" in result:
        desc = result.get("error_description") or result.get("error")
        return HTMLResponse(f"<h3>Token error: {desc}</h3>")

    claims = result.get("id_token_claims") or {}
    email = (
        claims.get("preferred_username")
        or claims.get("email")
        or ""
    ).strip()
    name = (claims.get("name") or email).strip()

    if not email.lower().endswith(f"@{ALLOWED_DOMAIN}"):
        return HTMLResponse(
            f"<h3>Access denied.</h3><p>Only @{ALLOWED_DOMAIN} accounts are allowed.</p>"
        )

    request.session["user"] = {"email": email, "name": name}
    return RedirectResponse("/")


@app.get("/auth/logout")
async def logout(request: Request):
    request.session.clear()
    logout_base = (
        f"https://login.microsoftonline.com/{AZURE_TENANT_ID}/oauth2/v2.0/logout"
    )
    post_uri = urllib.parse.quote(POST_LOGOUT_REDIRECT_URI, safe="")
    return RedirectResponse(f"{logout_base}?post_logout_redirect_uri={post_uri}")


@app.get("/auth/me")
async def me(request: Request):
    user = request.session.get("user")
    if not user:
        return {"authenticated": False}
    return {"authenticated": True, **user}


@app.get("/", response_class=HTMLResponse)
async def root(request: Request):
    user = request.session.get("user")
    if not user:
        return RedirectResponse("/auth/login")
    with open("static/index.html", encoding="utf-8") as f:
        return f.read()


@app.post("/query")
async def query_endpoint(body: Query, user=Depends(require_auth)):
    if not body.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty")
    try:
        from memory_engine import ask

        return await asyncio.to_thread(ask, body.question, False)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e


@app.post("/meeting-prep")
async def meeting_prep_endpoint(body: MeetingPrepRequest, user=Depends(require_auth)):
    company = body.company.strip()
    if not company:
        raise HTTPException(status_code=400, detail="Company name required")
    try:
        return await asyncio.to_thread(_meeting_prep_sync, company)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e


@app.post("/score-deal")
async def score_deal_endpoint(body: ScoreDealRequest, user=Depends(require_auth)):
    company = body.company.strip()
    description = body.description.strip()
    if not company or not description:
        raise HTTPException(
            status_code=400, detail="Company and description required"
        )
    payload = body.model_copy(
        update={
            "company": company,
            "description": description,
            "sector": body.sector.strip(),
            "stage": body.stage.strip(),
            "ask": body.ask.strip(),
            "extra_context": body.extra_context.strip(),
        }
    )
    try:
        return await asyncio.to_thread(_score_deal_sync, payload)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e


@app.get("/digest")
async def get_digest(user=Depends(require_auth)):
    from proactive_engine import get_latest_digest

    digest = get_latest_digest()
    if not digest:
        return {"digest": "No digest generated yet.", "generated_at": None}
    return digest


@app.post("/digest/generate")
async def generate_digest(user=Depends(require_auth)):
    from proactive_engine import generate_weekly_digest

    return await asyncio.to_thread(generate_weekly_digest)


@app.get("/sync-status")
async def sync_status(user=Depends(require_auth)):
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
async def trigger_sync(user=Depends(require_auth)):
    from sync_affinity import run_sync

    results = await asyncio.to_thread(run_sync)
    return {"status": "complete", "results": results}


@app.get("/health")
async def health():
    return {"status": "ok"}
