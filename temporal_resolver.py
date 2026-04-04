import os
from datetime import datetime, timezone
from typing import Optional

from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field

load_dotenv()

llm = ChatOpenAI(
    model="gpt-4o",
    temperature=0,
    openai_api_key=os.getenv("OPENAI_API_KEY"),
)


class TemporalResolution(BaseModel):
    resolved_status: str = Field(
        description="The current best-known status based on all available information"
    )
    confidence: str = Field(
        description="high, medium, or low — how confident we are this is current"
    )
    conflicts_detected: bool = Field(
        description="Whether conflicting information was found across time periods"
    )
    conflict_summary: Optional[str] = Field(
        default=None,
        description="Brief summary of the conflict if one exists",
    )
    oldest_info: Optional[str] = Field(
        default=None,
        description="What the oldest information says",
    )
    newest_info: Optional[str] = Field(
        default=None,
        description="What the newest information says",
    )
    recommendation: str = Field(
        description="What action the VC should take given this information"
    )


TEMPORAL_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """You are a temporal reasoning engine for a venture capital firm.

You will receive multiple pieces of information about companies, each with a date.
Your job is to resolve conflicting information by treating newer information as ground truth.

Key rules:
1. NEWER information always supersedes OLDER information about the same fact
2. If a company was "raising" in October but "closed round" in February, they are NOT raising
3. If notes say "marked cold" after earlier notes said "interested", they are cold
4. If engagement went from "meeting scheduled" to "no response for 3 weeks", flag as stale
5. A company with only old notes (6+ months ago) and no recent activity should be flagged as stale

Always be explicit about WHEN information was recorded and flag any gaps in recent data.""",
        ),
        (
            "human",
            """Query: {query}

Retrieved information (sorted by date, newest first):
{context}

Resolve the temporal status of the relevant companies for this query.""",
        ),
    ]
)

structured_llm = llm.with_structured_output(TemporalResolution)


def parse_date(date_str: str) -> datetime:
    if not date_str:
        return datetime.min.replace(tzinfo=timezone.utc)
    try:
        dt = datetime.fromisoformat(date_str.replace("Z", "+00:00"))
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt
    except Exception:
        try:
            dt = datetime.strptime(date_str[:10], "%Y-%m-%d")
            return dt.replace(tzinfo=timezone.utc)
        except Exception:
            return datetime.min.replace(tzinfo=timezone.utc)


def group_by_company(docs: list[Document]) -> dict:
    groups: dict[str, list[Document]] = {}
    for doc in docs:
        company = doc.metadata.get("company", "Unknown")
        if company not in groups:
            groups[company] = []
        groups[company].append(doc)
    for company in groups:
        groups[company].sort(
            key=lambda d: parse_date(str(d.metadata.get("date", ""))),
            reverse=True,
        )
    return groups


def format_context_for_resolution(groups: dict) -> str:
    lines = []
    for company, company_docs in groups.items():
        lines.append(f"\n=== {company} ===")
        for doc in company_docs:
            date = doc.metadata.get("date", "unknown date")
            lines.append(f"[{date}]\n{doc.page_content[:500]}")
    return "\n".join(lines)


def detect_staleness(docs: list[Document]) -> bool:
    if not docs:
        return True
    newest_date = max(
        parse_date(str(d.metadata.get("date", ""))) for d in docs
    )
    invalid = datetime.min.replace(tzinfo=timezone.utc)
    if newest_date == invalid:
        return True
    now = datetime.now(timezone.utc)
    nd = newest_date.astimezone(timezone.utc)
    days_since = (now - nd).days
    return days_since > 180


def resolve_temporal(query: str, docs: list[Document]) -> dict:
    if not docs:
        return {
            "answer": "No relevant information found in the deal history.",
            "conflicts": False,
            "confidence": "low",
            "stale_companies": [],
        }

    groups = group_by_company(docs)
    context = format_context_for_resolution(groups)

    chain = TEMPORAL_PROMPT | structured_llm
    resolution = chain.invoke({"query": query, "context": context})

    stale_companies = [
        company
        for company, company_docs in groups.items()
        if detect_staleness(company_docs)
    ]

    answer = f"{resolution.resolved_status}\n"

    if resolution.conflicts_detected and resolution.conflict_summary:
        answer += f"\nConflict detected: {resolution.conflict_summary}"
        if resolution.oldest_info:
            answer += f"\n- Earlier: {resolution.oldest_info}"
        if resolution.newest_info:
            answer += f"\n- Latest: {resolution.newest_info}"

    if stale_companies:
        answer += (
            "\n\nNote: The following companies have no activity in 6+ months "
            f"and may need follow-up: {', '.join(stale_companies)}"
        )

    answer += f"\n\nRecommendation: {resolution.recommendation}"
    answer += f"\nConfidence: {resolution.confidence}"

    return {
        "answer": answer,
        "conflicts": resolution.conflicts_detected,
        "confidence": resolution.confidence,
        "stale_companies": stale_companies,
    }
