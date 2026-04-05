import json
import os
from collections import defaultdict
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

load_dotenv()

llm = ChatOpenAI(
    model="gpt-4o",
    temperature=0,
    openai_api_key=os.getenv("OPENAI_API_KEY"),
)

CACHE_DIR = Path("affinity_cache")


def load(name):
    path = CACHE_DIR / f"{name.replace(' ', '_').lower()}.json"
    if path.exists():
        with open(path, encoding="utf-8") as f:
            return json.load(f)["data"]
    return []


def get_all_entries():
    tracker = load("Master DVC Advantage Tracker")
    us_pipeline = load("DVC US Pipeline")
    india_pipeline = load("DVC India Pipeline")
    interactions = load("DVC VC Ecosystem Interactions")
    return {
        "tracker": tracker,
        "us_pipeline": us_pipeline,
        "india_pipeline": india_pipeline,
        "interactions": interactions,
    }


def get_name(entry):
    entity = entry.get("entity", {}) or {}
    affinity = entry.get("_affinity", {}) or {}
    return (
        affinity.get("company_name") or entity.get("name", "") or "Unknown"
    ).strip()


def get_date(entry):
    created = entry.get("created_at", "")
    return created[:10] if created else ""


def get_notes(entry):
    affinity = entry.get("_affinity", {}) or {}
    return affinity.get("notes", []) or []


def get_note_text(entry):
    notes = get_notes(entry)
    return " ".join([n.get("content", "") for n in notes]).lower()


def days_since(date_str):
    if not date_str:
        return 999
    try:
        date = datetime.fromisoformat(date_str.replace("Z", "+00:00"))
        return (datetime.now(date.tzinfo) - date).days
    except Exception:
        return 999


def get_latest_note_date(entry):
    notes = get_notes(entry)
    if not notes:
        return ""
    dates = [n.get("created_at", "")[:10] for n in notes if n.get("created_at")]
    return max(dates) if dates else ""


def analyze_pipeline_velocity(entries) -> dict:
    by_month = defaultdict(int)
    for entry in entries:
        date = get_date(entry)
        if date:
            month = date[:7]
            by_month[month] += 1
    sorted_months = sorted(by_month.items())
    return {
        "by_month": dict(sorted_months),
        "total": len(entries),
        "peak_month": max(by_month, key=by_month.get) if by_month else None,
        "recent_30_days": sum(
            1 for e in entries if days_since(get_date(e)) <= 30
        ),
    }


def analyze_engagement_patterns(entries) -> dict:
    cold_keywords = ["cold", "dead end", "not interested", "no response", "marked cold"]
    active_keywords = [
        "re-engaging",
        "follow up",
        "scheduled",
        "meeting",
        "interested",
    ]
    raise_keywords = ["raising", "fundraising", "round", "seeking investment"]

    cold = []
    active = []
    raising = []
    no_notes = []

    for entry in entries:
        text = get_note_text(entry)
        name = get_name(entry)
        if not text:
            no_notes.append(name)
        elif any(k in text for k in cold_keywords):
            cold.append(name)
        elif any(k in text for k in raise_keywords):
            raising.append(name)
        elif any(k in text for k in active_keywords):
            active.append(name)

    return {
        "cold_count": len(cold),
        "active_count": len(active),
        "raising_count": len(raising),
        "no_notes_count": len(no_notes),
        "cold_companies": cold[:20],
        "active_companies": active[:20],
        "raising_companies": raising[:10],
    }


def analyze_sector_distribution(entries) -> dict:
    sectors = {
        "cybersecurity": [
            "security",
            "cyber",
            "browser",
            "endpoint",
            "threat",
            "sase",
            "zero trust",
        ],
        "ai_ml": [
            "ai",
            "machine learning",
            "llm",
            "generative",
            "artificial intelligence",
            "nlp",
        ],
        "fintech": [
            "fintech",
            "financial",
            "payment",
            "banking",
            "lending",
            "insurtech",
        ],
        "saas_b2b": [
            "saas",
            "b2b",
            "enterprise software",
            "workflow",
            "automation",
            "crm",
        ],
        "healthtech": [
            "health",
            "medical",
            "clinical",
            "pharma",
            "biotech",
            "wellness",
        ],
        "edtech": [
            "education",
            "learning",
            "edtech",
            "training",
            "skills",
        ],
        "devtools": [
            "developer",
            "devops",
            "api",
            "infrastructure",
            "cloud",
            "platform",
        ],
        "hr_future_of_work": [
            "hr",
            "talent",
            "workforce",
            "remote work",
            "hiring",
            "payroll",
        ],
    }

    counts = defaultdict(int)
    company_sectors = defaultdict(list)

    for entry in entries:
        text = get_note_text(entry)
        name = get_name(entry)
        for sector, keywords in sectors.items():
            if any(k in text for k in keywords):
                counts[sector] += 1
                company_sectors[sector].append(name)

    return {
        "distribution": dict(
            sorted(counts.items(), key=lambda x: x[1], reverse=True)
        ),
        "by_sector": {k: v[:10] for k, v in company_sectors.items()},
    }


def analyze_stale_deals(entries, days_threshold=90) -> list:
    stale = []
    for entry in entries:
        name = get_name(entry)
        latest_note = get_latest_note_date(entry)
        entry_date = get_date(entry)
        reference_date = latest_note or entry_date
        age = days_since(reference_date)
        text = get_note_text(entry)
        is_cold = any(
            k in text for k in ["cold", "dead end", "not interested"]
        )
        if age >= days_threshold and not is_cold and text:
            stale.append(
                {
                    "company": name,
                    "last_activity": reference_date,
                    "days_inactive": age,
                    "has_notes": bool(text),
                }
            )
    return sorted(stale, key=lambda x: x["days_inactive"])


def analyze_source_effectiveness(entries) -> dict:
    source_keywords = {
        "TIAA Ventures": ["tiaa"],
        "Dreamit": ["dreamit"],
        "Direct outreach": ["outreach", "cold"],
        "Network intro": ["introduced", "introduction", "referred"],
        "Inbound": ["inbound", "reached out to us"],
    }
    source_counts = defaultdict(int)
    for entry in entries:
        text = get_note_text(entry)
        for source, keywords in source_keywords.items():
            if any(k in text for k in keywords):
                source_counts[source] += 1
    return dict(
        sorted(source_counts.items(), key=lambda x: x[1], reverse=True)
    )


def run_full_analysis() -> dict:
    data = get_all_entries()
    all_entries = (
        data["tracker"] + data["us_pipeline"] + data["india_pipeline"]
    )

    print("Running pattern analysis...")
    return {
        "pipeline_velocity": analyze_pipeline_velocity(all_entries),
        "engagement": analyze_engagement_patterns(all_entries),
        "sectors": analyze_sector_distribution(all_entries),
        "stale_deals": analyze_stale_deals(all_entries),
        "source_effectiveness": analyze_source_effectiveness(all_entries),
        "generated_at": datetime.now().isoformat(),
    }


def answer_pattern_query(query: str) -> str:
    analysis = run_full_analysis()

    context = f"""
Pipeline velocity:
- Total companies tracked: {analysis['pipeline_velocity']['total']}
- Added in last 30 days: {analysis['pipeline_velocity']['recent_30_days']}
- Peak month: {analysis['pipeline_velocity']['peak_month']}
- By month: {json.dumps(analysis['pipeline_velocity']['by_month'])}

Engagement breakdown:
- Active/re-engaging: {analysis['engagement']['active_count']} companies
- Cold/unresponsive: {analysis['engagement']['cold_count']} companies
- Currently raising: {analysis['engagement']['raising_count']} companies
- No notes recorded: {analysis['engagement']['no_notes_count']} companies
- Active companies: {', '.join(analysis['engagement']['active_companies'][:10])}
- Raising: {', '.join(analysis['engagement']['raising_companies'])}

Sector distribution:
{json.dumps(analysis['sectors']['distribution'], indent=2)}

Stale deals (90+ days no activity, not cold):
{json.dumps([s for s in analysis['stale_deals'][:15]], indent=2)}

Deal source effectiveness:
{json.dumps(analysis['source_effectiveness'], indent=2)}
"""

    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """You are a senior analyst at Dallas Venture Capital.
You have access to aggregated pipeline analytics. Answer the question with specific numbers,
named companies where relevant, and actionable insights.
Be direct and concise — this is for a partner meeting.""",
            ),
            ("human", "Question: {query}\n\nData:\n{context}"),
        ]
    )

    chain = prompt | llm
    out = chain.invoke({"query": query, "context": context})
    return out.content
