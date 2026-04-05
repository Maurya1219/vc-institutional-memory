import json
import os
from datetime import datetime
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

from pattern_engine import (
    analyze_engagement_patterns,
    analyze_stale_deals,
    get_all_entries,
    get_date,
    get_latest_note_date,
    get_name,
    get_note_text,
    days_since,
)

load_dotenv()

llm = ChatOpenAI(
    model="gpt-4o",
    temperature=0.2,
    openai_api_key=os.getenv("OPENAI_API_KEY"),
)

DIGEST_LOG = Path("affinity_cache/digest_log.json")


def find_reengagement_opportunities(entries) -> list:
    opportunities = []
    for entry in entries:
        text = get_note_text(entry)
        name = get_name(entry)
        latest = get_latest_note_date(entry)
        age = days_since(latest or get_date(entry))

        showed_interest = any(
            k in text
            for k in [
                "interested",
                "impressed",
                "promising",
                "strong",
                "good fit",
                "excited",
                "potential",
            ]
        )
        not_cold = not any(
            k in text
            for k in [
                "cold",
                "dead end",
                "not interested",
                "pass",
                "passed",
            ]
        )

        if showed_interest and not_cold and 60 <= age <= 365:
            opportunities.append(
                {
                    "company": name,
                    "last_activity": latest,
                    "days_inactive": age,
                    "signal": "showed interest, went quiet",
                }
            )

    return sorted(opportunities, key=lambda x: x["days_inactive"])[:10]


def find_thesis_matches(entries) -> list:
    thesis_keywords = [
        "b2b saas",
        "enterprise",
        "ai",
        "machine learning",
        "cybersecurity",
        "fintech",
        "series a",
        "series b",
        "strong team",
        "iit",
        "repeat founder",
    ]
    anti_thesis = [
        "consumer",
        "b2c",
        "hardware",
        "pre-revenue",
        "cold",
        "dead end",
        "not interested",
    ]

    matches = []
    for entry in entries:
        text = get_note_text(entry)
        name = get_name(entry)
        latest = get_latest_note_date(entry)
        age = days_since(latest or get_date(entry))

        thesis_score = sum(1 for k in thesis_keywords if k in text)
        anti_score = sum(1 for k in anti_thesis if k in text)

        if thesis_score >= 2 and anti_score == 0 and age > 45:
            matches.append(
                {
                    "company": name,
                    "thesis_score": thesis_score,
                    "days_inactive": age,
                    "last_activity": latest,
                }
            )

    return sorted(matches, key=lambda x: x["thesis_score"], reverse=True)[:10]


def find_network_leverage(entries) -> list:
    high_value_intros = []
    intro_keywords = [
        "introduced by",
        "introduction from",
        "referred by",
        "via tiaa",
        "via dreamit",
    ]

    for entry in entries:
        text = get_note_text(entry)
        name = get_name(entry)
        latest = get_latest_note_date(entry)
        age = days_since(latest or get_date(entry))

        has_intro = any(k in text for k in intro_keywords)
        not_cold = not any(
            k in text for k in ["cold", "dead end", "not interested"]
        )

        if has_intro and not_cold and age > 30:
            high_value_intros.append(
                {
                    "company": name,
                    "days_inactive": age,
                    "last_activity": latest,
                    "signal": "warm intro, needs follow-up",
                }
            )

    return sorted(high_value_intros, key=lambda x: x["days_inactive"])[:10]


def generate_weekly_digest() -> dict:
    data = get_all_entries()
    all_entries = (
        data["tracker"] + data["us_pipeline"] + data["india_pipeline"]
    )

    print("Generating weekly digest...")

    stale = analyze_stale_deals(all_entries, days_threshold=60)
    engagement = analyze_engagement_patterns(all_entries)
    reengagement = find_reengagement_opportunities(all_entries)
    thesis_matches = find_thesis_matches(all_entries)
    network = find_network_leverage(all_entries)

    context = f"""
Weekly Pipeline Intelligence Digest
Generated: {datetime.now().strftime('%B %d, %Y')}

PIPELINE HEALTH:
- Active companies: {engagement['active_count']}
- Cold/dead: {engagement['cold_count']}
- Currently raising: {engagement['raising_count']}

TOP RE-ENGAGEMENT OPPORTUNITIES (showed interest, went quiet):
{json.dumps(reengagement[:5], indent=2)}

THESIS MATCHES NEEDING FOLLOW-UP:
{json.dumps(thesis_matches[:5], indent=2)}

WARM INTROS NEEDING ACTION:
{json.dumps(network[:5], indent=2)}

STALE DEALS (60+ days, not cold):
{json.dumps(stale[:10], indent=2)}
"""

    prompt = f"""You are a chief of staff at Dallas Venture Capital writing the weekly pipeline digest for partners.

Write a crisp, actionable digest based on this data. Format it as:

## Weekly Pipeline Digest — {datetime.now().strftime('%B %d, %Y')}

### Pipeline health
[2-3 sentences on overall pipeline status]

### Top 3 re-engagement opportunities
[List the 3 most promising companies to re-engage this week with one line each]

### Deals needing immediate attention
[List deals that match thesis but have gone quiet]

### Warm intros to act on
[Network introductions that haven't been followed up]

### This week's recommended actions
[3-5 specific action items]

Keep it under 400 words. Be specific with company names and timeframes.

Data:
{context}"""

    digest = llm.invoke(prompt).content

    result = {
        "digest": digest,
        "generated_at": datetime.now().isoformat(),
        "stats": {
            "active": engagement["active_count"],
            "cold": engagement["cold_count"],
            "raising": engagement["raising_count"],
            "stale": len(stale),
            "reengagement_opportunities": len(reengagement),
        },
    }

    log = []
    if DIGEST_LOG.exists():
        with open(DIGEST_LOG, encoding="utf-8") as f:
            log = json.load(f)
    log.append(result)
    log = log[-12:]
    DIGEST_LOG.parent.mkdir(parents=True, exist_ok=True)
    with open(DIGEST_LOG, "w", encoding="utf-8") as f:
        json.dump(log, f, indent=2)

    return result


def get_latest_digest() -> Optional[dict]:
    if not DIGEST_LOG.exists():
        return None
    with open(DIGEST_LOG, encoding="utf-8") as f:
        log = json.load(f)
    return log[-1] if log else None
