import json
import os
from datetime import datetime

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field

from pattern_engine import (
    analyze_engagement_patterns,
    analyze_sector_distribution,
    get_all_entries,
    get_name,
    get_note_text,
)

load_dotenv()

llm = ChatOpenAI(
    model="gpt-4o",
    temperature=0,
    openai_api_key=os.getenv("OPENAI_API_KEY"),
)


class DealScore(BaseModel):
    overall_score: int = Field(description="Overall fit score 1-10")
    thesis_fit: int = Field(description="How well it fits DVC thesis 1-10")
    team_score: int = Field(description="Team quality score 1-10")
    market_score: int = Field(description="Market opportunity score 1-10")
    traction_score: int = Field(description="Traction and momentum score 1-10")
    pass_risk: int = Field(description="Risk of passing on a good deal 1-10")
    recommendation: str = Field(description="Pass / Watch / Pursue / Priority")
    thesis_alignment: str = Field(description="How it aligns with DVC thesis")
    key_strengths: list[str] = Field(description="Top 3 strengths")
    key_concerns: list[str] = Field(description="Top 3 concerns")
    comparable_deals: list[str] = Field(description="Similar companies we've seen")
    suggested_diligence: list[str] = Field(description="Key diligence questions")
    one_line_summary: str = Field(
        description="One sentence investment thesis or pass reason"
    )


def build_thesis_context() -> str:
    data = get_all_entries()
    all_entries = data["tracker"] + data["us_pipeline"] + data["india_pipeline"]

    sectors = analyze_sector_distribution(all_entries)
    engagement = analyze_engagement_patterns(all_entries)

    invested_signals = []
    passed_signals = []

    for entry in all_entries:
        text = get_note_text(entry)
        name = get_name(entry)
        if any(k in text for k in ["invested", "closed", "portfolio"]):
            invested_signals.append(name)
        elif any(k in text for k in ["pass", "passed", "not a fit", "too early"]):
            passed_signals.append(name)

    return f"""DVC Investment Thesis Context:

Top sectors by pipeline volume:
{json.dumps(sectors['distribution'], indent=2)}

Companies invested in or close to closing:
{', '.join(invested_signals[:20])}

Companies we passed on (sample names for pattern context):
{', '.join(passed_signals[:20])}

Patterns among passes (qualitative themes to weigh):
- Too early: common pass reason
- Customer concentration risk: seen in multiple passes
- Not B2B SaaS: outside thesis
- Dual US/India structure: complicates fundraising

Active pipeline companies: {engagement['active_count']}
Cold companies: {engagement['cold_count']}
"""


def score_deal(
    company_name: str,
    description: str,
    sector: str = "",
    stage: str = "",
    ask: str = "",
    extra_context: str = "",
) -> DealScore:
    thesis_context = build_thesis_context()

    user_prompt = f"""You are a senior VC analyst at Dallas Venture Capital scoring a new deal.

COMPANY: {company_name}
DESCRIPTION: {description}
SECTOR: {sector}
STAGE: {stage}
ASK: {ask}
ADDITIONAL CONTEXT: {extra_context}

DVC HISTORICAL CONTEXT:
{thesis_context}

Score this deal against DVC's thesis and historical patterns.
Be honest and specific. Reference comparable companies we've seen where relevant."""

    structured_llm = llm.with_structured_output(DealScore)
    messages = [
        SystemMessage(
            content=(
                "You are a rigorous VC analyst. Score deals honestly based on "
                "historical patterns."
            )
        ),
        HumanMessage(content=user_prompt),
    ]
    return structured_llm.invoke(messages)


def format_score(score: DealScore, company_name: str) -> str:
    rec_colors = {
        "Pass": "🔴",
        "Watch": "🟡",
        "Pursue": "🟢",
        "Priority": "⭐",
    }
    icon = rec_colors.get(score.recommendation, "⚪")

    lines = [
        f"## Deal Score: {company_name}",
        f"### {icon} {score.recommendation} — Overall: {score.overall_score}/10",
        f"\n_{score.one_line_summary}_",
        "\n### Scorecard",
        "| Dimension | Score |",
        "|---|---|",
        f"| Thesis fit | {score.thesis_fit}/10 |",
        f"| Team | {score.team_score}/10 |",
        f"| Market | {score.market_score}/10 |",
        f"| Traction | {score.traction_score}/10 |",
        f"| Pass risk | {score.pass_risk}/10 |",
        "\n### Thesis alignment",
        score.thesis_alignment,
        "\n### Strengths",
        *[f"- {s}" for s in score.key_strengths],
        "\n### Concerns",
        *[f"- {c}" for c in score.key_concerns],
        "\n### Comparable deals we've seen",
        *[f"- {d}" for d in score.comparable_deals],
        "\n### Diligence questions",
        *[f"- {q}" for q in score.suggested_diligence],
    ]
    return "\n".join(lines)
