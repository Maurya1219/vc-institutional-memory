import re
from datetime import datetime

from langchain_core.documents import Document
from langchain_openai import ChatOpenAI

PORTFOLIO_DOMAINS = [
    "amplifai.com",
    "arnica.io",
    "secufusion.ai",
    "bluesapphire.com",
    "nirmata.com",
]

DVC_PARTNERS = [
    "sriram",
    "ravish",
    "mridul",
    "manish",
    "ashok",
    "dayakar",
]


def score_source(doc: Document, query_company: str = "") -> dict:
    source = doc.metadata.get("source", "")
    date = doc.metadata.get("date", "") or doc.metadata.get("modified", "")
    content = doc.page_content.lower()
    sender = doc.metadata.get("sender", "").lower()
    filename = doc.metadata.get("filename", "").lower()
    url = doc.metadata.get("url", "").lower()

    credibility = 30
    source_label = "Unknown"
    recency_bonus = 0

    if source == "email":
        source_label = "Email"
        credibility = 45

        company_lower = query_company.lower()

        is_from_company = any(
            domain in sender
            for domain in PORTFOLIO_DOMAINS
            if company_lower and (company_lower in domain or domain.split(".")[0] in company_lower)
        ) or (company_lower and len(company_lower) >= 4 and company_lower[:6] in sender)

        if is_from_company:
            credibility = 90
            source_label = f"Email from {sender}"

            if any(title in content for title in ["cfo", "ceo", "coo", "cto", "chief"]):
                credibility = 96
                source_label = f"Email from executive at {query_company}"

        elif any(p in sender for p in DVC_PARTNERS):
            credibility = 55
            source_label = "Email from DVC partner"

        elif company_lower and company_lower in content:
            credibility = 60
            source_label = "Email mentioning company"

    elif source == "sharepoint":
        source_label = f"Document: {filename}"
        credibility = 55

        if any(k in filename for k in ["financial", "model", "metrics", "kpi", "board", "update"]):
            credibility = 82
            source_label = f"Financial document: {filename}"
        elif any(k in filename for k in ["pitch", "deck", "investor"]):
            credibility = 75
            source_label = f"Pitch deck: {filename}"
        elif any(k in filename for k in ["memo", "investment"]):
            credibility = 70
            source_label = f"Investment memo: {filename}"

    elif source == "affinity":
        source_label = "Affinity partner note"
        credibility = 48

        if any(k in content for k in ["nrr", "arr", "mrr", "revenue", "burn", "runway"]):
            credibility = 62
            source_label = "Partner note with metrics"

        if any(f'"{k}' in content or f"'{k}" in content for k in ["nrr", "arr", "mrr"]):
            credibility = 72
            source_label = "Partner note with quoted metrics"

    elif source == "affinity_contact":
        credibility = 35
        source_label = "CRM contact record"

    elif source == "website":
        credibility = 35
        source_label = f"Public website: {url}"

        if any(k in content for k in ["press release", "announced", "raised"]):
            credibility = 42
            source_label = "Press release / public announcement"

    days_old = 999
    if date:
        try:
            parsed = datetime.fromisoformat(date.replace("Z", "+00:00"))
            days_old = (datetime.now(parsed.tzinfo) - parsed).days
        except Exception:
            try:
                parsed = datetime.strptime(date[:10], "%Y-%m-%d")
                days_old = (datetime.now() - parsed).days
            except Exception:
                pass

    if days_old <= 7:
        recency_bonus = 15
        recency_label = "This week"
    elif days_old <= 30:
        recency_bonus = 10
        recency_label = "This month"
    elif days_old <= 90:
        recency_bonus = 5
        recency_label = "Last 3 months"
    elif days_old <= 180:
        recency_bonus = 0
        recency_label = "Last 6 months"
    elif days_old <= 365:
        recency_bonus = -5
        recency_label = "Last year"
    else:
        recency_bonus = -15
        recency_label = f"{max(1, days_old // 365)}+ years ago"

    final_score = min(99, max(1, credibility + recency_bonus))

    return {
        "credibility": final_score,
        "base_credibility": credibility,
        "recency_bonus": recency_bonus,
        "source_label": source_label,
        "date": date[:10] if date else "Unknown date",
        "days_old": days_old,
        "recency_label": recency_label,
    }


def rank_docs_for_metrics(
    docs: list[Document],
    query_company: str = "",
) -> list[tuple[Document, dict]]:
    scored = []
    for doc in docs:
        score_info = score_source(doc, query_company)
        scored.append((doc, score_info))

    scored.sort(
        key=lambda x: (x[1]["credibility"], -x[1]["days_old"]),
        reverse=True,
    )
    return scored


def format_metric_answer(
    query: str,
    ranked_docs: list[tuple[Document, dict]],
    llm: ChatOpenAI,
) -> str:
    if not ranked_docs:
        return "No relevant information found for this metric."

    context_parts = []
    for doc, score_info in ranked_docs[:6]:
        context_parts.append(
            f"[Credibility: {score_info['credibility']}/100 | "
            f"Source: {score_info['source_label']} | "
            f"Date: {score_info['date']} ({score_info['recency_label']})]\n"
            f"{doc.page_content[:600]}"
        )

    context = "\n\n---\n\n".join(context_parts)

    prompt = f"""You are a precise financial analyst for Dallas Venture Capital.

Query: {query}

Retrieved information (ranked by source credibility and recency):
{context}

Instructions:
- State the metric clearly with its value if found
- Always cite the source credibility score and date
- If multiple sources disagree, show the most credible/recent one first and note the discrepancy
- If the top source has credibility below 60, flag it as uncertain
- If no metric is found, say so explicitly — never estimate
- Format numbers clearly (e.g. "$2.4M ARR", "118% NRR")

Format your answer as:

**[Metric]: [Value]**
Source: [source label] · Credibility: [score]/100 · As of: [date]

[Any discrepancies or caveats]"""

    return llm.invoke(prompt).content


def guess_company_from_query(query: str) -> str:
    skip = {
        "what",
        "who",
        "how",
        "when",
        "where",
        "which",
        "the",
        "a",
        "an",
        "is",
        "are",
        "was",
        "were",
        "does",
        "did",
        "for",
        "with",
        "from",
        "about",
        "nrr",
        "arr",
        "mrr",
        "burn",
        "runway",
        "rate",
    }
    possessive = re.search(
        r"\b([A-Za-z][A-Za-z0-9.\-]*)\s*'s\b", query
    )
    if possessive:
        w = possessive.group(1)
        if len(w) > 2 and w.lower() not in skip:
            return w
    for raw in re.findall(r"[A-Za-z][A-Za-z0-9.\-]*", query):
        w = raw.strip(".,?!'\"")
        if len(w) <= 2:
            continue
        if w.lower() in skip:
            continue
        if w[0].isupper():
            return w
    return ""
