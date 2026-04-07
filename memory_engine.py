import json
import os
import re
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv
from langchain_classic.agents import AgentExecutor, create_openai_functions_agent
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.tools import Tool
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

from classifier import classify_query
from critic import critique_answer
from graph_query import answer_graph_query
from pattern_engine import answer_pattern_query
from query_expander import expand_query
from reranker import rerank
from source_ranker import format_metric_answer, guess_company_from_query, rank_docs_for_metrics
from temporal_resolver import resolve_temporal

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

_ROOT = Path(__file__).resolve().parent
_VECTORSTORE_DIR = _ROOT / "dvc_vectorstore"

embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
llm = ChatOpenAI(
    model="gpt-4o",
    temperature=0,
    openai_api_key=OPENAI_API_KEY,
)

vectorstore = None


def get_vectorstore():
    global vectorstore
    if vectorstore is None:
        vectorstore = FAISS.load_local(
            str(_VECTORSTORE_DIR),
            embeddings,
            allow_dangerous_deserialization=True,
        )
    return vectorstore


def reload_vectorstore() -> None:
    """Drop in-memory FAISS after sync/rebuild on disk."""
    global vectorstore
    vectorstore = None


def dedupe_docs(docs: list[Document]) -> list[Document]:
    seen = set()
    unique = []
    for doc in docs:
        key = doc.page_content[:100]
        if key not in seen:
            seen.add(key)
            unique.append(doc)
    return unique


def search_deals(query: str, k: int = 20) -> str:
    queries = expand_query(query)
    all_docs = []
    for q in queries:
        docs = get_vectorstore().similarity_search(q, k=10)
        all_docs.extend(docs)
    all_docs = dedupe_docs(all_docs)
    all_docs = rerank(query, all_docs, top_n=5)
    return "\n\n".join(
        [
            f"Company: {d.metadata.get('company')}\n"
            f"List: {d.metadata.get('list')}\n"
            f"Date: {d.metadata.get('date')}\n"
            f"{d.page_content}"
            for d in all_docs
        ]
    )


def search_recent(query: str) -> str:
    queries = expand_query(query)
    all_docs = []
    for q in queries:
        docs = get_vectorstore().similarity_search(q, k=10)
        all_docs.extend(docs)
    all_docs = dedupe_docs(all_docs)
    all_docs = sorted(
        all_docs,
        key=lambda d: d.metadata.get("date", ""),
        reverse=True,
    )
    all_docs = rerank(query, all_docs[:20], top_n=5)
    return "\n\n".join(
        [
            f"Company: {d.metadata.get('company')}\n"
            f"Date added: {d.metadata.get('date')}\n"
            f"{d.page_content}"
            for d in all_docs
        ]
    )


def search_with_notes(query: str) -> str:
    queries = expand_query(query)
    all_docs = []
    for q in queries:
        docs = get_vectorstore().similarity_search(q, k=10)
        all_docs.extend(docs)
    all_docs = dedupe_docs(all_docs)
    docs_with_notes = [
        d
        for d in all_docs
        if "partner notes" in d.page_content.lower()
        and "no partner notes" not in d.page_content.lower()
    ]
    candidates = docs_with_notes if docs_with_notes else all_docs
    reranked = rerank(query, candidates, top_n=5)
    return "\n\n".join(
        [
            f"Company: {d.metadata.get('company')}\n"
            f"Date: {d.metadata.get('date')}\n"
            f"{d.page_content}"
            for d in reranked
        ]
    )


def build_system_prompt(classification) -> str:
    base = """You are an AI analyst for Dallas Venture Capital with access to the firm's complete deal history.

Be specific: name companies, dates, and context from the data.
Always cite which companies or notes your answer comes from.
If information seems incomplete, say so clearly."""

    category_guidance = {
        "deal_search": "\nFocus on finding relevant companies and what the firm knows about them.",
        "relationship": "\nFocus on people, introductions, and relationship context in the notes.",
        "temporal": "\nIMPORTANT: Pay close attention to dates. Newer information supersedes older information. If a company was raising in October but the notes show they closed in February, they are NOT currently raising. Always note when information was recorded.",
        "portfolio": "\nFocus on portfolio companies and investment status.",
        "counting": "\nCount and aggregate carefully. Be precise about numbers and time ranges.",
        "factual": "\nAnswer only with explicit numbers or facts from the data; if missing, say it is not in records.",
        "graph": "\nUse the relationship graph for connectivity and paths between entities.",
        "pattern": "\nUse aggregated pipeline analytics (velocity, sectors, engagement, sources).",
        "metrics": "\nUse source-ranked retrieval; cite credibility and dates for KPIs.",
        "general": "",
    }

    return base + category_guidance.get(classification.category, "")


def _extract_meeting_prep_company(query: str) -> Optional[str]:
    raw = query.strip()
    low = raw.lower()
    if "meeting prep" not in low and "prepare for meeting" not in low:
        return None
    needles = [
        "prepare for meeting with ",
        "prepare for meeting for ",
        "prepare for meeting ",
        "meeting prep for ",
        "meeting prep:",
        "meeting prep — ",
        "meeting prep - ",
        "meeting prep ",
    ]
    for needle in needles:
        i = low.find(needle)
        if i != -1:
            rest = raw[i + len(needle) :].strip()
            if not rest:
                return None
            low_rest = rest.lower()
            if low_rest.startswith("with "):
                rest = rest[5:].strip()
            elif low_rest.startswith("for "):
                rest = rest[4:].strip()
            return rest or None
    return None


_GENERIC_EMAIL_DOMAINS = frozenset(
    {
        "gmail.com",
        "yahoo.com",
        "outlook.com",
        "hotmail.com",
        "icloud.com",
        "me.com",
        "aol.com",
        "protonmail.com",
        "live.com",
    }
)


def enrich_person_query(name: str) -> str:
    """Cross-reference Affinity contacts with company deal history via email domain."""
    cache_dir = Path(__file__).resolve().parent / "affinity_cache"
    contacts_path = cache_dir / "dvc_contacts_4d.json"

    if not contacts_path.exists():
        return ""

    with open(contacts_path, encoding="utf-8") as f:
        contacts = json.load(f)["data"]

    name_lower = name.lower().strip()
    name_tokens = [t for t in re.split(r"\s+", name_lower) if t]
    matched = None
    for contact in contacts:
        entity = contact.get("entity", {}) or {}
        affinity = contact.get("_affinity", {}) or {}
        first = (entity.get("first_name") or "").strip().lower()
        last = (entity.get("last_name") or "").strip().lower()
        full_name = f"{first} {last}".strip()
        contact_name = (affinity.get("contact_name") or "").lower()
        if not full_name and not contact_name:
            continue
        if name_lower in full_name or name_lower in contact_name:
            matched = contact
            break
        if name_tokens and all(
            t in full_name or t in contact_name for t in name_tokens
        ):
            matched = contact
            break

    if not matched:
        return ""

    entity = matched.get("entity", {}) or {}
    affinity = matched.get("_affinity", {}) or {}
    email = (entity.get("primary_email") or affinity.get("email") or "").strip()
    full_name = (
        affinity.get("contact_name")
        or f"{entity.get('first_name', '')} {entity.get('last_name', '')}".strip()
    )

    company_domain = ""
    company_hint = ""
    if "@" in email:
        domain = email.split("@")[1].lower().strip()
        if domain and domain not in _GENERIC_EMAIL_DOMAINS:
            company_domain = domain
            company_hint = domain.split(".")[0].replace("-", " ").title()

    if not company_domain:
        return (
            f"{full_name} is in DVC's contact database. Email: {email or 'unknown'}. "
            f"No company association found from email domain."
        )

    company_context = search_deals(f"{company_hint} {company_domain}")

    return f"""Contact found: {full_name}
Email: {email}
Company domain: {company_domain} → likely works at {company_hint}

Company context from DVC deal history:
{company_context}"""


def ask(query: str, verbose: bool = False, user_context: str = "") -> dict:
    prep_company = _extract_meeting_prep_company(query)
    if prep_company:
        from meeting_prep import run_meeting_prep

        result = run_meeting_prep(
            prep_company,
            lambda name: search_with_notes(f"everything about {name}"),
            lambda name: answer_graph_query(f"who do we know at {name}"),
        )
        return {
            "answer": result["brief"],
            "category": "meeting_prep",
            "time_sensitive": False,
            "quality_score": 9,
            "grounded": True,
            "retried": False,
        }

    lq = query.lower().strip()
    person_query_patterns = [
        "who is ",
        "who's ",
        "tell me about ",
        "what does ",
        "what do you know about ",
    ]
    is_person_query = any(lq.startswith(p) for p in person_query_patterns)

    if is_person_query:
        name = query.strip()
        for p in person_query_patterns:
            if lq.startswith(p):
                name = query[len(p) :].strip()
                break

        words = name.split()
        company_markers = (
            "company",
            "firm",
            "inc",
            "llc",
            "corp",
            "venture",
            "capital",
        )
        looks_like_person = (
            2 <= len(words) <= 3
            and not any(k in name.lower() for k in company_markers)
            and all(re.match(r"^[A-Za-z][A-Za-z\-']*$", w) for w in words)
        )

        if looks_like_person:
            enriched = enrich_person_query(name)
            if enriched:
                if verbose:
                    print("\nPerson query: enriched from contacts + deal history")
                extra = (
                    f"\n\nAdditional user context: {user_context.strip()}"
                    if user_context.strip()
                    else ""
                )
                prompt = f"""You are an analyst at Dallas Venture Capital.
Someone asked: "{query}"

Here is everything we know:
{enriched}
{extra}

Answer the question directly and completely:
- Who is this person
- What company do they work for and what is their role if known
- What is DVC's relationship with their company
- Any relevant context from our deal history

Be specific. Use the company context to infer their role if not explicitly stated."""

                response = llm.invoke(prompt)
                return {
                    "answer": response.content,
                    "category": "relationship",
                    "time_sensitive": False,
                    "quality_score": 8,
                    "grounded": True,
                    "retried": False,
                }

    classification = classify_query(query)

    if verbose:
        print(f"\nClassified as: {classification.category}")
        print(f"Time sensitive: {classification.time_sensitive}")

    if classification.category == "factual":
        context = search_deals(query)

        factual_prompt = f"""You are a precise data analyst for Dallas Venture Capital.

The user is asking for a specific data point: {query}

Retrieved context:
{context}

Rules:
- If the exact data point is present in the context, state it clearly with the source
- If it is NOT in the context, say exactly: "This information is not available in our records. You may want to check the company's pitch deck or financial model directly."
- Never estimate or approximate
- Never pivot to related information — answer the specific question asked"""

        response = llm.invoke(factual_prompt)
        return {
            "answer": response.content,
            "category": "factual",
            "time_sensitive": classification.time_sensitive,
            "quality_score": 9,
            "grounded": True,
            "retried": False,
        }

    if classification.category == "graph":
        answer = answer_graph_query(query)
        return {
            "answer": answer,
            "category": "graph",
            "time_sensitive": classification.time_sensitive,
            "quality_score": 9,
            "grounded": True,
            "retried": False,
        }

    if classification.category == "pattern":
        answer = answer_pattern_query(query)
        return {
            "answer": answer,
            "category": "pattern",
            "time_sensitive": classification.time_sensitive,
            "quality_score": 9,
            "grounded": True,
            "retried": False,
        }

    if classification.category == "metrics":
        company = guess_company_from_query(query)
        raw_docs = get_vectorstore().similarity_search(query, k=20)
        for q in expand_query(query)[1:]:
            raw_docs.extend(get_vectorstore().similarity_search(q, k=10))
        raw_docs = dedupe_docs(raw_docs)
        raw_docs = rerank(query, raw_docs, top_n=10)

        ranked = rank_docs_for_metrics(raw_docs, company)
        answer = format_metric_answer(query, ranked, llm)

        top_score = ranked[0][1] if ranked else {"credibility": 0}
        confidence = (
            "high"
            if top_score["credibility"] >= 80
            else "medium"
            if top_score["credibility"] >= 55
            else "low"
        )
        cred = top_score.get("credibility", 0)
        qs = max(1, min(10, (cred // 10) or 1))

        return {
            "answer": answer,
            "category": "metrics",
            "time_sensitive": classification.time_sensitive,
            "quality_score": qs,
            "grounded": True,
            "confidence": confidence,
            "retried": False,
        }

    if classification.time_sensitive or classification.category == "temporal":
        docs = get_vectorstore().similarity_search(query, k=20)
        docs_sorted = sorted(
            docs,
            key=lambda d: d.metadata.get("date", ""),
            reverse=True,
        )
        result = resolve_temporal(query, docs_sorted)
        return {
            "answer": result["answer"],
            "category": classification.category,
            "time_sensitive": classification.time_sensitive,
            "conflicts": result.get("conflicts", False),
            "confidence": result.get("confidence", "medium"),
            "stale_companies": result.get("stale_companies", []),
        }

    if classification.requires_notes or classification.category == "relationship":
        retrieval_fn = search_with_notes
    else:
        retrieval_fn = search_deals

    context = retrieval_fn(query)

    tools = [
        Tool(
            name="search_deal_history",
            func=retrieval_fn,
            description=(
                "Search DVC's complete deal history, pipeline, contacts, and partner notes."
            ),
        )
    ]

    system_prompt = build_system_prompt(classification)
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            ("human", "{input}"),
            MessagesPlaceholder(variable_name="agent_scratchpad"),
        ]
    )

    agent = create_openai_functions_agent(llm=llm, tools=tools, prompt=prompt)
    executor = AgentExecutor(
        agent=agent, tools=tools, verbose=False, max_iterations=3
    )
    result = executor.invoke({"input": query})
    answer = result["output"]

    critique = critique_answer(query, context, answer)

    if verbose:
        print(f"\nCritique: {critique.critique}")
        print(f"Quality: {critique.quality_score}/10")
        print(f"Grounded: {critique.is_grounded}")
        print(f"Hallucination: {critique.hallucination_detected}")

    should_retry = (
        critique.quality_score < 7
        or critique.hallucination_detected
        or not critique.is_complete
    ) and bool(critique.refined_query and critique.refined_query.strip())

    if should_retry:
        refined = critique.refined_query.strip()

        if verbose:
            print(f"\nRetrying with refined query: {refined}")

        context2 = retrieval_fn(refined)

        def _tool_with_refined(_q: str, rq: str = refined) -> str:
            return retrieval_fn(rq)

        tools2 = [
            Tool(
                name="search_deal_history",
                func=_tool_with_refined,
                description="Search with refined query.",
            )
        ]

        agent2 = create_openai_functions_agent(llm=llm, tools=tools2, prompt=prompt)
        executor2 = AgentExecutor(
            agent=agent2, tools=tools2, verbose=False, max_iterations=3
        )
        result2 = executor2.invoke({"input": query})
        answer = result2["output"]

        critique2 = critique_answer(query, context2, answer)
        if verbose:
            print(f"Retry quality: {critique2.quality_score}/10")

        return {
            "answer": answer,
            "category": classification.category,
            "time_sensitive": classification.time_sensitive,
            "quality_score": critique2.quality_score,
            "grounded": critique2.is_grounded,
            "retried": True,
        }

    return {
        "answer": answer,
        "category": classification.category,
        "time_sensitive": classification.time_sensitive,
        "quality_score": critique.quality_score,
        "grounded": critique.is_grounded,
        "retried": False,
    }


if __name__ == "__main__":
    test_queries = [
        "Who is currently raising that we've been tracking?",
        "Which companies have gone cold or unresponsive?",
        "What deals were active in early 2024 that we haven't followed up on?",
        "Which companies showed interest but we lost contact with?",
        "What is the current status of Secufusion?",
    ]

    for q in test_queries:
        print(f"\nQ: {q}")
        result = ask(q, verbose=True)
        print(f"A: {result['answer']}")
        if result.get("stale_companies"):
            print(f"Stale: {result['stale_companies']}")
        print("-" * 60)
