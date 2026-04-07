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
from source_router import filter_docs_by_source, get_source_plan
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


def routed_search(
    query: str,
    category: str,
    k_primary: int = 15,
    k_secondary: int = 10,
    rerank_top_n: int = 5,
) -> tuple[list[Document], str]:
    plan = get_source_plan(category)
    queries = expand_query(query)

    primary_docs: list[Document] = []
    for q in queries:
        raw = get_vectorstore().similarity_search(q, k=k_primary)
        filtered = filter_docs_by_source(raw, plan.primary)
        primary_docs.extend(filtered)

    primary_docs = dedupe_docs(primary_docs)

    if len(primary_docs) >= 5:
        all_docs = primary_docs
    elif plan.secondary:
        secondary_docs: list[Document] = []
        for q in queries:
            raw = get_vectorstore().similarity_search(q, k=k_secondary)
            filtered = filter_docs_by_source(raw, plan.secondary)
            secondary_docs.extend(filtered)
        secondary_docs = dedupe_docs(secondary_docs)
        all_docs = dedupe_docs(primary_docs + secondary_docs)
    else:
        all_docs = primary_docs

    all_docs = dedupe_docs(all_docs)
    all_docs = rerank(query, all_docs, top_n=rerank_top_n)
    return all_docs, plan.explanation


def format_docs(docs: list[Document]) -> str:
    return "\n\n".join(
        [
            f"Company: {d.metadata.get('company', '')}\n"
            f"Source: {d.metadata.get('source', '')}\n"
            f"Date: {d.metadata.get('date', '')}\n"
            f"{d.page_content}"
            for d in docs
        ]
    )


def build_system_prompt(classification, user_context: str = "") -> str:
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

    out = base + category_guidance.get(classification.category, "")
    if user_context.strip():
        out += "\n\n" + user_context.strip()
    return out


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


def ask(query: str, verbose: bool = False, user_context: str = "") -> dict:
    prep_company = _extract_meeting_prep_company(query)
    if prep_company:
        from meeting_prep import run_meeting_prep

        def _prep_rag(name: str) -> str:
            docs, _ = routed_search(
                f"everything about {name} notes history status",
                "meeting_prep",
                k_primary=20,
                k_secondary=10,
                rerank_top_n=8,
            )
            return format_docs(docs)

        result = run_meeting_prep(
            prep_company,
            _prep_rag,
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

    classification = classify_query(query)

    if verbose:
        print(f"\nClassified as: {classification.category}")
        print(f"Time sensitive: {classification.time_sensitive}")
        plan = get_source_plan(classification.category)
        print(
            f"Source plan: primary={plan.primary}, secondary={plan.secondary}, "
            f"skip={plan.skip}"
        )
        print(f"Reason: {plan.explanation}")

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

    if classification.category == "factual":
        docs, _ = routed_search(query, "factual")
        context = format_docs(docs)
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
        docs, _ = routed_search(
            query,
            "metrics",
            k_primary=20,
            k_secondary=10,
            rerank_top_n=10,
        )
        ranked = rank_docs_for_metrics(docs, company)
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
        docs, _ = routed_search(
            query,
            "temporal",
            k_primary=20,
            k_secondary=10,
            rerank_top_n=20,
        )
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

    if classification.category == "relationship":
        docs, _ = routed_search(query, "relationship")
        context = format_docs(docs)
        content_volume = sum(len(d.page_content) for d in docs)

        web_context = ""
        tavily_key = os.getenv("TAVILY_API_KEY")
        if (
            tavily_key
            and (
                content_volume < 500
                or "no additional notes" in context.lower()
            )
        ):
            try:
                from langchain_community.tools.tavily_search import TavilySearchResults

                search = TavilySearchResults(
                    max_results=3,
                    tavily_api_key=tavily_key,
                )
                web_results = search.invoke({"query": query})
                if web_results:
                    lines = []
                    for r in web_results:
                        if isinstance(r, dict):
                            snippet = r.get("content") or r.get("snippet") or ""
                            lines.append(f"- {snippet[:300]}")
                        else:
                            lines.append(f"- {str(r)[:300]}")
                    web_context = "\n\nWeb search results:\n" + "\n".join(lines)
            except Exception as e:
                print(f"Web search failed: {e}")

        graph_context = ""
        try:
            qstrip = query.strip()
            if re.search(r"(?i)^who\s+is\s+", qstrip):
                focus = re.sub(r"(?i)^who\s+is\s+", "", qstrip).strip()
                graph_q = (
                    f"who is {focus} and what companies and people are they "
                    f"connected to in our network?"
                )
            else:
                graph_q = query
            graph_result = answer_graph_query(graph_q)
            if graph_result and "No entity found" not in graph_result:
                graph_context = f"\n\nGraph connections:\n{graph_result}"
        except Exception as e:
            if verbose:
                print(f"Graph enrichment skipped: {e}")

        combined_context = context + graph_context + web_context

        person_prompt = f"""You are an analyst for Dallas Venture Capital.

Question: {query}

Internal CRM data:
{context}
{graph_context}
{web_context}

Answer comprehensively:
- Who is this person and what is their role
- Which company do they work for and what does that company do
- How does DVC know them — when was first contact, what interactions have happened
- Any relevant context about their background
- If web search provided info, incorporate it and note it came from public sources

Be specific. If their company is in our pipeline, mention that context."""

        response = llm.invoke(person_prompt)
        critique = critique_answer(query, combined_context, response.content)

        return {
            "answer": response.content,
            "category": "relationship",
            "time_sensitive": classification.time_sensitive,
            "quality_score": critique.quality_score,
            "grounded": critique.is_grounded,
            "retried": False,
        }

    cat = classification.category
    docs, plan_explanation = routed_search(query, cat)
    context = format_docs(docs)

    def retrieval_fn(q: str) -> str:
        d, _ = routed_search(q, cat)
        return format_docs(d)

    tools = [
        Tool(
            name="search_deal_history",
            func=retrieval_fn,
            description=(
                "Search DVC's institutional memory (Affinity, email, SharePoint, website). "
                f"{plan_explanation}"
            ),
        )
    ]

    system_prompt = build_system_prompt(classification, user_context)
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

        retry_docs, _ = routed_search(refined, cat)
        context2 = format_docs(retry_docs)

        def _tool_with_refined(_q: str, rq: str = refined) -> str:
            d, _ = routed_search(rq, cat)
            return format_docs(d)

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
