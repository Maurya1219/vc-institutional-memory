import os
from pathlib import Path

from dotenv import load_dotenv
from langchain_classic.agents import AgentExecutor, create_openai_functions_agent
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.tools import Tool
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

from classifier import classify_query
from temporal_resolver import resolve_temporal

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

_ROOT = Path(__file__).resolve().parent
_VECTORSTORE_DIR = _ROOT / "dvc_vectorstore"
_FAISS_INDEX = _VECTORSTORE_DIR / "index.faiss"
_FAISS_PKL = _VECTORSTORE_DIR / "index.pkl"

if not _FAISS_INDEX.is_file() or not _FAISS_PKL.is_file():
    raise SystemExit(
        f"No FAISS index under {_VECTORSTORE_DIR} (need index.faiss and index.pkl).\n"
        "Build it first: python3 ingest_affinity.py"
    )

embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
llm = ChatOpenAI(
    model="gpt-4o",
    temperature=0,
    openai_api_key=OPENAI_API_KEY,
)

vectorstore = FAISS.load_local(
    str(_VECTORSTORE_DIR),
    embeddings,
    allow_dangerous_deserialization=True,
)


def search_deals(query: str, k: int = 8) -> str:
    docs = vectorstore.similarity_search(query, k=k)
    return "\n\n".join(
        [
            f"Company: {d.metadata.get('company')}\n"
            f"Date: {d.metadata.get('date')}\n"
            f"Source: {d.metadata.get('source')}\n"
            f"{d.page_content}"
            for d in docs
        ]
    )


def search_recent(query: str) -> str:
    docs = vectorstore.similarity_search(query, k=20)
    docs_sorted = sorted(
        docs,
        key=lambda d: d.metadata.get("date", ""),
        reverse=True,
    )
    return "\n\n".join(
        [
            f"Company: {d.metadata.get('company')}\n"
            f"Date added: {d.metadata.get('date')}\n"
            f"{d.page_content}"
            for d in docs_sorted[:8]
        ]
    )


def search_with_notes(query: str) -> str:
    docs = vectorstore.similarity_search(query, k=10)
    results = []
    for d in docs:
        pc = d.page_content.lower()
        if "partner notes" in pc and "no partner notes" not in pc:
            results.append(
                f"Company: {d.metadata.get('company')}\n"
                f"Date: {d.metadata.get('date')}\n"
                f"{d.page_content}"
            )
    if not results:
        return search_deals(query)
    return "\n\n".join(results[:6])


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
        "general": "",
    }

    return base + category_guidance.get(classification.category, "")


def ask(query: str, verbose: bool = False) -> dict:
    classification = classify_query(query)

    if verbose:
        print(f"\nClassified as: {classification.category}")
        print(f"Reasoning: {classification.reasoning}")
        print(f"Time sensitive: {classification.time_sensitive}")

    if classification.time_sensitive or classification.category == "temporal":
        docs = vectorstore.similarity_search(query, k=20)
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

    tools = [
        Tool(
            name="search_deal_history",
            func=retrieval_fn,
            description="Search DVC's complete deal history, pipeline companies, and partner notes.",
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

    return {
        "answer": result["output"],
        "category": classification.category,
        "time_sensitive": classification.time_sensitive,
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
