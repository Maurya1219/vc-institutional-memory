import os

import cohere
from dotenv import load_dotenv
from langchain_core.documents import Document

load_dotenv()

_api_key = os.getenv("COHERE_API_KEY")
co = cohere.Client(_api_key) if _api_key else None


def rerank(query: str, docs: list[Document], top_n: int = 5) -> list[Document]:
    if not docs:
        return []

    if not co:
        return docs[: min(top_n, len(docs))]

    passages = [doc.page_content[:512] for doc in docs]

    try:
        results = co.rerank(
            query=query,
            documents=passages,
            top_n=min(top_n, len(docs)),
            model="rerank-english-v3.0",
        )
        reranked = [docs[r.index] for r in results.results]
        return reranked

    except Exception as e:
        print(f"Rerank failed, falling back to original order: {e}")
        return docs[:top_n]
