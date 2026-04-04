import os
from pathlib import Path

from dotenv import load_dotenv
from langchain_classic.agents import AgentExecutor, create_openai_functions_agent
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.tools import Tool
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

load_dotenv()

_ROOT = Path(__file__).resolve().parent
_VECTORSTORE_DIR = _ROOT / "dvc_vectorstore"
_FAISS_INDEX = _VECTORSTORE_DIR / "index.faiss"
_FAISS_PKL = _VECTORSTORE_DIR / "index.pkl"

if not _FAISS_INDEX.is_file() or not _FAISS_PKL.is_file():
    raise SystemExit(
        f"No FAISS index under {_VECTORSTORE_DIR} (need index.faiss and index.pkl).\n"
        "Build it first: python3 ingest_affinity.py"
    )

embeddings = OpenAIEmbeddings(openai_api_key=os.getenv("OPENAI_API_KEY"))
llm = ChatOpenAI(
    model="gpt-4o",
    temperature=0,
    openai_api_key=os.getenv("OPENAI_API_KEY"),
)

vectorstore = FAISS.load_local(
    str(_VECTORSTORE_DIR),
    embeddings,
    allow_dangerous_deserialization=True,
)


def ask(question: str):
    retriever = vectorstore.as_retriever(search_kwargs={"k": 8})

    def search_docs(q):
        docs = retriever.invoke(q)
        return "\n\n".join(
            [
                f"Company: {d.metadata.get('company')}\nDate: {d.metadata.get('date')}\n{d.page_content}"
                for d in docs
            ]
        )

    tools = [
        Tool(
            name="search_deal_history",
            func=search_docs,
            description="Search DVC's full deal history, pipeline, and partner notes.",
        )
    ]

    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """You are an AI assistant for Dallas Venture Capital with access to
the firm's complete deal history and relationship data.

When answering:
- Be specific — name companies, dates, and context
- If you find relevant deals, summarize what you know about them
- Flag if information seems incomplete or if you need more context
- Always cite which companies or notes your answer comes from""",
            ),
            ("human", "{input}"),
            MessagesPlaceholder(variable_name="agent_scratchpad"),
        ]
    )

    agent = create_openai_functions_agent(llm=llm, tools=tools, prompt=prompt)
    executor = AgentExecutor(
        agent=agent, tools=tools, verbose=False, max_iterations=3
    )
    result = executor.invoke({"input": question})
    return result["output"]


if __name__ == "__main__":
    questions = [
        "What companies have we tracked that are in cybersecurity?",
        "Have we seen any companies related to enterprise browsers?",
        "What are the most recently added companies to our pipeline?",
    ]
    for q in questions:
        print(f"\nQ: {q}")
        print(f"A: {ask(q)}")
        print("-" * 60)
