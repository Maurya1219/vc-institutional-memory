import os

from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field

load_dotenv()

llm = ChatOpenAI(
    model="gpt-4o",
    temperature=0.3,
    openai_api_key=os.getenv("OPENAI_API_KEY"),
)


class ExpandedQueries(BaseModel):
    queries: list[str] = Field(
        description="3 alternative phrasings of the original query"
    )


EXPANSION_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """You are a query expansion engine for a venture capital firm's institutional memory system.

Given a query, generate 3 alternative phrasings that would retrieve different but relevant results.

Rules:
- Use different vocabulary and angles
- Think about how the information might be stored in CRM notes
- Include both broader and more specific versions
- For people queries, include their role and company variations
- For company queries, include sector and stage variations
- For status queries, include synonyms for the status

Examples:
Query: "who do we know at Google"
Expansions: ["Google contacts relationships introductions", "Alphabet employees our network", "Google executives meetings DVC"]

Query: "which companies are raising"
Expansions: ["fundraising round active pipeline", "seeking investment capital round", "term sheet closing funding"]

Query: "what did Ravish say"
Expansions: ["Ravish Ailinani notes comments feedback", "Ravish meeting discussion evaluation", "Ravish partner notes deals"]

Keep each expansion under 10 words.""",
        ),
        ("human", "Original query: {query}\n\nGenerate 3 alternative phrasings."),
    ]
)

structured_llm = llm.with_structured_output(ExpandedQueries)


def expand_query(query: str) -> list[str]:
    try:
        chain = EXPANSION_PROMPT | structured_llm
        result = chain.invoke({"query": query})
        all_queries = [query] + result.queries
        return all_queries
    except Exception as e:
        print(f"Query expansion failed, using original: {e}")
        return [query]
