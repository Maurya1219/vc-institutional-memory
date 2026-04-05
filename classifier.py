import os
from typing import Literal

from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field

load_dotenv()

llm = ChatOpenAI(
    model="gpt-4o",
    temperature=0,
    openai_api_key=os.getenv("OPENAI_API_KEY"),
)


class QueryClassification(BaseModel):
    category: Literal[
        "deal_search",
        "relationship",
        "temporal",
        "portfolio",
        "counting",
        "factual",
        "graph",
        "pattern",
        "general",
    ] = Field(description="The category of the query")
    reasoning: str = Field(description="Why this category was chosen")
    requires_notes: bool = Field(
        description="Whether partner notes are important for this query"
    )
    time_sensitive: bool = Field(
        description="Whether this query involves current status or time"
    )


CLASSIFICATION_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """You are a query classifier for a venture capital firm's institutional memory system.

Classify each query into one of these categories:

- deal_search: Looking for specific companies, deals, sectors, or investments
  Examples: "have we seen any fintech companies", "what do we know about Secufusion", "show me enterprise SaaS deals"

- relationship: Questions about people, founders, network connections, introductions
  Examples: "who knows this CEO", "have we met this founder before", "who introduced us to X"

- temporal: Questions about current status, what's happening now, recent activity
  Examples: "who is currently raising", "what deals are active", "what's changed recently"
  These require checking if older information has been superseded by newer information.

- portfolio: Questions about existing investments, portfolio companies, follow-ons (not raw financial metrics)
  Examples: "how are our portfolio companies doing", "which portfolio company is in this space"
  If the user asks for revenue, ARR, valuation, funding amount, or headcount, use factual instead.

- counting: Simple aggregation or counting from records (not full-pipeline analytics)
  Examples: "how many deals did we see in 2024", "how many companies in the India pipeline", "average check size"
  Prefer pattern for sector mix, pipeline growth, source effectiveness, or cold-rate style questions.

- factual: Specific data points about a company — revenue, ARR, funding amount, valuation, headcount
  Examples: "what is Arnica's revenue", "how much has X raised", "what is Y's valuation"
  These require precise data. If not available, say so clearly — never estimate.

- graph: Questions about connections, paths, network relationships between entities
  Examples: "how is Ravish connected to Arnica", "what do these two companies have in common",
  "who are our most connected contacts", "what's the path from DVC to this founder"
  Prefer graph over relationship when the user wants paths, hops, overlap between two companies, or "most connected" hubs.

- pattern: Analytical questions about trends, distributions, pipeline health, deal-source effectiveness
  Examples: "what sectors do we invest in most", "how has our pipeline grown",
  "which deal sources work best", "what percentage of deals go cold",
  "show me pipeline analytics", "what are our investment patterns"

- general: Anything else, general questions about the firm

Also determine:
- requires_notes: true if partner meeting notes or interaction history would help answer this
- time_sensitive: true if the answer depends on current status vs historical records""",
        ),
        ("human", "Classify this query: {query}"),
    ]
)

structured_llm = llm.with_structured_output(QueryClassification)


def classify_query(query: str) -> QueryClassification:
    chain = CLASSIFICATION_PROMPT | structured_llm
    return chain.invoke({"query": query})
