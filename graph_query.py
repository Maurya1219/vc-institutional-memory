import os
from typing import Optional

from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field

from graph_builder import find_connections, find_shortest_path, get_graph, get_most_connected

load_dotenv()

llm = ChatOpenAI(
    model="gpt-4o",
    temperature=0,
    openai_api_key=os.getenv("OPENAI_API_KEY"),
)


class GraphQueryPlan(BaseModel):
    query_type: str = Field(
        description="connections, path, most_connected, or overlap"
    )
    entity_a: str = Field(description="Primary entity to query")
    entity_b: Optional[str] = Field(
        default=None, description="Second entity for path queries"
    )
    node_type: Optional[str] = Field(
        default=None, description="company or person filter"
    )


PLAN_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """You extract graph query parameters from natural language questions.

Query types:
- connections: find all connections to one entity
  "who do we know at Arnica", "how is BluSapphire connected to us"
- path: find the shortest path between two entities
  "how is Ravish connected to Arnica", "what's the path from DVC to this founder"
- most_connected: find the most connected nodes
  "who are our most connected contacts", "which companies have the most relationships"
- overlap: find shared connections between two companies
  "what do Arnica and BluSapphire have in common"

Extract the entity names exactly as they would appear in a CRM.""",
        ),
        ("human", "{query}"),
    ]
)

structured_llm = llm.with_structured_output(GraphQueryPlan)


def format_connections(result: dict) -> str:
    if not result["found"]:
        return f"No entity found matching '{result['entity']}' in the graph."

    lines = [f"**{result['entity']}** ({result['type']})"]

    people = [c for c in result["direct_connections"] if c["type"] == "person"]
    companies = [c for c in result["direct_connections"] if c["type"] == "company"]

    if people:
        lines.append(f"\nPeople connected ({len(people)}):")
        for p in people[:10]:
            lines.append(
                f"- {p['name']} ({p['relationship']}, strength: {p['weight']})"
            )

    if companies:
        lines.append(f"\nRelated companies ({len(companies)}):")
        for c in companies[:10]:
            lines.append(f"- {c['name']} ({c['relationship']})")

    if not people and not companies:
        lines.append("\nNo direct connections in the graph.")

    if result["second_hop"]:
        lines.append("\nReachable in 2 hops:")
        for hop in result["second_hop"][:8]:
            via = f" via {hop['via']}" if hop["via"] else ""
            lines.append(f"- {hop['name']}{via}")

    return "\n".join(lines)


def format_path(result: dict, source: str, target: str) -> str:
    if not result["found"]:
        return f"No path found between {source} and {target}."
    return (
        f"Path found in {result['hops']} hop(s):\n" f"{result['description']}"
    )


def format_most_connected(nodes: list) -> str:
    lines = ["Most connected entities:"]
    for i, node in enumerate(nodes, 1):
        lines.append(
            f"{i}. {node['name']} ({node['type']}) — {node['connections']} connections"
        )
    return "\n".join(lines)


def answer_graph_query(query: str) -> str:
    G = get_graph()

    try:
        chain = PLAN_PROMPT | structured_llm
        plan = chain.invoke({"query": query})
    except Exception as e:
        return f"Could not parse graph query: {e}"

    qt = (plan.query_type or "").lower().strip()

    if qt == "connections":
        result = find_connections(G, plan.entity_a)
        return format_connections(result)

    if qt == "path":
        if not plan.entity_b:
            return "Need two entities to find a path."
        result = find_shortest_path(G, plan.entity_a, plan.entity_b)
        return format_path(result, plan.entity_a, plan.entity_b)

    if qt == "most_connected":
        nodes = get_most_connected(G, node_type=plan.node_type, top_n=10)
        return format_most_connected(nodes)

    if qt == "overlap":
        if not plan.entity_b:
            return "Need two entities to find overlap."
        conn_a = find_connections(G, plan.entity_a)
        conn_b = find_connections(G, plan.entity_b)
        names_a = {c["name"] for c in conn_a.get("direct_connections", [])}
        names_b = {c["name"] for c in conn_b.get("direct_connections", [])}
        shared = names_a & names_b
        if not shared:
            return (
                f"No shared connections found between {plan.entity_a} and "
                f"{plan.entity_b}."
            )
        return (
            f"Shared connections between {plan.entity_a} and {plan.entity_b}:\n"
            + "\n".join(f"- {n}" for n in sorted(shared))
        )

    return "Unknown query type."
