from dataclasses import dataclass

from langchain_core.documents import Document


@dataclass
class SourcePlan:
    primary: list[str]
    secondary: list[str]
    skip: list[str]
    explanation: str


SOURCE_PLANS = {
    "metrics": SourcePlan(
        primary=["email", "sharepoint"],
        secondary=["affinity"],
        skip=["website"],
        explanation="Financial metrics most reliable from emails and documents",
    ),
    "relationship": SourcePlan(
        primary=["affinity_contact", "email"],
        secondary=["affinity"],
        skip=["website"],
        explanation="Relationship data lives in contacts and email threads",
    ),
    "deal_search": SourcePlan(
        primary=["affinity"],
        secondary=["sharepoint", "email"],
        skip=["website"],
        explanation="Deal history and notes live in Affinity",
    ),
    "temporal": SourcePlan(
        primary=["affinity"],
        secondary=["email"],
        skip=["website", "affinity_contact"],
        explanation="Current status requires latest Affinity notes and recent emails",
    ),
    "portfolio": SourcePlan(
        primary=["email", "affinity"],
        secondary=["sharepoint"],
        skip=["website"],
        explanation="Portfolio updates come from emails and Affinity",
    ),
    "pattern": SourcePlan(
        primary=["affinity"],
        secondary=[],
        skip=["email", "website", "sharepoint"],
        explanation="Pattern analysis only needs Affinity pipeline data",
    ),
    "graph": SourcePlan(
        primary=["affinity_contact", "affinity"],
        secondary=[],
        skip=["email", "website", "sharepoint"],
        explanation="Graph reasoning uses contact and company data",
    ),
    "factual": SourcePlan(
        primary=["website", "sharepoint"],
        secondary=["affinity"],
        skip=["email"],
        explanation="Firm facts and formal figures on website and documents",
    ),
    "general": SourcePlan(
        primary=["affinity", "email"],
        secondary=["sharepoint", "website"],
        skip=[],
        explanation="General queries search broadly across sources",
    ),
    "meeting_prep": SourcePlan(
        primary=["affinity", "email"],
        secondary=["sharepoint", "website"],
        skip=[],
        explanation="Meeting prep needs all available context",
    ),
    "counting": SourcePlan(
        primary=["affinity"],
        secondary=[],
        skip=["email", "website", "sharepoint"],
        explanation="Counting queries only need Affinity pipeline data",
    ),
}


def get_source_plan(category: str) -> SourcePlan:
    return SOURCE_PLANS.get(category, SOURCE_PLANS["general"])


def should_search_source(source: str, plan: SourcePlan, is_primary_pass: bool) -> bool:
    if source in plan.skip:
        return False
    if is_primary_pass:
        return source in plan.primary
    return source in plan.secondary


def filter_docs_by_source(
    docs: list[Document], allowed_sources: list[str]
) -> list[Document]:
    if not allowed_sources:
        return []
    return [
        d for d in docs if d.metadata.get("source", "") in allowed_sources
    ]
