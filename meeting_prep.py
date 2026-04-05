import os
from datetime import datetime

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

load_dotenv()

llm = ChatOpenAI(
    model="gpt-4o",
    temperature=0,
    openai_api_key=os.getenv("OPENAI_API_KEY"),
)


def run_meeting_prep(company_name: str, context_fn, graph_fn) -> dict:
    rag_context = context_fn(company_name)
    graph_context = graph_fn(company_name)

    prompt = f"""You are a chief of staff at Dallas Venture Capital preparing a partner for a meeting.

Company: {company_name}
Meeting date: {datetime.now().strftime('%B %d, %Y')}

Retrieved intelligence:
{rag_context}

Relationship graph:
{graph_context}

Write a complete meeting prep brief in this exact format:

## Meeting Prep: {company_name}
### One-line summary
[What this company does in one sentence]

### Current status
[Active / Cold / Raising / Re-engaging — with last activity date]

### Our relationship history
[Chronological list of key interactions with dates]

### People we know there
[Named individuals, their roles, how we know them]

### What partners have said
[Direct quotes or paraphrases from partner notes]

### Connections to our portfolio
[Any overlap or conflict with existing investments]

### Red flags or concerns
[Any concerns raised in notes — be honest]

### Recommended talking points
[3-5 specific things to raise in the meeting based on history]

### Suggested next step
[One clear action after this meeting]

Be specific. Use real names, dates, and quotes from the notes. Under 500 words total."""

    response = llm.invoke(prompt)

    return {
        "company": company_name,
        "brief": response.content,
        "generated_at": datetime.now().isoformat(),
    }
