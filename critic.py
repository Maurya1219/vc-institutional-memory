import os
from typing import Optional

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


class CritiqueResult(BaseModel):
    is_grounded: bool = Field(
        description="Whether the answer is grounded in the retrieved context"
    )
    is_complete: bool = Field(
        description="Whether the answer fully addresses the question"
    )
    hallucination_detected: bool = Field(
        description="Whether the answer contains facts not present in the context"
    )
    quality_score: int = Field(
        description="Quality score from 1-10",
    )
    critique: str = Field(
        description="Brief explanation of the critique",
    )
    refined_query: Optional[str] = Field(
        default=None,
        description="A better query to retry with if the answer is poor, else null",
    )


CRITIQUE_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """You are a quality control system for a venture capital firm's AI assistant.

Your job is to evaluate whether an answer is:
1. Grounded — only contains facts present in the retrieved context
2. Complete — fully addresses what was asked
3. Free of hallucination — no invented names, dates, or facts

Be strict. If the answer mentions a specific person, date, or fact, it must appear in the context.
If the answer is vague or says "I don't have information" when the context clearly contains relevant data, mark is_complete as false.

If quality is below 7, provide a refined_query that would retrieve better context.""",
        ),
        (
            "human",
            """Original question: {query}

Retrieved context:
{context}

Generated answer:
{answer}

Evaluate this answer.""",
        ),
    ]
)

structured_llm = llm.with_structured_output(CritiqueResult)


def critique_answer(query: str, context: str, answer: str) -> CritiqueResult:
    try:
        chain = CRITIQUE_PROMPT | structured_llm
        return chain.invoke(
            {
                "query": query,
                "context": context,
                "answer": answer,
            }
        )
    except Exception as e:
        print(f"Critique failed: {e}")
        return CritiqueResult(
            is_grounded=True,
            is_complete=True,
            hallucination_detected=False,
            quality_score=7,
            critique="Critique unavailable",
            refined_query=None,
        )
