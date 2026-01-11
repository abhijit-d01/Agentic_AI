from langchain_core.prompts import ChatPromptTemplate
from llm import get_llm


def write_summary(question: str, result_text: str) -> str:
    """
    Summarizes SQL output without hallucinating columns.
    """

    system_prompt = """
You are a data analyst explaining query results.

RULES:
- ONLY refer to columns that appear in the result
- DO NOT infer or rename columns
- DO NOT mention columns not present in the data
- Be concise and factual
"""

    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", f"Question: {question}\n\nResult:\n{result_text}")
    ])

    llm = get_llm()
    response = llm.invoke(prompt.format_messages())

    return response.content.strip()
