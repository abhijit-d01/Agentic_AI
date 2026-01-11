from langchain_core.prompts import ChatPromptTemplate
from llm import get_llm


def plan_sql(question: str, columns: list[str]) -> str:
    """
    Generates DuckDB SQL using ONLY provided columns.
    Prevents quoting column names as string literals.
    """

    system_prompt = f"""
You are a senior data analyst writing SQL for DuckDB.

CRITICAL RULES (DO NOT VIOLATE):
- Use ONLY these column names exactly as provided: {columns}
- Column names are identifiers, NOT strings
- NEVER wrap column names in single quotes or double quotes
- NEVER use string literals in GROUP BY
- If the user mentions a column (e.g. Size), you MUST use that exact column
- Use GROUP BY for aggregation queries
- Output ONLY valid SQL (no markdown, no explanation)
- Table name is: sales

EXAMPLE (CORRECT):
SELECT Size, SUM(Amount) FROM sales GROUP BY Size;

EXAMPLE (WRONG):
SELECT 'Size', SUM(Amount) FROM sales GROUP BY 'Size';
"""

    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", question)
    ])

    llm = get_llm()
    response = llm.invoke(prompt.format_messages())

    return response.content.strip()
