import duckdb
import re
from typing import Dict, List

from agents.query_planner import plan_sql
from agents.writer import write_summary

AGG_FUNCS = ["sum(", "count(", "avg(", "min(", "max("]


def clean_sql(sql: str) -> str:
    return re.sub(r"```sql|```", "", sql, flags=re.IGNORECASE).strip()


def is_aggregate(sql: str) -> bool:
    return any(fn in sql.lower() for fn in AGG_FUNCS)


def fix_literal_columns(sql: str, columns: List[str]) -> str:
    for col in columns:
        sql = re.sub(
            rf"select\s+['\"]{col}['\"]",
            f"SELECT {col}",
            sql,
            flags=re.IGNORECASE,
        )
    return sql


def alias_aggregates(sql: str) -> str:
    patterns = {
        r"SUM\((.*?)\)(?!\s+AS)": r"SUM(\1) AS total_sales",
        r"COUNT\((.*?)\)(?!\s+AS)": r"COUNT(\1) AS total_count",
        r"AVG\((.*?)\)(?!\s+AS)": r"AVG(\1) AS avg_value",
        r"MIN\((.*?)\)(?!\s+AS)": r"MIN(\1) AS min_value",
        r"MAX\((.*?)\)(?!\s+AS)": r"MAX(\1) AS max_value",
    }
    for p, r in patterns.items():
        sql = re.sub(p, r, sql, flags=re.IGNORECASE)
    return sql


def extract_group_columns(sql: str) -> List[str]:
    m = re.search(r"select\s+(.*?)\s+from", sql, re.I | re.S)
    if not m:
        return []

    cols = []
    for part in m.group(1).split(","):
        if not any(fn in part.lower() for fn in AGG_FUNCS):
            cols.append(part.strip().split()[-1])
    return cols


def inject_group_by(sql: str, cols: List[str]) -> str:
    if "group by" in sql.lower() or not cols:
        return sql

    group = " GROUP BY " + ", ".join(cols)

    for kw in [" order by ", " limit "]:
        idx = sql.lower().find(kw)
        if idx != -1:
            return sql[:idx] + group + sql[idx:]

    return sql + group


def detect_chart_type(question: str) -> str:
    q = question.lower()
    if "pie" in q:
        return "pie"
    if "line" in q or "trend" in q or "over time" in q:
        return "line"
    return "bar"


def df_to_text(df, max_rows=20):
    if len(df) <= max_rows:
        return df.to_string(index=False)

    return (
        f"Showing {max_rows} of {len(df)} rows\n\n"
        + df.head(max_rows).to_string(index=False)
    )


def run_query_with_dataframe(question: str, dataframes: Dict):
    df = list(dataframes.values())[0]
    duckdb.register("sales", df)

    sql = plan_sql(question, df.columns.tolist())
    sql = clean_sql(sql)
    sql = fix_literal_columns(sql, df.columns.tolist())
    sql = alias_aggregates(sql)

    try:
        result_df = duckdb.sql(sql).df()
    except duckdb.BinderException:
        if is_aggregate(sql):
            sql = inject_group_by(sql, extract_group_columns(sql))
            result_df = duckdb.sql(sql).df()
        else:
            raise

    return {
        "answer": write_summary(question, df_to_text(result_df)),
        "data": result_df,
        "sql": sql,
        "chart_type": detect_chart_type(question),
    }
