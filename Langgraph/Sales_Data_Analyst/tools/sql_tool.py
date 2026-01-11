import duckdb
import pandas as pd
from langchain.tools import tool

@tool
def query_sales_data(sql: str) -> pd.DataFrame:
    """
    Execute SQL query on sales data.
    Table name: sales
    """
    try:
        return duckdb.sql(sql).df()
    except Exception as e:
        return pd.DataFrame({"error": [str(e)]})
