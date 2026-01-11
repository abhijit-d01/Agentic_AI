from typing import TypedDict, Optional
import pandas as pd

class AgentState(TypedDict):
    user_query: str
    sql_query: Optional[str]
    dataframe: Optional[pd.DataFrame]
    final_answer: Optional[str]
