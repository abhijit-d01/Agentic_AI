import os
from dotenv import load_dotenv
from typing import Literal, Optional
from typing_extensions import TypedDict
from pydantic import BaseModel, Field

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import StateGraph, START, END

load_dotenv()

# 1. Setup Local LLM (Corrected Base URL for llama.cpp)
llm = ChatOpenAI(
    model="llama-3",
    base_url="http://127.0.0.1:8080/v1", # Ensure the port and /v1 are included
    api_key="local",
    streaming=False,
    temperature=0.1
)

# 2. Robust Schema (Catches extra fields local models often hallucinate)
class Route(BaseModel):
    step: Literal["poem", "story", "joke"] = Field(
        default="joke", 
        description="The next node to visit"
    )
    # These fields catch 'extra' data from llama.cpp to prevent ValidationErrors
    type: Optional[str] = None
    content: Optional[str] = None

# 3. Use json_mode (The most stable method for llama.cpp)
router = llm.with_structured_output(Route, method="json_mode")

# 4. State Definition
class State(TypedDict):
    input: str
    decision: str
    output: str

def make_router_graph():
    # --- NODES ---
    
    def llm_call_1(state: State):
        result = llm.invoke(f"Write a short story about: {state['input']}")
        return {"output": result.content}

    def llm_call_2(state: State):
        result = llm.invoke(f"Tell a funny joke about: {state['input']}")
        return {"output": result.content}

    def llm_call_3(state: State):
        result = llm.invoke(f"Write a short poem about: {state['input']}")
        return {"output": result.content}

    def llm_call_router(state: State):
        prompt = [
            SystemMessage(content="You are a routing agent. Output ONLY a JSON object with the key 'step' set to 'poem', 'story', or 'joke'."),
            HumanMessage(content=state["input"])
        ]
        try:
            decision = router.invoke(prompt)
            
            # Logic Fallback: check both 'step' and 'type' fields
            final_choice = decision.step
            if decision.type in ["poem", "story", "joke"]:
                final_choice = decision.type
                
            return {"decision": final_choice}
            
        except Exception as e:
            print(f"Router fallback due to error: {e}")
            return {"decision": "joke"}

    # --- CONDITIONAL EDGE LOGIC ---
    def route_decision(state: State):
        return state["decision"]

    # --- BUILD WORKFLOW ---
    router_builder = StateGraph(State)

    router_builder.add_node("llm_call_1", llm_call_1)
    router_builder.add_node("llm_call_2", llm_call_2)
    router_builder.add_node("llm_call_3", llm_call_3)
    router_builder.add_node("llm_call_router", llm_call_router)

    router_builder.add_edge(START, "llm_call_router")
    
    router_builder.add_conditional_edges(
        "llm_call_router",
        route_decision,
        {
            "story": "llm_call_1",
            "joke": "llm_call_2",
            "poem": "llm_call_3",
        },
    )

    router_builder.add_edge("llm_call_1", END)
    router_builder.add_edge("llm_call_2", END)
    router_builder.add_edge("llm_call_3", END)

    return router_builder.compile()

agent = make_router_graph()
