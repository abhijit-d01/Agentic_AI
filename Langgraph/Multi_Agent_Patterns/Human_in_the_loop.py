import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage
from langgraph.graph import StateGraph, START, MessagesState
from langgraph.prebuilt import ToolNode, tools_condition
# from langgraph.checkpoint.memory import MemorySaver

# 1. Load Environment Variables
load_dotenv()

# 2. Setup LLM
# Ensure your local server is running at this address.
# Adjust 'base_url' and 'api_key' if using a different provider.
llm = ChatOpenAI(
    model="llama-3",
    base_url="http://127.0.0.1:8080/v1",
    api_key="local", 
    temperature=0.1
)

def make_human_graph():
    """Make a human in the loop feedback agent"""
    # 3. Define Tools
    def multiply(a: int, b: int) -> int:
        """Multiply a and b."""
        return a * b

    def add(a: int, b: int) -> int:
        """Adds a and b."""
        return a + b

    def divide(a: int, b: int) -> float:
        """Divide a by b."""
        return a / b

    tools = [add, multiply, divide]

    # Bind tools to LLM
    llm_with_tools = llm.bind_tools(tools)

    # 4. Define Logic
    # System prompt to guide the assistant
    sys_msg = SystemMessage(content="You are a helpful assistant tasked with performing arithmetic on a set of inputs.")

    def assistant(state: MessagesState):
        """
        The assistant node invokes the LLM with the current state history.
        """
        return {"messages": [llm_with_tools.invoke([sys_msg] + state["messages"])]}

    # 5. Build the Graph
    builder = StateGraph(MessagesState)

    # Add Nodes
    builder.add_node("assistant", assistant)
    builder.add_node("tools", ToolNode(tools))

    # Add Edges
    # Start -> Assistant
    builder.add_edge(START, "assistant")

    # Conditional Edge: Assistant -> Tools OR End
    # If the LLM requests a tool call, route to "tools"
    # If the LLM generates text, route to END
    builder.add_conditional_edges(
        "assistant",
        tools_condition,
    )

    # Tools -> Assistant
    # After a tool runs, feed the result back to the assistant
    builder.add_edge("tools", "assistant")

    # # 6. Add Persistence
    # # MemorySaver allows the graph to pause, resume, and manage state
    # memory = MemorySaver()

    # 7. Compile and Export Graph
    # IMPORTANT: This 'graph' variable is what the LangGraph Studio UI imports.
    # We interrupt *before* the assistant node runs to allow human input/review.
    agent = builder.compile(
        interrupt_before=["assistant"],
    )
    return agent

agent=make_human_graph()