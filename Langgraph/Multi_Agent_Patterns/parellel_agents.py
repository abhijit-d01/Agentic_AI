from typing import Annotated
from typing_extensions import TypedDict
from langchain_openai import ChatOpenAI
from langgraph.graph import END, START
from langgraph.graph.state import StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from langchain_core.tools import tool
from langchain_core.messages import BaseMessage
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END
from IPython.display import Image, display
import os


load_dotenv()

os.environ["LANGSMITH_API_KEY"]=os.getenv("LANGSMITH_API_KEY")

llm = ChatOpenAI(
        model="llama-3",
        base_url="http://127.0.0.1:8080/v1",
        api_key="local",
        temperature=0.1
    )

# Graph state
class State(TypedDict):
    topic: str
    characters: str
    settings: str
    premises: str
    story_intro: str

def make_parellel_graph():
    """Make a Story Writing Agent"""
    def generate_characters(state: State):
        """Generate character descriptions"""
        msg = llm.invoke(f"Create two character names and brief traits for a story about {state['topic']}")
        return {"characters": msg.content}

    def generate_setting(state: State):
        """Generate a story setting"""
        msg = llm.invoke(f"Describe a vivid setting for a story about {state['topic']}")
        return {"settings": msg.content}

    def generate_premise(state: State):
        """Generate a story premise"""
        msg = llm.invoke(f"Write a one-sentence plot premise for a story about {state['topic']}")
        return {"premises": msg.content}

    def combine_elements(state: State):
        """Combine characters, setting, and premise into an intro"""
        msg = llm.invoke(
            f"Write a short story introduction using these elements:\n"
            f"Characters: {state['characters']}\n"
            f"Setting: {state['settings']}\n"
            f"Premise: {state['premises']}"
        )
        return {"story_intro": msg.content}


    # Build the graph
    graph = StateGraph(State)
    graph.add_node("character", generate_characters)
    graph.add_node("setting", generate_setting)
    graph.add_node("premise", generate_premise)
    graph.add_node("combine", combine_elements)


    # Define edges (parallel execution from START)
    graph.add_edge(START, "character")
    graph.add_edge(START, "setting")
    graph.add_edge(START, "premise")
    graph.add_edge("character", "combine")
    graph.add_edge("setting", "combine")
    graph.add_edge("premise", "combine")
    graph.add_edge("combine", END)

    # Compile and run
    agent = graph.compile()
    return agent

agent=make_parellel_graph()