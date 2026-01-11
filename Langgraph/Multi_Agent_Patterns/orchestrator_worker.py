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
from pydantic import BaseModel, Field
from typing import List
import operator
from langchain_core.messages import SystemMessage, HumanMessage
import os
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import PromptTemplate
from langgraph.constants import Send


load_dotenv()

os.environ["LANGSMITH_API_KEY"]=os.getenv("LANGSMITH_API_KEY")

llm = ChatOpenAI(
        model="llama-3",
        base_url="http://127.0.0.1:8080/v1",
        api_key="local",
        temperature=0.1
    )

class Section(BaseModel):
    name: str = Field(description="Name for this Section of the report")
    description: str = Field(description="Brief overview of the main topics") # <--- FIXED

class Sections(BaseModel):
    sections:List[Section]=Field(
        descriptions="Sections of the Report"
    )


# planner=llm.with_structured_output(Sections)



# Graph state
class State(TypedDict):
    topic: str  # Report topic
    sections: list[Section]  # List of report sections
    completed_sections: Annotated[
        list, operator.add
    ]  # All workers write to this key in parallel
    final_report: str  # Final report

# Worker state
class WorkerState(TypedDict):
    section: Section
    completed_sections: Annotated[list, operator.add]

def make_orchestrator_graph():
# Nodes
    def orchestrator(state: State):
        """Generates the plan using a strict One-Shot Example prompt."""
        print(f"Orchestrator planning for topic: {state['topic']}")
        
        # 1. Define the Parser (we still use this to validate the output)
        parser = PydanticOutputParser(pydantic_object=Sections)
        
        # 2. Define a clear, manual prompt with an EXAMPLE
        # We generally ignore parser.get_format_instructions() for local models
        # because it is too complex.
        json_structure = """
        {
          "sections": [
             {
               "name": "Section Name",
               "description": "Content description here"
             }
          ]
        }
        """

        prompt = PromptTemplate(
            template="""You are an expert technical writer. 
            Generate a comprehensive plan for a report on the following topic.
            
            Topic: {topic}
            
            Return the output as a SINGLE VALID JSON object. Do not include any explanation or markdown code blocks (like ```json).
            The format must match this structure exactly:
            {json_structure}
            """,
            input_variables=["topic"],
            partial_variables={"json_structure": json_structure},
        )

        chain = prompt | llm | parser
        
        try:
            plan = chain.invoke({"topic": state["topic"]})
            return {"sections": plan.sections}
        except Exception as e:
            print(f"Orchestrator parsing error: {e}")
            # Fallback: Create a simple default plan so the graph doesn't crash
            return {
                "sections": [
                    Section(name="Introduction", description=f"Introduction to {state['topic']}"),
                    Section(name="Deep Dive", description=f"Detailed analysis of {state['topic']}"),
                    Section(name="Conclusion", description="Summary of key findings")
                ]
            }

    def llm_call(state: WorkerState):
        """Worker writes a section of the report"""

        # Generate section
        section = llm.invoke(
            [
                SystemMessage(
                    content="Write a report section following the provided name and description. Include no preamble for each section. Use markdown formatting."
                ),
                HumanMessage(
                    content=f"Here is the section name: {state['section'].name} and description: {state['section'].description}"
                ),
            ]
        )

        # Write the updated section to completed sections
        return {"completed_sections": [section.content]}

    # Conditional edge function to create llm_call workers that each write a section of the report
    def assign_workers(state: State):
        """Assign a worker to each section in the plan"""

        # Kick off section writing in parallel via Send() API
        return [Send("llm_call", {"section": s}) for s in state["sections"]]

    def synthesizer(state: State):
        """Synthesize full report from sections"""

        # List of completed sections
        completed_sections = state["completed_sections"]

        # Format completed section to str to use as context for final sections
        completed_report_sections = "\n\n---\n\n".join(completed_sections)

        return {"final_report": completed_report_sections}
    # Build workflow


    from langgraph.graph import StateGraph, START, END
    from IPython.display import Image, display
    orchestrator_worker_builder = StateGraph(State)

    # Add the nodes
    orchestrator_worker_builder.add_node("orchestrator", orchestrator)
    orchestrator_worker_builder.add_node("llm_call", llm_call)
    orchestrator_worker_builder.add_node("synthesizer", synthesizer)

    # Add edges to connect nodes
    orchestrator_worker_builder.add_edge(START, "orchestrator")
    orchestrator_worker_builder.add_conditional_edges(
        "orchestrator", assign_workers, ["llm_call"]
    )
    orchestrator_worker_builder.add_edge("llm_call", "synthesizer")
    orchestrator_worker_builder.add_edge("synthesizer", END)

    # Compile the workflow
    agent=orchestrator_worker_builder.compile()
    return agent

# Invoke
agent=make_orchestrator_graph()