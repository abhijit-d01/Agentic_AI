import streamlit as st
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage, ToolMessage
from langgraph.graph import StateGraph, START, MessagesState
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.checkpoint.memory import MemorySaver

# 1. Load Config & Setup
load_dotenv()
st.set_page_config(page_title="Human-in-the-Loop Calculator", layout="centered")

# Initialize LLM
@st.cache_resource
def get_llm():
    return ChatOpenAI(
        model="llama-3",
        base_url="http://127.0.0.1:8080/v1",
        api_key="local",
        temperature=0.1
    )

# 2. Define Tools (With Docstrings for LangChain)
def multiply(a: int, b: int) -> int: 
    """Multiply a and b."""
    return a * b

def add(a: int, b: int) -> int: 
    """Add a and b."""
    return a + b

def divide(a: int, b: int) -> float: 
    """Divide a by b."""
    return a / b

tools = [add, multiply, divide]

@st.cache_resource
def build_graph():
    """Builds the LangGraph agent once and caches it."""
    llm = get_llm()
    llm_with_tools = llm.bind_tools(tools)
    
    sys_msg = SystemMessage(content="You are a helpful assistant tasked with performing arithmetic on a set of inputs.")

    def assistant(state: MessagesState):
        return {"messages": [llm_with_tools.invoke([sys_msg] + state["messages"])]}

    builder = StateGraph(MessagesState)
    builder.add_node("assistant", assistant)
    builder.add_node("tools", ToolNode(tools))
    
    builder.add_edge(START, "assistant")
    builder.add_conditional_edges("assistant", tools_condition)
    builder.add_edge("tools", "assistant")

    memory = MemorySaver()
    
    # Interrupt BEFORE the tools run
    return builder.compile(checkpointer=memory, interrupt_before=["tools"])

graph = build_graph()

# 3. Session State
if "thread_id" not in st.session_state:
    st.session_state.thread_id = "streamlit_user_1"

config = {"configurable": {"thread_id": st.session_state.thread_id}}

# 4. Helper to Stream Graph Updates
def run_graph(input_data=None):
    """Runs the graph and updates the UI with the stream."""
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        
        # input_data is None when resuming from an interrupt
        for event in graph.stream(input_data, config, stream_mode="values"):
            if "messages" in event:
                last_msg = event["messages"][-1]
                
                # Render AI responses
                if isinstance(last_msg, AIMessage) and last_msg.content:
                    message_placeholder.markdown(last_msg.content)
                # Render Tool outputs (if any)
                elif isinstance(last_msg, ToolMessage):
                    message_placeholder.markdown(f"**Tool Calculation:** `{last_msg.content}`")

# 5. UI Layout
st.title("🧮 Human-in-the-Loop Calculator")
st.caption("This agent pauses before calculating. You can **Approve** the math or **Correct** it.")

# --- DISPLAY CHAT HISTORY ---
snapshot = graph.get_state(config)
current_messages = snapshot.values.get("messages", [])

for msg in current_messages:
    if isinstance(msg, HumanMessage):
        st.chat_message("user").write(msg.content)
    elif isinstance(msg, AIMessage):
        st.chat_message("assistant").write(msg.content)
        # Show proposed tool calls in an expander
        if msg.tool_calls:
            with st.expander("🛠️ Proposed Calculation", expanded=False):
                st.code(msg.tool_calls)
    elif isinstance(msg, ToolMessage):
        with st.chat_message("assistant"):
            st.markdown(f"**Result:** `{msg.content}`")

# --- HANDLING INTERRUPTS ---
if snapshot.next:
    # Graph is paused at 'tools' node
    st.divider()
    st.warning("⏸️ **Execution Paused**")
    st.info("The agent wants to calculate. Please choose an action:")

    col1, col2 = st.columns([1, 1])
    
    with col1:
        if st.button("✅ Approve & Run", type="primary", use_container_width=True):
            run_graph(None) # Resume execution
            st.rerun()

    with col2:
        # Correction logic
        with st.form("correction_form", clear_on_submit=True):
            user_correction = st.text_input("Correction", placeholder="e.g. 'No, multiply by 10'")
            submitted = st.form_submit_button("Update Instruction")
            
            if submitted and user_correction:
                # Update state AS THE 'TOOLS' NODE.
                # This tells the graph: "Skip the tool, here is the result/feedback instead."
                # The edge tools->assistant will then trigger the LLM to process this feedback.
                graph.update_state(
                    config, 
                    {"messages": [HumanMessage(content=user_correction)]}, 
                    as_node="tools"
                )
                run_graph(None)
                st.rerun()

# --- STANDARD CHAT INPUT ---
else:
    if prompt := st.chat_input("Calculate something..."):
        st.chat_message("user").write(prompt)
        run_graph({"messages": [HumanMessage(content=prompt)]})
        st.rerun()