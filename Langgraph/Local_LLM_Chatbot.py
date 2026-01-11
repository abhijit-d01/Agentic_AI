"""
Simple Streamlit Chat Agent using Local Llama 3B
Running on llama.cpp server (Mac)
"""

import streamlit as st
import requests
import json

# ============================================================================
# CONFIGURATION
# ============================================================================

# llama.cpp server endpoint (default when running llama-server)
LLAMA_SERVER_URL = "http://127.0.0.1:8080/completion"

# ============================================================================
# PAGE CONFIG
# ============================================================================

st.set_page_config(
    page_title="Local Llama Chat",
    page_icon="🦙",
    layout="centered"
)

# ============================================================================
# SESSION STATE
# ============================================================================

if 'messages' not in st.session_state:
    st.session_state.messages = []

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def call_local_llama(prompt: str, temperature: float = 0.7) -> str:
    """
    Call local llama.cpp server
    
    Args:
        prompt: User message
        temperature: Sampling temperature (0.0 to 1.0)
        
    Returns:
        Model response text
    """
    try:
        # Prepare request payload for llama.cpp
        payload = {
            "prompt": prompt,
            "temperature": temperature,
            "max_tokens": 500,
            "stop": ["User:", "Assistant:", "\n\n\n"],
            "stream": False
        }
        
        # Call the server
        response = requests.post(
            LLAMA_SERVER_URL,
            json=payload,
            timeout=30
        )
        
        if response.status_code == 200:
            result = response.json()
            return result.get("content", "").strip()
        else:
            return f"Error: Server returned {response.status_code}"
            
    except requests.exceptions.ConnectionError:
        return "❌ Cannot connect to llama.cpp server. Make sure it's running on localhost:8080"
    except Exception as e:
        return f"❌ Error: {str(e)}"


def build_prompt_with_history(user_message: str) -> str:
    """
    Build prompt with conversation history
    
    Args:
        user_message: Current user message
        
    Returns:
        Formatted prompt with history
    """
    # Start with system prompt
    prompt = "You are a helpful AI assistant. Be concise and friendly.\n\n"
    
    # Add conversation history (last 5 messages)
    history = st.session_state.messages[-10:] if len(st.session_state.messages) > 10 else st.session_state.messages
    
    for msg in history:
        role = "User" if msg["role"] == "user" else "Assistant"
        prompt += f"{role}: {msg['content']}\n"
    
    # Add current message
    prompt += f"User: {user_message}\nAssistant:"
    
    return prompt


# ============================================================================
# SIDEBAR
# ============================================================================

with st.sidebar:
    st.title("⚙️ Settings")
    
    # Server status check
    st.subheader("🔌 Server Status")
    try:
        test_response = requests.get("http://127.0.0.1:8080/health", timeout=2)
        if test_response.status_code == 200:
            st.success("✅ llama.cpp server running")
        else:
            st.warning("⚠️ Server responding but may have issues")
    except:
        st.error("❌ Server not reachable")
        st.info("Start with: `llama-server -m model.gguf`")
    
    st.divider()
    
    # Temperature control
    st.subheader("🌡️ Temperature")
    temperature = st.slider(
        "Controls randomness",
        min_value=0.0,
        max_value=1.0,
        value=0.7,
        step=0.1,
        help="Lower = more focused, Higher = more creative"
    )
    
    st.divider()
    
    # Clear chat button
    if st.button("🗑️ Clear Chat", use_container_width=True):
        st.session_state.messages = []
        st.rerun()
    
    st.divider()
    
    # Stats
    st.subheader("📊 Stats")
    st.metric("Messages", len(st.session_state.messages))
    
    st.divider()
    
    # Instructions
    with st.expander("ℹ️ How to Setup"):
        st.markdown("""
        **1. Install llama.cpp:**
        ```bash
        brew install llama.cpp
        ```
        
        **2. Download Llama 3B model:**
        ```bash
        # Example with Llama 3.2 3B
        wget https://huggingface.co/...
        ```
        
        **3. Start server:**
        ```bash
        llama-server -m /path/to/model.gguf -c 2048
        ```
        
        **4. Run this app:**
        ```bash
        streamlit run app_local.py
        ```
        """)

# ============================================================================
# MAIN CHAT INTERFACE
# ============================================================================

st.title("🦙 Local Llama Chat")
st.caption("Chat with Llama 3B running locally via llama.cpp")

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])

# Chat input
if prompt := st.chat_input("Type your message..."):
    
    # Add user message to history
    st.session_state.messages.append({
        "role": "user",
        "content": prompt
    })
    
    # Display user message
    with st.chat_message("user"):
        st.write(prompt)
    
    # Get AI response
    with st.chat_message("assistant"):
        with st.spinner("🤔 Thinking..."):
            
            # Build prompt with history
            full_prompt = build_prompt_with_history(prompt)
            
            # Call local Llama
            response = call_local_llama(full_prompt, temperature)
            
            # Display response
            st.write(response)
    
    # Add assistant response to history
    st.session_state.messages.append({
        "role": "assistant",
        "content": response
    })

# ============================================================================
# FOOTER
# ============================================================================

st.divider()
st.caption("Powered by Llama 3B running locally on llama.cpp")