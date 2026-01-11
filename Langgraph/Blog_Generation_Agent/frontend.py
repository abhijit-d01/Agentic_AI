import streamlit as st
import requests
import json

# --- Configuration ---
API_URL = "http://127.0.0.1:8000/blogs"

# --- Page Config ---
st.set_page_config(
    page_title="AI Blog Agent",
    page_icon="✍️",
    layout="centered"
)

# --- Header ---
st.title("✍️ Blog Generator")
st.markdown("Enter a topic below, and the agent will research and write a blog post for you.")

# --- Input Area ---
with st.form("blog_form"):
    topic = st.text_input("Blog Topic", placeholder="E.g., The Future of Quantum Computing")
    
    # You can add more inputs here if you update your API later (e.g., tone, length)
    submitted = st.form_submit_button("Generate Blog")

# --- Logic ---
if submitted and topic:
    # 1. Show a loading spinner while waiting for the FastAPI response
    with st.spinner(f"🚀 Researching and writing about '{topic}'... This may take a minute."):
        try:
            # 2. Send POST request to FastAPI
            response = requests.post(
                API_URL, 
                json={"topic": topic},
                timeout=300 # 5-minute timeout for long generations
            )
            
            # 3. Handle Response
            if response.status_code == 200:
                result = response.json()
                state_data = result.get("data", {})
                
                # --- Success Display ---
                st.success("Generation Complete!")
                
                # Create tabs for better UX
                tab1, tab2 = st.tabs(["📄 Rendered Blog", "🔍 Agent State (Debug)"])
                
                with tab1:
                    # --- REVISED LOGIC FOR NESTED DICTIONARY ---
                    # Your backend returns structure: {"blog": {"title": "...", "content": "..."}}
                    # We default to empty dict {} if "blog" key is missing
                    blog_data = state_data.get("blog", {})
                    
                    # 1. Extract specifically from the nested 'blog' dictionary
                    title = blog_data.get("title")
                    content = blog_data.get("content")

                    # 2. Fallback: If nested 'blog' dict is empty, check flat keys
                    # This adds robustness if you change your backend structure later
                    if not content:
                        content = state_data.get("generation") or \
                                  state_data.get("blog_post") or \
                                  state_data.get("final_output") or \
                                  state_data.get("answer")

                    # --- Display Logic ---
                    if title:
                        st.header(title)
                    
                    if content:
                        st.markdown(content)
                    else:
                        st.warning("⚠️ Content generated, but frontend couldn't parse it.")
                        st.info("The agent state might be structured differently than expected. Check the 'Agent State' tab.")
                        st.json(state_data) # Show raw data in main tab if parsing fails

                with tab2:
                    st.write("Full JSON State returned by the graph:")
                    st.json(state_data)

            else:
                st.error(f"Error {response.status_code}: {response.text}")

        except requests.exceptions.ConnectionError:
            st.error("❌ Could not connect to the backend. Is 'app.py' running on port 8000?")
        except Exception as e:
            st.error(f"An error occurred: {e}")

elif submitted and not topic:
    st.warning("Please enter a topic first.")