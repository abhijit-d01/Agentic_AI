import uvicorn
import os
import traceback
from fastapi import FastAPI, Request
from dotenv import load_dotenv

# Import your specific modules
from src.graphs.graph_builder import GraphBuilder
from src.llms.qwenllm import QwenLLM 

# --- 1. Environment Setup ---
load_dotenv()

app = FastAPI()

# Ensure LangSmith/LangChain keys are set if you are using tracing
if os.getenv("LANGCHAIN_API_KEY"):
    os.environ["LANGSMITH_API_KEY"] = os.getenv("LANGCHAIN_API_KEY")

# --- 2. API Endpoints ---

@app.post("/blogs")
async def create_blogs(request: Request):
    """
    Endpoint to generate a blog post given a topic.
    """
    # 1. Parse Input
    try:
        data = await request.json()
        topic = data.get("topic", "").strip()
    except Exception:
        return {"error": "Invalid JSON body provided."}

    # Validation
    if not topic:
        return {"error": "Topic is required in the request body."}

    print(f"\n🚀 [API] Received Request: Generating blog for topic '{topic}'")

    try:
        # 2. Initialize LLM
        
        llm_wrapper = QwenLLM() 
        llm = llm_wrapper.get_llm()

        # 3. Build Graph
        print("--- [Graph] Building Graph ---")
        graph_builder = GraphBuilder(llm)
        graph = graph_builder.setup_graph(usecase="topic")
        
        # 4. Invoke Graph
        print("--- [Graph] Invoking... (This may take time) ---")
        # We pass the initial state dictionary
        state = graph.invoke({"topic": topic})
        
        # 5. Debug Output (Check your terminal!)
        print(f"🏁 [Graph] Execution Finished.")
        print(f"📊 [Debug] State Keys Returned: {list(state.keys())}")
        
        # Check if 'blog' data exists to verify success
        blog_data = state.get("blog", {})
        if blog_data:
            title = blog_data.get("title", "No Title")
            content_len = len(blog_data.get("content", ""))
            print(f"✅ [Success] Generated Title: '{title}'")
            print(f"✅ [Success] Content Length: {content_len} chars")
        else:
            print("⚠️ [Warning] 'blog' key is empty or missing in final state.")
            print(f"   Full State Dump: {state}")

        # Return the full state to the frontend
        return {"data": state}

    except Exception as e:
        # Detailed error logging
        print(f"❌ [Error] Critical failure in create_blogs: {e}")
        traceback.print_exc() # Prints the full stack trace to terminal
        
        # Return the error message to Streamlit so you see it in the UI
        return {"error": str(e)}

# --- 3. Entry Point ---
if __name__ == "__main__":
    print("✨ Starting FastAPI Server on port 8000...")
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)

    asdasd
