from langchain_openai import ChatOpenAI

def get_llm():
    return ChatOpenAI(
        model="llama-3",
        base_url="http://127.0.0.1:8080/v1",
        api_key="local",
        temperature=0.1
    )
