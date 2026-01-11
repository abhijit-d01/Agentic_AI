import os
from langchain_openai import ChatOpenAI

class QwenLLM:
    def get_llm(self):  
        llm = ChatOpenAI(
            model="qwen-2.5",
            base_url="http://127.0.0.1:8080/v1",
            api_key="local",
            temperature=0.1
            # max_tokens=2000,
            # timeout=300
        )
        return llm