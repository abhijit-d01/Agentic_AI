import os
import pprint
from typing import Literal, List
from typing_extensions import TypedDict
from dotenv import load_dotenv
import langchainhub as hub
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field
from langchain_core.output_parsers import StrOutputParser
from langchain_community.utilities import DuckDuckGoSearchAPIWrapper
from langgraph.graph import END, StateGraph, START
from langchain_core.documents import Document
# --- 1. Environment & Setup ---
load_dotenv()

# --- 2. Configuration ---
CHROMA_PATH = "./chroma_db_adaptive"
# Use host.docker.internal if running in LangGraph Studio (Docker), otherwise 127.0.0.1
# LLM_BASE_URL = "http://host.docker.internal:8080/v1" 
# EMBED_BASE_URL = "http://host.docker.internal:2304/v1"

# --- 3. Model & Components ---

llm = ChatOpenAI(
    model="qwen-2.5",
    base_url="http://127.0.0.1:8080/v1",
    api_key="local",
    temperature=0,
    max_tokens=2000,
    timeout=60
)

embeddings = OpenAIEmbeddings(
    base_url="http://127.0.0.1:2304/v1",
    api_key="local",
    model="Embeddinggemma-300M",
    check_embedding_ctx_length=False
)

# --- 4. Vector Store (Persisted) ---
def get_retriever():
    if os.path.exists(CHROMA_PATH):
        print("---LOADING EXISTING VECTOR STORE---")
        vectorstore = Chroma(
            persist_directory=CHROMA_PATH,
            embedding_function=embeddings,
            collection_name="adaptive_rag"
        )
    else:
        print("---BUILDING NEW VECTOR STORE---")
        urls = [
            "https://lilianweng.github.io/posts/2023-06-23-agent/",
            "https://lilianweng.github.io/posts/2023-03-15-prompt-engineering/",
            "https://lilianweng.github.io/posts/2023-10-25-adv-attack-llm/",
        ]
        docs = [WebBaseLoader(url).load() for url in urls]
        docs_list = [item for sublist in docs for item in sublist]
        text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
            chunk_size=256, chunk_overlap=20
        )
        doc_splits = text_splitter.split_documents(docs_list)
        
        vectorstore = Chroma.from_documents(
            documents=doc_splits,
            embedding=embeddings,
            collection_name="adaptive_rag",
            persist_directory=CHROMA_PATH
        )
    return vectorstore.as_retriever()

retriever = get_retriever()

# --- 5. Tools & Graders ---

# Router
class RouteQuery(BaseModel):
    datasource: Literal["vectorstore", "web_search"] = Field(
        ...,
        description="Given a user question choose to route it to web search or a vectorstore.",
    )

structured_llm_router = llm.with_structured_output(RouteQuery)
system_router = """You are an expert at routing a user question to a vectorstore or web search.
The vectorstore contains documents related to agents, prompt engineering, and adversarial attacks.
Use the vectorstore for questions on these topics. Otherwise, use web-search."""
route_prompt = ChatPromptTemplate.from_messages(
    [("system", system_router), ("human", "{question}")]
)
question_router = route_prompt | structured_llm_router

# Retrieval Grader
class GradeDocuments(BaseModel):
    binary_score: str = Field(description="Documents are relevant to the question, 'yes' or 'no'")

structured_llm_grader = llm.with_structured_output(GradeDocuments)
system_grader = """You are a grader assessing relevance of a retrieved document to a user question. \n 
    If the document contains keyword(s) or semantic meaning related to the user question, grade it as relevant. \n
    Give a binary score 'yes' or 'no' score to indicate whether the document is relevant to the question."""
grade_prompt = ChatPromptTemplate.from_messages(
    [("system", system_grader), ("human", "Retrieved document: \n\n {document} \n\n User question: {question}")]
)
retrieval_grader = grade_prompt | structured_llm_grader

# Generator
from langsmith import Client

client = Client()
prompt = client.pull_prompt("rlm/rag-prompt")
llm=llm
rag_chain = prompt | llm | StrOutputParser()

# Hallucination Grader
class GradeHallucinations(BaseModel):
    binary_score: str = Field(description="Answer is grounded in the facts, 'yes' or 'no'")

structured_llm_hallucination = llm.with_structured_output(GradeHallucinations)
system_hallucination = """You are a grader assessing whether an LLM generation is grounded in / supported by a set of retrieved facts. \n 
     Give a binary score 'yes' or 'no'. 'Yes' means that the answer is grounded in / supported by the set of facts."""
hallucination_prompt = ChatPromptTemplate.from_messages(
    [("system", system_hallucination), ("human", "Set of facts: \n\n {documents} \n\n LLM generation: {generation}")]
)
hallucination_grader = hallucination_prompt | structured_llm_hallucination

# Answer Grader
class GradeAnswer(BaseModel):
    binary_score: str = Field(description="Answer addresses the question, 'yes' or 'no'")

structured_llm_answer = llm.with_structured_output(GradeAnswer)
system_answer = """You are a grader assessing whether an answer addresses / resolves a question \n 
     Give a binary score 'yes' or 'no'. Yes' means that the answer resolves the question."""
answer_prompt = ChatPromptTemplate.from_messages(
    [("system", system_answer), ("human", "User question: \n\n {question} \n\n LLM generation: {generation}")]
)
answer_grader = answer_prompt | structured_llm_answer

# Question Rewriter
system_rewriter = """You a question re-writer that converts an input question to a better version that is optimized \n 
     for vectorstore retrieval. Look at the input and try to reason about the underlying semantic intent / meaning."""
re_write_prompt = ChatPromptTemplate.from_messages(
    [("system", system_rewriter), ("human", "Here is the initial question: \n\n {question} \n Formulate an improved question.")]
)
question_rewriter = re_write_prompt | llm | StrOutputParser()

# Web Search Wrapper
search_wrapper = DuckDuckGoSearchAPIWrapper(max_results=3)

# --- 6. Graph State & Nodes ---
def make_adaptive_graph():
    class GraphState(TypedDict):
        question: str
        generation: str
        documents: List[Document] # Corrected type

    def retrieve(state):
        print("---RETRIEVE---")
        question = state["question"]
        documents = retriever.invoke(question)
        return {"documents": documents, "question": question}

    def generate(state):
        print("---GENERATE---")
        question = state["question"]
        documents = state["documents"]
        
        # Format docs for the chain
        docs_txt = "\n\n".join([d.page_content for d in documents])
        
        generation = rag_chain.invoke({"context": docs_txt, "question": question})
        return {"documents": documents, "question": question, "generation": generation}

    def grade_documents(state):
        print("---CHECK DOCUMENT RELEVANCE---")
        question = state["question"]
        documents = state["documents"]

        filtered_docs = []
        for d in documents:
            try:
                score = retrieval_grader.invoke({"question": question, "document": d.page_content})
                grade = score.binary_score
            except:
                grade = "no" # Fallback

            if grade.lower() == "yes":
                print("---GRADE: RELEVANT---")
                filtered_docs.append(d)
            else:
                print("---GRADE: NOT RELEVANT---")
                continue
                
        return {"documents": filtered_docs, "question": question}

    def transform_query(state):
        print("---TRANSFORM QUERY---")
        question = state["question"]
        documents = state["documents"]
        better_question = question_rewriter.invoke({"question": question})
        return {"documents": documents, "question": better_question}

    def web_search(state):
        print("---WEB SEARCH---")
        question = state["question"]

        # Fix: Use wrapper directly to get list of dicts
        try:
            results = search_wrapper.results(question, max_results=3)
            web_content = "\n".join([r.get("snippet", r.get("body", "")) for r in results])
            web_doc = Document(page_content=web_content, metadata={"source": "web_search"})
            # Return as a list containing the doc
            return {"documents": [web_doc], "question": question}
        except Exception as e:
            print(f"Web search error: {e}")
            return {"documents": [], "question": question}

    # --- 7. Edges ---

    def route_question(state):
        print("---ROUTE QUESTION---")
        question = state["question"]
        try:
            source = question_router.invoke({"question": question})
            if source.datasource == "web_search":
                print("---ROUTE TO WEB SEARCH---")
                return "web_search"
            elif source.datasource == "vectorstore":
                print("---ROUTE TO RAG---")
                return "vectorstore"
        except:
            return "vectorstore" # Default fallback

    def decide_to_generate(state):
        print("---ASSESS GRADED DOCUMENTS---")
        filtered_documents = state["documents"]

        if not filtered_documents:
            # FIX: Break loop by forcing web search
            print("---DECISION: ALL DOCUMENTS FILTERED -> WEB SEARCH---")
            return "web_search"
        else:
            print("---DECISION: GENERATE---")
            return "generate"

    def grade_generation_v_documents_and_question(state):
        print("---CHECK HALLUCINATIONS---")
        question = state["question"]
        documents = state["documents"]
        generation = state["generation"]
        
        # Format docs for grader
        docs_txt = "\n".join([d.page_content for d in documents])

        try:
            score = hallucination_grader.invoke({"documents": docs_txt, "generation": generation})
            grade = score.binary_score
        except:
            grade = "yes" # Benefit of doubt on local LLM failure

        if grade.lower() == "yes":
            print("---DECISION: GENERATION IS GROUNDED---")
            print("---GRADE GENERATION vs QUESTION---")
            try:
                score = answer_grader.invoke({"question": question, "generation": generation})
                grade = score.binary_score
            except:
                grade = "yes"
                
            if grade.lower() == "yes":
                print("---DECISION: GENERATION ADDRESSES QUESTION---")
                return "useful"
            else:
                print("---DECISION: GENERATION DOES NOT ADDRESS QUESTION---")
                return "not useful"
        else:
            print("---DECISION: HALLUCINATION DETECTED -> RETRY---")
            return "not supported"

    # --- 8. Graph Construction ---

    workflow = StateGraph(GraphState)

    workflow.add_node("web_search", web_search)
    workflow.add_node("retrieve", retrieve)
    workflow.add_node("grade_documents", grade_documents)
    workflow.add_node("generate", generate)
    workflow.add_node("transform_query", transform_query)

    workflow.add_conditional_edges(
    START,
    route_question,
    {"web_search": "web_search", "vectorstore": "retrieve"},
    )
    workflow.add_edge("web_search", "generate")
    workflow.add_edge("retrieve", "grade_documents")

    # UPDATED CONDITIONAL EDGES
    workflow.add_conditional_edges(
        "grade_documents",
        decide_to_generate,
        {
            "web_search": "web_search", # FIX: Redirect to web search
            "generate": "generate",
        },
    )

    workflow.add_edge("transform_query", "retrieve")
    workflow.add_conditional_edges(
        "generate",
        grade_generation_v_documents_and_question,
        {
            "not supported": "generate",
            "useful": END,
            "not useful": "transform_query",
        },
    )

    # Compile for LangGraph Studio
    agent = workflow.compile()
    return agent

agent=make_adaptive_graph()