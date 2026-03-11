import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

# Assignment 5: RAG Agent
# Data source: Lilian Weng's blog post about Prompt Engineering
# Different from Assignment 4 (Vision 2030 PDF)

import bs4
from langchain_community.document_loaders import WebBaseLoader
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.tools import tool
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langgraph.prebuilt import create_react_agent

from utils import model


# --- 1. Load from web ---

bs4_strainer = bs4.SoupStrainer(class_=("post-title", "post-header", "post-content"))
loader = WebBaseLoader(
    web_paths=("https://lilianweng.github.io/posts/2023-03-15-prompt-engineering/",),
    bs_kwargs={"parse_only": bs4_strainer},
)
docs = loader.load()
print(f"Loaded {len(docs)} document(s) from blog post")
print(f"Content length: {len(docs[0].page_content)} characters\n")


# --- 2. Split & Embed ---

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000, chunk_overlap=200, add_start_index=True
)
all_splits = text_splitter.split_documents(docs)
print(f"Split into {len(all_splits)} chunks\n")

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-mpnet-base-v2"
)


# --- 3. Store ---

vector_store = InMemoryVectorStore(embeddings)
vector_store.add_documents(documents=all_splits)
print("Stored all chunks in vector store\n")


# --- 4. RAG Agent with retrieve tool ---

@tool
def retrieve(query: str) -> str:
    """Search the prompt engineering blog post for relevant information."""
    retrieved_docs = vector_store.similarity_search(query, k=3)
    return "\n\n".join(
        f"Source: {doc.metadata}\nContent: {doc.page_content}"
        for doc in retrieved_docs
    )

agent = create_react_agent(
    model=model,
    tools=[retrieve],
    prompt=(
        "You are a helpful assistant that answers questions about prompt engineering techniques. "
        "You have a retrieve tool that searches a blog post for relevant info. "
        "Use it to answer questions. You can call it multiple times with different queries if needed."
    ),
)


# --- Demo ---

queries = [
    # Simple question
    "What is few-shot prompting?",
    # Multi-step question: agent needs to search twice
    "What is Chain of Thought prompting? And how does it compare to zero-shot prompting?",
]

for query in queries:
    print("=" * 70)
    print(f"Question: {query}")
    print("=" * 70)

    result = agent.invoke({"messages": [HumanMessage(content=query)]})
    print(f"\nAnswer: {result['messages'][-1].content}\n")
