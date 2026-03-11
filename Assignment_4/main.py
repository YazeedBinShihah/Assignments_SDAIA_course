import sys
import io
import os
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

# Assignment 4: Semantic Search on the Vision 2030 PDF
# Load → Embed → Store → Retrieve

from langchain_community.document_loaders import PyPDFLoader
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter


# --- 1. Load the PDF ---

pdf_path = os.path.join(os.path.dirname(__file__), "vision2030.pdf")
loader = PyPDFLoader(pdf_path)
docs = loader.load()

print(f"Loaded {len(docs)} pages from Vision 2030 PDF\n")
print(f"First page preview:\n{docs[0].page_content[:300]}\n")
print(f"Metadata: {docs[0].metadata}\n")


# --- 2. Split & Embed ---

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000, chunk_overlap=200, add_start_index=True
)
all_splits = text_splitter.split_documents(docs)
print(f"Split into {len(all_splits)} chunks\n")

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-mpnet-base-v2"
)

sample_vector = embeddings.embed_query(all_splits[0].page_content)
print(f"Embedding dimension: {len(sample_vector)}")
print(f"First 5 values: {sample_vector[:5]}\n")


# --- 3. Store in vector store ---

vector_store = InMemoryVectorStore(embeddings)
ids = vector_store.add_documents(documents=all_splits)
print(f"Stored {len(ids)} chunks in the vector store\n")


# --- 4. Retrieve ---

queries = [
    "What is the main goal of Saudi Vision 2030?",
    "How will the economy be diversified away from oil?",
    "What are the plans for tourism in Saudi Arabia?",
    "How is education being reformed?",
    "What is the role of the private sector?",
]

print("=" * 70)
print("SEMANTIC SEARCH RESULTS")
print("=" * 70)

for query in queries:
    print(f"\nQuery: {query}")
    print("-" * 50)

    results = vector_store.similarity_search(query, k=2)
    for i, doc in enumerate(results, 1):
        print(f"  [{i}] (page: {doc.metadata.get('page', '?')})")
        print(f"      {doc.page_content[:150]}...")

    scored_results = vector_store.similarity_search_with_score(query, k=1)
    doc, score = scored_results[0]
    print(f"  Best match score: {score:.4f}")

print()

# Retriever interface (batch query)
retriever = vector_store.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 1},
)

batch_results = retriever.batch([
    "What are the healthcare goals?",
    "How will housing be improved?",
])

print("=" * 70)
print("RETRIEVER BATCH RESULTS")
print("=" * 70)

for i, result_docs in enumerate(batch_results):
    print(f"\n  Query {i+1} → page: {result_docs[0].metadata.get('page', '?')}")
    print(f"    {result_docs[0].page_content[:150]}...")

print()
