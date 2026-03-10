import os
import requests
from dotenv import load_dotenv
from openai import OpenAI
from duckduckgo_search import DDGS
from bs4 import BeautifulSoup

load_dotenv(os.path.join(os.path.dirname(__file__), '..', '.env'))

# 1. Initialize OpenRouter Client
client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=os.getenv("OPENROUTER_API_KEY"),
)

# 2. Define the Agent's Tools
def internet_search(query):
    print(f"🔍 Searching for: {query}...")
    try:
        with DDGS() as ddgs:
            results = [r['href'] for r in ddgs.text(query, max_results=3)]
        return results
    except Exception as e:
        print(f"❌ Search error: {e}")
        return []

def fetch_url(url):
    print(f"📄 Fetching content from: {url}")
    try:
        headers = {'User-Agent': 'Mozilla/5.0'}
        response = requests.get(url, timeout=10, headers=headers)
        soup = BeautifulSoup(response.content, 'html.parser')
        
        for script_or_style in soup(["script", "style"]):
            script_or_style.decompose()
            
        text = soup.get_text(separator=' ')
        lines = (line.strip() for line in text.splitlines())
        chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
        clean_text = '\n'.join(chunk for chunk in chunks if chunk)
        
        return clean_text[:4000] 
    except Exception as e:
        return f"Failed to retrieve content: {e}"

# 3. Create Agent Logic
def run_research_agent(user_question):
    print(f"🚀 Starting Research Agent...")

    FREE_MODEL ="arcee-ai/trinity-large-preview:free"

    # STEP 1: Refine the query
    refine_msg = client.chat.completions.create(
        model=FREE_MODEL,
        messages=[{"role": "user", "content": f"Refine this into a professional search query: {user_question}"}]
    )
    search_query = refine_msg.choices[0].message.content
    print(f"✨ Refined Query: {search_query}")

    # STEP 2: Execute Search
    top_3_links = internet_search(search_query)
    
    if not top_3_links:
        return "Sorry, I couldn't find any search results."

    # STEP 3: Fetch Data
    combined_source_data = ""
    for i, link in enumerate(top_3_links, 1):
        content = fetch_url(link)
        combined_source_data += f"\n--- DATA FROM SOURCE {i} ({link}) ---\n{content}\n"

    # STEP 4: Final Synthesis
    system_prompt = (
        "You are an expert Research Agent. "
        "Use the provided content from the top-3 ranking sites to answer the user's question. "
        "Ensure your answer is based ONLY on the provided web data. Be professional and concise."
    )

    final_answer = client.chat.completions.create(
        model=FREE_MODEL,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"User Question: {user_question}\n\nWeb Data:\n{combined_source_data}"}
        ]
    )
    
    return final_answer.choices[0].message.content

# --- Run the Agent ---
if __name__ == "__main__":
    user_query = "write small paragraph about CR7?"
    result = run_research_agent(user_query)
    print("\n✅ FINAL RESEARCH REPORT:\n", result)