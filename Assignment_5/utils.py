import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

load_dotenv(os.path.join(os.path.dirname(__file__), '..', '.env'))

model = ChatOpenAI(
    model="nvidia/nemotron-3-nano-30b-a3b:free",
    base_url="https://openrouter.ai/api/v1",
    api_key=os.getenv("OPENROUTER_API_KEY"),
)
