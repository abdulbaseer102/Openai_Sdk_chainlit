import os
import json
import requests
import chainlit as cl
from bs4 import BeautifulSoup
from dotenv import load_dotenv
from agents import Agent, Runner, AsyncOpenAI, OpenAIChatCompletionsModel
from agents.run import RunConfig
from agents.tool import function_tool

# Load environment variables
load_dotenv()

# 🔑 API Keys (Replace with your own if hardcoding)
GEMINI_API_KEY = "GEMINI_API_KEY"
SERPER_API_KEY = "SERPER_API_KEY"

# ✅ Initialize Gemini AI via OpenAI SDK Format
external_client = AsyncOpenAI(
    api_key=GEMINI_API_KEY,
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/"
)
model = OpenAIChatCompletionsModel(
    model="gemini-2.0-flash",
    openai_client=external_client
)
config = RunConfig(model=model, model_provider=external_client, tracing_disabled=True)

# ✅ Google Search Tool using Serper API
@function_tool
def google_search_tool(query: str) -> str:
    """
    Perform a Google search using Serper API and return the top 3 results.
    """
    url = "https://google.serper.dev/search"
    headers = {"X-API-KEY": SERPER_API_KEY, "Content-Type": "application/json"}
    payload = json.dumps({"q": query, "num": 3})
    
    response = requests.post(url, headers=headers, data=payload)
    results = response.json()
    
    if "organic" in results:
        return "\n".join([f"🔎 {res['title']}: {res['link']}" for res in results["organic"]])
    return "No results found."

# ✅ Fetch Latest News Articles
@function_tool
def search_latest_news() -> str:
    url = "https://google.serper.dev/news"
    headers = {"X-API-KEY": SERPER_API_KEY, "Content-Type": "application/json"}
    payload = json.dumps({"q": "latest news 2025", "num": 5})
    
    response = requests.post(url, headers=headers, data=payload)
    results = response.json()
    
    if "news" in results:
        return "\n".join([f"📰 {res['title']}: {res['link']}" for res in results["news"]])
    return "No news found."

# ✅ Create AI Agent
agent = Agent(
    name="World Information & News AI",
    instructions="You provide real-time global news updates and world information using Google Search and News.",
    model=model
)

# 🔌 Add tools
agent.tools.append(google_search_tool)
agent.tools.append(search_latest_news)

# ✅ Chainlit Setup
@cl.on_chat_start
async def start():
    cl.user_session.set("agent", agent)
    cl.user_session.set("config", config)
    await cl.Message(content="🌍 Welcome! Ask me anything about real-time global events.").send()

@cl.on_message
async def main(message: cl.Message):
    msg = cl.Message(content="🔍 Fetching latest information...")
    await msg.send()

    agent = cl.user_session.get("agent")
    config = cl.user_session.get("config")

    try:
        result = Runner.run_sync(agent, message.content, run_config=config)
        msg.content = result.final_output
    except Exception as e:
        msg.content = f"⚠ Error: {str(e)}"
    
    await msg.update()

