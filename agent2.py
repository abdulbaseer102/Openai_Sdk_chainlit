import os
from dotenv import load_dotenv
from agents import Agent, Runner, AsyncOpenAI, OpenAIChatCompletionsModel, ModelSettings, function_tool
from agents.run import RunConfig
import asyncio

# Load environment variables
load_dotenv()

gemini_api_key = os.getenv("GEMINI_API_KEY")

if not gemini_api_key:
    raise ValueError("GEMINI_API_KEY is not set. Please ensure it is defined in your .env file.")

# Setup external client
external_client = AsyncOpenAI(
    api_key=gemini_api_key,
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
)

# Configure model
model = OpenAIChatCompletionsModel(
    model="gemini-2.0-flash",
    openai_client=external_client
)

config = RunConfig(
    model=model,
    model_provider=external_client,
)

# Define a function tool


# Define an agent
agent = Agent(
    name="weather agent",
    instructions="Always provide the current weather.",
    model=model,
)


# Running the agent
async def main():
    result = await Runner.run(agent, "weather in tandoallahyar", run_config=config)
    print(result.final_output)

if __name__ == "__main__":
    asyncio.run(main())
