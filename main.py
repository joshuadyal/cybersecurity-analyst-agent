# pip install -qU langchain langchain-ollama
from langchain.agents import create_agent
from vector import retriever


def get_weather(city: str) -> str:
    """Get weather for a given city."""
    return f"It's always sunny in {city}!"


def get_incidents(question: str):
    """Get previous cybersecurity incidents based on a question."""
    return retriever.invoke(question)


agent = create_agent(
    model="ollama:qwen3:0.6b",
    tools=[get_weather, get_incidents],
    system_prompt="You are a helpful cybersecurity assistant analyst.",
)

user_input = input("What is your input?: ")

result = agent.invoke({"messages": [{"role": "user", "content": user_input}]})
print(result["messages"][-1].content_blocks)
