# pip install -qU langchain langchain-ollama
from langchain.agents import create_agent
from langchain_ollama import ChatOllama
from langchain_core.utils.uuid import uuid7
from langchain_core.runnables import Runnable
from vector import (
    incident_retriever,
    incident_solution_retriever,
    playbook_retriever,
    policy_retriever,
    security_history_retriever,
)


def get_incidents(query: str):
    """Search the company's incident database for past cybersecurity incidents that match the query.

    Args:
        query: Search terms to look for
    """
    return incident_retriever.invoke(query)


def get_incident_solutions(query: str):
    """Search the company's incident solutions database for solutions to past cybersecurity incidents that match the query.

    Args:
        query: Search terms to look for"""
    return incident_solution_retriever.invoke(query)


def get_playbooks(query: str):
    """Search the company playbooks database for playbook records matching the query. Playbooks list the steps to take when an incident occurs.

    Args:
        query: Search terms to look for"""
    return playbook_retriever.invoke(query)


def get_policies(query: str):
    """Search the company policies database for records matching the query. Policies are set out by the company and must be followed at all times.

    Args:
        query: Search terms to look for"""
    return policy_retriever.invoke(query)


def get_security_history(query: str):
    """Search the company security-history database for records matching the query. Historical records are split into quarters and summarise & total the number of incidents.

    Args:
        query: Search terms to look for"""
    return security_history_retriever.invoke(query)


llm = ChatOllama(model="qwen3:4b", num_ctx=8192)

system_prompt = """
    You are a cybersecurity assistant.

    Your job is to help analyse and research security incidents, vulnerabilities,
    security events, and answer cybersecurity questions.

    You will be used by cybersecurity specialists.

    Use the available tools when you need information that is not already
    available in the conversation.

    When answering questions about the company, prioritise information
    retrieved from the company's internal knowledge base.

    The internal knowledge base consists of:
    - Incidents - past cybersecurity incidents that have occured
    - Incident solutions - the solutions that resolved the incidents that have occured in the past
    - Playbooks - company-specific guidance that should be followed in the event of a specific incident
    - Policies - a separate database containing company-specific policies that must be followed at all times
    - Security history - Quaterly history logs that include statistics and key observations.
    
    You have access to tools that let you search the internal knowledge base.

    Always follow company policies.
    
    Do not invent company-specific facts, incidents, vulnerabilities,
    policies, or procedures.

    If you do not have enough information to answer confidently, say so.

    Clearly distinguish between:
    - Company-specific information
    - General cybersecurity knowledge
    - Information obtained from external sources

    Provide technically accurate and practical answers.

    You may use more than one tool per response.
    You may use the same tool more than once, if necessary.
"""

agent: Runnable = create_agent(
    model=llm,
    tools=[
        get_incidents,
        get_incident_solutions,
        get_playbooks,
        get_policies,
        get_security_history,
    ],
    system_prompt=system_prompt,
)

user_input = input("What is your input?: ")

# # SIMPLE OUTPUT
# result = agent.invoke({"messages": [{"role": "user", "content": user_input}]})
# print(result["messages"][-1].content_blocks[0]["text"])

# # STREAMED OUTPUT
config = {"configurable": {"thread_id": str(uuid7())}}
stream = agent.stream_events(
    {"messages": [{"role": "user", "content": user_input}]},
    config=config,
    version="v3",
)

for kind, item in stream.interleave("messages", "tool_calls"):
    if kind == "messages":
        for token in item.text:
            print(token, end="", flush=True)
    elif kind == "tool_calls":
        print(f"\nTool call: {item.tool_name}({item.input})")
        for delta in item.output_deltas:
            print(delta, end="", flush=True)
        print(f"\nTool result: {item.output}")

final_state = stream.output
