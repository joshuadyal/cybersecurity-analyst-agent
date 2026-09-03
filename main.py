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


def search_incidents(query: str):
    """
    Search the company's historical cybersecurity incidents.

    Use this tool when looking for:
    - previous incidents
    - whether the company has experienced a particular attack
    - incidents involving a particular vulnerability or attack type
    - historical examples of an incident

    This tool contains INCIDENTS ONLY.
    It does not contain remediation steps.

    Args:
        query: Search terms to look for
    """
    return incident_retriever.invoke(query)


def search_incident_solutions(query: str):
    """
    Search solutions/remediation for previous company cybersecurity incidents.

    Use this tool when looking for:
    - how a previous incident was resolved
    - remediation steps
    - previous fixes
    - lessons learned from previous incidents

    This tool contains INCIDENT SOLUTIONS ONLY.

    Args:
        query: Search terms to look for"""
    return incident_solution_retriever.invoke(query)


def search_playbooks(query: str):
    """
    Search company playbooks that list the steps to take when an incident occurs.

    Use this tool when looking for:
    - a guide on how to approach an incident

    This tool contains PLAYBOOKS ONLY.

    Args:
        query: Search terms to look for"""
    return playbook_retriever.invoke(query)


def search_policies(query: str):
    """
    Search company policies. Policies are set out by the company and must be followed at all times.

    Use this tool when:
    - looking for specific policies
    - checking if a policy exists, in which case it must be adhered to
    - ensuring processes adhere to company policies



    Args:
        query: Search terms to look for"""
    return policy_retriever.invoke(query)


def search_security_history(query: str):
    """
    Search the company security-history records, which are split into quarters and summarise & total the number of incidents.

    Use this tool when the user asks about:
    - historical security trends or patterns
    - recurring types of security incidents
    - changes in the company's security posture over time
    - statistics about previous security events
    - quarterly security observations
    - whether a particular security issue or attack type has been observed
      repeatedly
    - historical context that may help understand a current security issue

    This tool contains security history and statistics only. It should not be
    used to determine the company's required procedures or policies. Use the
    Playbooks tool for incident response procedures and the Policies tool for
    mandatory company requirements.

    It should also not be used as the primary source for detailed information
    about a specific past incident. Use the Incidents tool for past incidents
    and the Incident Solutions tool for how those incidents were resolved.

    When searching, use the security topic, attack type, vulnerability,
    incident type, or relevant historical concept from the user's question.

    Args:
        query: Search terms to look for"""
    return security_history_retriever.invoke(query)


llm = ChatOllama(model="qwen3:4b", num_ctx=8192)

system_prompt = """
    You are a cybersecurity analyst.

    Your job is to help analyse and research security incidents, vulnerabilities,
    security events, and answer cybersecurity questions.

    You will be used by cybersecurity engineers.

    Use the available tools when you need information that is not already
    available in the conversation.

    When answering questions about the company, prioritise information
    retrieved from the company's internal knowledge base.

    The internal knowledge base consists of 5 separate collections:
    - Incidents - past cybersecurity incidents that have occured
    - Incident solutions - the solutions that resolved the incidents that have occured in the past
    - Playbooks - company-specific guidance that should be followed in the event of a specific incident
    - Policies - a separate database containing company-specific policies that must be followed at all times
    - Security history - Quaterly history logs that include statistics and key observations.

    You have access to tools that let you search the internal knowledge base.

    Company policies are only defined in the 'Policies' collection, and should not be inferred from different collections.
    
    Do not invent company-specific facts, incidents, vulnerabilities,
    policies, or procedures.

    If you do not have enough information to answer confidently, say so.

    When answering a question, determine whether multiple sources of
    information are required.

    If the question contains multiple distinct information requirements,
    use all relevant tools necessary to answer each requirement.

    Do not stop after using one tool if another tool is required to
    fully answer the user's question.

    For example, if a question asks both:
    1. whether a vulnerability exists, and
    2. whether the company has previously experienced that vulnerability,

    you should retrieve the relevant external vulnerability information
    AND search the company's internal incident knowledge base.

    After receiving a tool result, reassess the original question and
    determine whether additional/different tools are required before providing
    the final answer.

    Do not assume that information from one tool answers questions
    that require a different source.

    Clearly distinguish between:
    - Company-specific information
    - General cybersecurity knowledge
    - Information obtained from external sources

    Provide technically accurate and practical answers.
"""

agent: Runnable = create_agent(
    model=llm,
    tools=[
        search_incidents,
        search_incident_solutions,
        search_playbooks,
        search_policies,
        search_security_history,
    ],
    system_prompt=system_prompt,
)

while True:
    user_input = input("\nWhat is your input?: (q to quit) ")
    if user_input == "q":
        break

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
