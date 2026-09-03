from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document
import os
import json

# Setup the embedding model
embeddings = OllamaEmbeddings(model="mxbai-embed-large")

# Set up vector db location
db_location = "./chrome_langchain_db"
add_documents = not os.path.exists(db_location)

# Create the vector stores
incident_store = Chroma(
    collection_name="incidents",
    persist_directory=db_location,
    embedding_function=embeddings,
)

incident_solution_store = Chroma(
    collection_name="incident_solutions",
    persist_directory=db_location,
    embedding_function=embeddings,
)

playbook_store = Chroma(
    collection_name="playbooks",
    persist_directory=db_location,
    embedding_function=embeddings,
)

policy_store = Chroma(
    collection_name="policies",
    persist_directory=db_location,
    embedding_function=embeddings,
)

security_history_store = Chroma(
    collection_name="security_history",
    persist_directory=db_location,
    embedding_function=embeddings,
)

# If we need to create the vectorDB, prepare the documents
if add_documents:
    # SET UP EMPTY LISTS FOR ALL VECTOR STORES
    incident_documents = []
    incident_ids = []

    incident_solution_documents = []
    incident_solution_ids = []

    playbook_documents = []
    playbook_ids = []

    policy_documents = []
    policy_ids = []

    security_history_documents = []
    security_history_ids = []

    # READ ALL JSON FILES
    with open("company-data/incidents.json", "r", encoding="utf-8") as f:
        incident_data = json.load(f)
    with open("company-data/incident-solutions.json", "r", encoding="utf-8") as f:
        incident_solution_data = json.load(f)
    with open("company-data/playbooks.json", "r", encoding="utf-8") as f:
        playbook_data = json.load(f)
    with open("company-data/policies.json", "r", encoding="utf-8") as f:
        policy_data = json.load(f)
    with open("company-data/security-history.json", "r", encoding="utf-8") as f:
        security_history_data = json.load(f)

    # ITERATE THROUGH ALL JSON DATA FILES
    for incident in incident_data["data"]:
        document = Document(
            page_content=f"""
                Incident ID: {incident["id"]}
                Title: {incident["title"]}
                Date: {incident["date"]}
                Severity: {incident["severity"]}
                Category: {incident["category"]}
                Status: {incident["status"]}
                Affected systems: {incident["affected_systems"]}
                Indicators: {incident["indicators"]}
                Summary: {incident["summary"]}
                Root cause: {incident["root_cause"]}
                Attack techniques: {incident["attack_techniques"]}
                Response: {incident["response"]}
                Outcome: {incident["outcome"]}
                Lessons learned: {incident['lessons_learned']}
                
            """,
            metadata={
                "incident_id": incident["id"],
                "severity": incident["severity"],
                "category": incident["category"],
                "status": incident["status"],
                "date": incident["date"],
            },
            id=incident["id"],
        )
        incident_ids.append(incident["id"])
        incident_documents.append(document)

    for solution in incident_solution_data["data"]:
        document = Document(
            page_content=f"""
                Solution ID: {solution["id"]}
                Incident ID: {solution["incident_id"]}
                Incident type: {solution["incident_type"]}
                Problem: {solution["problem"]}
                Investigation approach: {solution["investigation_approach"]}
                Actions taken: {solution["actions_taken"]}
                Result: {solution["result"]}
                Effectiveness: {solution["effectiveness"]}
                Lessons learned: {solution["lessons_learned"]}
                
            """,
            metadata={
                "solution_id": solution["id"],
                "incident_id": solution["incident_id"],
                "effectiveness": solution["effectiveness"],
            },
            id=solution["id"],
        )
        incident_solution_ids.append(solution["id"])
        incident_solution_documents.append(document)

    for playbook in playbook_data["data"]:
        document = Document(
            page_content=f"""
                Playbook ID: {playbook["id"]}
                Name: {playbook["name"]}
                Version: {playbook["version"]}
                Owner: {playbook["owner"]}
                Severity: {playbook["severity"]}
                Trigger conditions: {playbook["trigger_conditions"]}
                Initial actions: {playbook["initial_actions"]}
                Investigation: {playbook["investigation"]}
                Containment: {playbook["containment"]}
                Escalation: {playbook["escalation"]}
                Related incidents: {playbook["related_incidents"]}
            """,
            metadata={
                "playbook_id": playbook["id"],
                "version": playbook["version"],
                "owner": playbook["owner"],
            },
            id=playbook["id"],
        )
        playbook_ids.append(playbook["id"])
        playbook_documents.append(document)

    for policy in policy_data["data"]:
        document = Document(
            page_content=f"""
                Policy ID: {policy["id"]}
                Name: {policy["name"]}
                Version: {policy["version"]}
                Requirements: {policy["requirements"]}
                Monitoring_requirements: {policy["monitoring_requirements"]}
                Exceptions: {policy["exceptions"]}
                Related incidents: {policy["related_incidents"]}
            """,
            metadata={"id": policy["id"], "version": policy["version"]},
            id=policy["id"],
        )

        policy_ids.append(policy["id"])
        policy_documents.append(document)

    for history_record in security_history_data["data"]:
        document = Document(
            page_content=f"""
                History Record ID: {history_record["id"]}
                Period: {history_record["period"]}
                Total incidents: {history_record["total_incidents"]}
                Critical incidents: {history_record["critical_incidents"]}
                High incidents: {history_record["high_incidents"]}
                Medium incidents: {history_record["medium_incidents"]}
                Phishing incidents: {history_record["phishing_incidents"]}
                Malware incidents: {history_record["malware_incidents"]}
                Credential incidents: {history_record["credential_incidents"]}
                Mean time to detect (hours): {history_record["mean_time_to_detect_hours"]}
                Mean time to respond (hours): {history_record["mean_time_to_respond_hours"]}
                False positive rate: {history_record["false_positive_rate"]}
                Major observations: {history_record["major_observations"]}
                Top categories: {history_record["top_categories"]}
                Related incidents: {history_record["related_incidents"]}
            """,
            metadata={"id": history_record["id"], "period": history_record["period"]},
            id=history_record["id"],
        )

        security_history_ids.append(history_record["id"])
        security_history_documents.append(document)

    # ADD ALL DOCUMENTS TO RELEVANT VECTOR STORES
    incident_store.add_documents(documents=incident_documents, ids=incident_ids)

    incident_solution_store.add_documents(
        documents=incident_solution_documents, ids=incident_solution_ids
    )
    playbook_store.add_documents(documents=playbook_documents, ids=playbook_ids)

    policy_store.add_documents(documents=policy_documents, ids=policy_ids)

    security_history_store.add_documents(
        documents=security_history_documents, ids=security_history_ids
    )

# CREATE THE RETRIEVERS

# search_kwargs={"k": 5} means return 5 results
incident_retriever = incident_store.as_retriever(search_kwargs={"k": 5})
incident_solution_retriever = incident_solution_store.as_retriever(
    search_kwargs={"k": 5}
)
playbook_retriever = playbook_store.as_retriever(search_kwargs={"k": 5})
policy_retriever = policy_store.as_retriever(search_kwargs={"k": 5})
security_history_retriever = security_history_store.as_retriever(search_kwargs={"k": 5})
