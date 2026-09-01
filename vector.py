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

# If we need to create the vectorDB, prepare the documents
if add_documents:
    documents = []
    ids = []

    with open("incidents.json", "r", encoding="utf-8") as f:
        incident_data = json.load(f)

    for incident in incident_data["incidents"]:
        document = Document(
            page_content=f"""
                Title: {incident["title"]}
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
        ids.append(incident["id"])
        documents.append(document)

vector_store = Chroma(
    collection_name="incident_history",
    persist_directory=db_location,
    embedding_function=embeddings,
)

if add_documents:
    vector_store.add_documents(documents=documents, ids=ids)

number_of_incidents_to_return = 10
retriever = vector_store.as_retriever(
    search_kwargs={"k": number_of_incidents_to_return}
)
