import os
import chromadb
from neo4j import GraphDatabase
from dotenv import load_dotenv

load_dotenv()

def seed_chroma():
    print("--- Seeding ChromaDB ---")
    chroma_path = os.getenv("CHROMA_PATH", "./chroma_db")
    client = chromadb.PersistentClient(path=chroma_path)
    collection = client.get_or_create_collection(name="episodes")

    episodes = [
        {
            "id": "ep_001",
            "text": "In Q1, the company increased subscription price by 10 percent. Customer churn rose among small business users, leading to a slight revenue decline.",
            "metadata": { "quarter": "Q1", "reward": -0.2, "agent": "CFO" }
        },
        {
            "id": "ep_002",
            "text": "Marketing spend was increased on social media campaigns in Q2. User acquisition improved significantly and monthly recurring revenue grew.",
            "metadata": { "quarter": "Q2", "reward": 0.4, "agent": "CMO" }
        },
        {
            "id": "ep_003",
            "text": "A new onboarding flow reduced user drop-off during signup. Activation rate improved and customer satisfaction scores increased.",
            "metadata": { "quarter": "Q3", "reward": 0.5, "agent": "CPO" }
        },
        {
            "id": "ep_004",
            "text": "Server outages occurred due to scaling issues. Several enterprise clients reported downtime, causing temporary churn risk.",
            "metadata": { "quarter": "Q4", "reward": -0.4, "agent": "CTO" }
        }
    ]

    ids = [e["id"] for e in episodes]
    documents = [e["text"] for e in episodes]
    metadatas = [e["metadata"] for e in episodes]

    try:
        collection.upsert(
            ids=ids,
            documents=documents,
            metadatas=metadatas
        )
        print(f"Successfully inserted {len(episodes)} episodes.")
    except Exception as e:
        print(f"Error inserting into Chroma: {e}")


def seed_neo4j():
    print("\n--- Seeding Neo4j ---")
    uri = os.getenv("NEO4J_URI", "bolt://localhost:7687")
    user = os.getenv("NEO4J_USER", "neo4j")
    password = os.getenv("NEO4J_PASSWORD", "password")

    causal_triples = [
        ("Price Increase", "CAUSES", "Customer Churn"),
        ("Customer Churn", "REDUCES", "Revenue"),
        ("Marketing Spend", "INCREASES", "User Acquisition"),
        ("User Acquisition", "INCREASES", "MRR"),
        ("Improved Onboarding", "INCREASES", "Activation Rate"),
        ("Activation Rate", "INCREASES", "Customer Satisfaction"),
        ("Server Outage", "CAUSES", "Downtime"),
        ("Downtime", "CAUSES", "Enterprise Churn Risk")
    ]

    try:
        driver = GraphDatabase.driver(uri, auth=(user, password))
        with driver.session() as session:
            for subj, pred, obj in causal_triples:
                cypher = (
                    f"MERGE (a:Entity {{name: $subj}}) "
                    f"MERGE (b:Entity {{name: $obj}}) "
                    f"MERGE (a)-[:{pred}]->(b)"
                )
                session.run(cypher, subj=subj, obj=obj)
        print(f"Successfully inserted {len(causal_triples)} causal triples.")
        driver.close()
    except Exception as e:
        print(f"Error inserting into Neo4j: {e}")

if __name__ == "__main__":
    seed_chroma()
    seed_neo4j()
