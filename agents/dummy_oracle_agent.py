import os
import chromadb
from neo4j import GraphDatabase
from typing import List, Tuple, Dict, Any
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class DummyOracleAgent:
    """
    A Dummy Oracle Agent acting as a reliable memory interface.
    
    Responsibilities:
    1. ChromaDB: Episodic memory (store/recall text + metadata)
    2. Neo4j: Causal/Semantic memory (store/recall entity relationships)
    """

    def __init__(self):
        # --- ChromaDB Setup ---
        chroma_path = os.getenv("CHROMA_PATH", "./chroma_db")
        self.chroma_client = chromadb.PersistentClient(path=chroma_path)
        self.collection = self.chroma_client.get_or_create_collection(name="episodes")

        # --- Neo4j Setup ---
        uri = os.getenv("NEO4J_URI", "bolt://localhost:7687")
        user = os.getenv("NEO4J_USER", "neo4j")
        password = os.getenv("NEO4J_PASSWORD", "password")
        
        self.driver = GraphDatabase.driver(uri, auth=(user, password))

    def close(self):
        """Closes the Neo4j driver connection."""
        self.driver.close()

    # --- 1. WRITE METHODS ---

    def store_episode(self, raw_text: str, metadata: Dict[str, Any]):
        """
        Stores a raw text episode into ChromaDB with metadata.
        """
        print(f"[Chroma] Storing episode: {raw_text[:50]}...")
        # Using a simple ID generation strategy (could be UUID in prod)
        import uuid
        doc_id = str(uuid.uuid4())
        
        self.collection.add(
            documents=[raw_text],
            metadatas=[metadata],
            ids=[doc_id]
        )

    def store_causal_links(self, triples: List[Tuple[str, str, str]]):
        """
        Stores entity-relationship triples in Neo4j.
        Format: (Subject, Predicate, Object) e.g. ("InterestRate", "INCREASES", "Inflation")
        """
        print(f"[Neo4j] Storing {len(triples)} triples...")
        with self.driver.session() as session:
            for subj, pred, obj in triples:
                # Cypher query to merge nodes and create relationship
                query = (
                    f"MERGE (a:Entity {{name: $subj}}) "
                    f"MERGE (b:Entity {{name: $obj}}) "
                    f"MERGE (a)-[:{pred}]->(b)"
                )
                session.run(query, subj=subj, obj=obj)

    # --- 2. READ METHODS ---

    def recall_similar_episodes(self, query: str, k: int = 3) -> Dict[str, Any]:
        """
        Retrieves top-k semantically similar episodes from ChromaDB.
        """
        print(f"[Chroma] Querying: '{query}'")
        results = self.collection.query(
            query_texts=[query],
            n_results=k
        )
        return results

    def recall_entity_context(self, entity_name: str) -> List[Dict[str, str]]:
        """
        Retrieves all outgoing relationships for a given entity from Neo4j.
        """
        print(f"[Neo4j] Recalling context for: '{entity_name}'")
        query = (
            "MATCH (a:Entity {name: $name})-[r]->(b:Entity) "
            "RETURN type(r) as predicate, b.name as object"
        )
        
        results = []
        with self.driver.session() as session:
            result = session.run(query, name=entity_name)
            for record in result:
                results.append({
                    "subject": entity_name,
                    "predicate": record["predicate"],
                    "object": record["object"]
                })
        return results
