import os
import json
import chromadb
import uuid
from neo4j import GraphDatabase
from typing import List, Tuple, Dict, Any
from dotenv import load_dotenv

from agents.llm_client import LLMClient

# Load environment variables
load_dotenv()

class OracleAgent:
    """
    The Oracle Agent (LLM-Augmented).
    
    Responsibilities:
    1.  Context Retrieval: Fetches episodic (Chroma) & causal (Neo4j) context.
    2.  Reasoning: Uses LLM (Ollama) to analyze the situation based on context.
    3.  Memory Synthesis: Writes insights back to Chroma and causal links to Neo4j.
    """

    def __init__(self, model_name="llama3.1:8b"):
        # --- Memory Setup ---
        chroma_path = os.getenv("CHROMA_PATH", "./chroma_db")
        self.chroma_client = chromadb.PersistentClient(path=chroma_path)
        self.collection = self.chroma_client.get_or_create_collection(name="episodes")

        uri = os.getenv("NEO4J_URI", "bolt://localhost:7687")
        user = os.getenv("NEO4J_USER", "neo4j")
        password = os.getenv("NEO4J_PASSWORD", "password")
        self.driver = GraphDatabase.driver(uri, auth=(user, password))

        # --- LLM Setup ---
        self.llm = LLMClient(model=model_name)

    def close(self):
        """Closes the Neo4j driver connection."""
        self.driver.close()

    # --- MAIN PIPELINE ---

    def analyze_situation(self, query: str) -> Dict[str, Any]:
        """
        The main cognitive loop of the Oracle.
        
        1. Recall context (Episodic + Causal).
        2. Reason (LLM).
        3. Write-back (Store new knowledge).
        4. Return structured insight.
        """
        print(f"\n[Oracle] Analyzing: '{query}'")

        # 1. Retrieval
        episodes = self.recall_similar_episodes(query)
        # Extract entities from query (naive split for demo) or use LLM extraction
        # For simplicity, we just query Neo4j for words in the query that might be entities.
        # In a real app, you'd use NER. Here we just grab all relations to see what matches.
        # Ideally, we pass the *entire* relevant subgraph. 
        # Let's just pass a generic request for now or search for specific keywords.
        causal_context = self.recall_entity_context(query) 

        # 2. Reasoning
        prompt = self._build_prompt(query, episodes, causal_context)
        raw_response = self.llm.complete(
            system_prompt="You are an expert Oracle Agent for a startup simulator.", 
            user_prompt=prompt
        )

        # 3. Parsing
        insight_json = self._parse_output(raw_response)

        # 4. Mandatory Write-Back
        if insight_json:
            # Store Summary to Chroma
            if "store_episode_summary" in insight_json:
                self.store_episode(
                    raw_text=insight_json["store_episode_summary"],
                    metadata={"source": "Oracle_Analysis", "query": query}
                )
            
            # Store Causal Links to Neo4j
            if "suggested_causal_links" in insight_json:
                self.store_causal_links(insight_json["suggested_causal_links"])

        return insight_json

    # --- HELPER METHODS ---

    def _build_prompt(self, query: str, episodes: Dict[str, Any], causal_context: List[Dict[str, str]]) -> str:
        # Format Episodes
        ep_text = ""
        if episodes["documents"]:
            ep_text = "\n".join([f"- {doc}" for doc in episodes["documents"][0]])
        
        # Format Causal Context
        causal_text = ""
        for relation in causal_context:
            causal_text += f"- ({relation['subject']}) --[{relation['predicate']}]--> ({relation['object']})\n"

        prompt = f"""
        Analyze the following situation based on memory.

        USER QUERY: "{query}"

        --- EPISODIC MEMORY (Past Events) ---
        {ep_text if ep_text else "No specific past episodes found."}

        --- CAUSAL MEMORY (Known Facts) ---
        {causal_text if causal_text else "No relevant causal links found."}

        --- INSTRUCTIONS ---
        1. Synthesize an insight about the situation.
        2. Identify if this triggers any known causal chains.
        3. Propose NEW causal links if you infer them.
        4. Summarize this analysis as a new 'episode' to store.

        OUTPUT JSON ONLY:
        {{
            "insight": "One sentence summary...",
            "predicted_effects": ["Effect 1", "Effect 2"],
            "suggested_causal_links": [ ["Subject", "PREDICATE", "Object"] ],
            "store_episode_summary": "Full text summary to save to memory..."
        }}
        """
        return prompt

    def _parse_output(self, raw_text: str) -> Dict[str, Any]:
        """Safely parses LLM output to JSON."""
        try:
            # Clean up potential markdown code blocks
            clean_text = raw_text.replace("```json", "").replace("```", "").strip()
            return json.loads(clean_text)
        except json.JSONDecodeError:
            print(f"[Oracle] JSON Parse Failed. Raw: {raw_text[:100]}...")
            return {
                "insight": raw_text, 
                "predicted_effects": [], 
                "suggested_causal_links": [],
                "store_episode_summary": raw_text # Fallback
            }

    # --- 1. WRITE METHODS ---

    def store_episode(self, raw_text: str, metadata: Dict[str, Any]):
        """Stores a raw text episode into ChromaDB with metadata."""
        print(f"[Chroma] Storing episode: {raw_text[:50]}...")
        doc_id = str(uuid.uuid4())
        self.collection.add(
            documents=[raw_text],
            metadatas=[metadata],
            ids=[doc_id]
        )

    def store_causal_links(self, triples: List[List[str]]):
        """
        Stores entity-relationship triples in Neo4j.
        Format: [[Subject, Predicate, Object], ...]
        """
        print(f"[Neo4j] Storing {len(triples)} triples...")
        if not triples:
            return

        with self.driver.session() as session:
            for item in triples:
                if len(item) != 3:
                    continue
                subj, pred, obj = item
                
                # Sanitize predicate: REPLACE SPACES with UNDERSCORES, UPPERCASE
                safe_pred = pred.replace(" ", "_").upper()
                # Remove any non-alphanumeric chars (except underscore) for safety
                safe_pred = "".join(c for c in safe_pred if c.isalnum() or c == '_')

                if not safe_pred:
                    continue

                # Cypher query
                query = (
                    f"MERGE (a:Entity {{name: $subj}}) "
                    f"MERGE (b:Entity {{name: $obj}}) "
                    f"MERGE (a)-[:{safe_pred}]->(b)"
                )
                session.run(query, subj=subj, obj=obj)

    # --- 2. READ METHODS ---

    def recall_similar_episodes(self, query: str, k: int = 3) -> Dict[str, Any]:
        """Retrieves top-k semantically similar episodes from ChromaDB."""
        print(f"[Chroma] Querying: '{query}'")
        return self.collection.query(query_texts=[query], n_results=k)

    def recall_entity_context(self, text_query: str) -> List[Dict[str, str]]:
        """
        Naive Context Retrieval from Neo4j:
        matches any node that appears as a substring in the query.
        """
        # 1. Get all node names (inefficient for huge DB, fine for prototype)
        # OR better: Fulltext search if configured.
        # For now, let's just search for nodes where 'name' is in the text_query words.
        
        words = text_query.split()
        results = []
        
        with self.driver.session() as session:
            for word in words:
                # Clean word
                clean_word = "".join(filter(str.isalnum, word))
                if len(clean_word) < 3: continue

                # Find relationships involving this entity (as subject or object)
                query = (
                    "MATCH (a:Entity)-[r]-(b:Entity) "
                    "WHERE a.name CONTAINS $token OR b.name CONTAINS $token "
                    "RETURN a.name as subj, type(r) as pred, b.name as obj "
                    "LIMIT 5"
                )
                res = session.run(query, token=clean_word)
                for record in res:
                    results.append({
                        "subject": record["subj"],
                        "predicate": record["pred"],
                        "object": record["obj"]
                    })
        return results
