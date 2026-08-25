"""Back up / restore the causal graph as JSON.

The founder advise path is read-only against Neo4j (graph writes need an active
shock or end_episode, neither of which a single Boardroom.decide() triggers),
but the research graph is not reproducible, so keep a snapshot before testing.

    venv\\Scripts\\python.exe neo4j_backup.py dump
    venv\\Scripts\\python.exe neo4j_backup.py verify
    venv\\Scripts\\python.exe neo4j_backup.py restore backups/neo4j_<stamp>.json
"""

from __future__ import annotations

import json
import os
import sys
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv
from neo4j import GraphDatabase

load_dotenv()

ROOT = Path(__file__).resolve().parent
BACKUP_DIR = ROOT / "backups"


def driver():
    return GraphDatabase.driver(
        os.getenv("NEO4J_URI", "bolt://localhost:7687"),
        auth=(os.getenv("NEO4J_USER", "neo4j"), os.getenv("NEO4J_PASSWORD", "password")),
    )


def counts(session) -> dict:
    return {
        "nodes": session.run("MATCH (n) RETURN count(n) AS c").single()["c"],
        "relationships": session.run("MATCH ()-[r]->() RETURN count(r) AS c").single()["c"],
        "by_label": {
            r["l"]: r["c"]
            for r in session.run(
                "MATCH (n) RETURN labels(n)[0] AS l, count(*) AS c ORDER BY l"
            )
        },
    }


def dump() -> Path:
    BACKUP_DIR.mkdir(exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = BACKUP_DIR / f"neo4j_{stamp}.json"

    with driver() as d, d.session() as s:
        summary = counts(s)
        nodes = [
            {"id": r["id"], "labels": r["labels"], "properties": dict(r["props"])}
            for r in s.run(
                "MATCH (n) RETURN elementId(n) AS id, labels(n) AS labels, properties(n) AS props"
            )
        ]
        rels = [
            {
                "start": r["start"],
                "end": r["end"],
                "type": r["type"],
                "properties": dict(r["props"]),
            }
            for r in s.run(
                "MATCH (a)-[r]->(b) RETURN elementId(a) AS start, elementId(b) AS end, "
                "type(r) AS type, properties(r) AS props"
            )
        ]

    path.write_text(
        json.dumps(
            {"created_at": stamp, "summary": summary, "nodes": nodes, "relationships": rels},
            indent=2,
            default=str,
        ),
        encoding="utf-8",
    )
    print(f"wrote {path}")
    print(f"  {summary['nodes']} nodes, {summary['relationships']} relationships")
    for label, count in summary["by_label"].items():
        print(f"    {label}: {count}")
    return path


def verify(expected_path: str | None = None) -> None:
    """Compare the live graph against a backup - use after a test run to prove
    nothing was written."""
    with driver() as d, d.session() as s:
        live = counts(s)

    print("live:", live["nodes"], "nodes,", live["relationships"], "relationships")

    if expected_path is None:
        backups = sorted(BACKUP_DIR.glob("neo4j_*.json"))
        if not backups:
            print("no backup to compare against; run `dump` first")
            return
        expected_path = str(backups[-1])

    saved = json.loads(Path(expected_path).read_text(encoding="utf-8"))["summary"]
    print("saved:", saved["nodes"], "nodes,", saved["relationships"], "relationships")
    print(f"  ({Path(expected_path).name})")

    if live == saved:
        print("UNCHANGED - the graph was not written to.")
    else:
        print("DIFFERS:")
        for label in sorted(set(live["by_label"]) | set(saved["by_label"])):
            a, b = saved["by_label"].get(label, 0), live["by_label"].get(label, 0)
            if a != b:
                print(f"    {label}: {a} -> {b}")


def restore(path: str) -> None:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    confirm = input(f"This DELETES the current graph and restores {path}. Type 'yes': ")
    if confirm.strip().lower() != "yes":
        print("aborted")
        return

    with driver() as d, d.session() as s:
        s.run("MATCH (n) DETACH DELETE n")
        id_map: dict[str, int] = {}
        for node in data["nodes"]:
            labels = ":".join(node["labels"]) or "Node"
            rec = s.run(
                f"CREATE (n:{labels}) SET n = $props RETURN elementId(n) AS id",
                props=node["properties"],
            ).single()
            id_map[node["id"]] = rec["id"]
        for rel in data["relationships"]:
            s.run(
                f"MATCH (a) WHERE elementId(a) = $a MATCH (b) WHERE elementId(b) = $b "
                f"CREATE (a)-[r:{rel['type']}]->(b) SET r = $props",
                a=id_map[rel["start"]], b=id_map[rel["end"]], props=rel["properties"],
            )
    print(f"restored {len(data['nodes'])} nodes, {len(data['relationships'])} relationships")


if __name__ == "__main__":
    command = sys.argv[1] if len(sys.argv) > 1 else "dump"
    if command == "dump":
        dump()
    elif command == "verify":
        verify(sys.argv[2] if len(sys.argv) > 2 else None)
    elif command == "restore":
        if len(sys.argv) < 3:
            sys.exit("usage: neo4j_backup.py restore <path>")
        restore(sys.argv[2])
    else:
        sys.exit(f"unknown command: {command}")
