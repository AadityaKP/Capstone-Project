"""Quick smoke-test for heterogeneous LLM routing."""
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from dotenv import load_dotenv
load_dotenv()

from agents.llm_client import create_llm_client

PROVIDERS = ["ollama", "openai", "anthropic", "dummy"]

for provider in PROVIDERS:
    client = create_llm_client(provider)
    result = client.complete(
        "You are a test assistant.",
        "Reply with the single word PONG and nothing else."
    )
    print(f"[{provider}] response snippet: {repr(result[:80])}")
    assert isinstance(result, str), f"{provider} did not return a string"

print("All provider smoke tests passed.")