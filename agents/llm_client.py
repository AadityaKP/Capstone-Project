import ollama
from typing import Optional, Dict, Any

class LLMClient:
    def __init__(self, model: str = "llama3.1:8b"):
        self.model = model

    def complete(self, system_prompt: str, user_prompt: str) -> str:
        try:
            response = ollama.chat(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ]
            )
            return response['message']['content']
        except Exception as e:
            print(f"[LLMClient] Error calling Ollama: {e}")
            return ""
