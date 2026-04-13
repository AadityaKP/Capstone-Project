import ollama


class LLMClient:
    def __init__(self, model: str = "llama3.1:8b"):
        self.model = model

    def complete(self, system_prompt: str, user_prompt: str) -> str:
        try:
            response = self._chat(system_prompt, user_prompt, json_mode=True)
            return response['message']['content']
        except Exception as e:
            print(f"[LLMClient] JSON-mode call failed: {e}")
            try:
                response = self._chat(system_prompt, user_prompt, json_mode=False)
                return response['message']['content']
            except Exception as fallback_error:
                print(f"[LLMClient] Error calling Ollama: {fallback_error}")
                return ""

    def _chat(self, system_prompt: str, user_prompt: str, json_mode: bool):
        kwargs = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "options": {"temperature": 0},
        }
        if json_mode:
            kwargs["format"] = "json"
        return ollama.chat(**kwargs)
