import ollama
from dotenv import load_dotenv

load_dotenv()


class LLMClient:
    def __init__(self, model: str = "llama3.1:8b"):
        self.model = model

    def complete(self, system_prompt: str, user_prompt: str) -> str:
        try:
            response = self._chat(system_prompt, user_prompt, json_mode=True)
            return response["message"]["content"]
        except Exception as exc:
            print(f"[LLMClient] JSON-mode call failed: {exc}")
            try:
                response = self._chat(system_prompt, user_prompt, json_mode=False)
                return response["message"]["content"]
            except Exception as fallback_error:
                print(f"[LLMClient] Error calling Ollama: {fallback_error}")
                return ""

    def complete_text(self, system_prompt: str, user_prompt: str) -> str:
        try:
            response = self._chat(system_prompt, user_prompt, json_mode=False)
            return response["message"]["content"]
        except Exception as exc:
            print(f"[LLMClient] Text-mode call failed: {exc}")
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


class DummyLLMClient:
    def complete(self, system_prompt: str, user_prompt: str) -> str:
        print("[WARNING] DummyLLMClient used - no provider response available")
        return ""

    def complete_text(self, system_prompt: str, user_prompt: str) -> str:
        return ""


def create_llm_client(provider: str, model: str | None = None):
    """
    provider: "ollama" | "openai" | "anthropic" | "dummy"
    Returns an object with complete(system_prompt, user_prompt) -> str.
    OpenAI and Anthropic remain placeholders in this branch and route to Ollama.
    """
    if provider == "ollama":
        return LLMClient(model=model or "llama3.1:8b")
    if provider in {"openai", "anthropic"}:
        print(f"[INFO] {provider} placeholder requested; routing to Ollama instead")
        return LLMClient(model=model or "llama3.1:8b")
    if provider == "dummy":
        return DummyLLMClient()
    raise ValueError(f"Unknown provider: {provider}")
