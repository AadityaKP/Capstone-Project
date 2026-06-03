import os
import ollama
from dotenv import load_dotenv

load_dotenv()


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

    def complete_text(self, system_prompt: str, user_prompt: str) -> str:
        try:
            response = self._chat(system_prompt, user_prompt, json_mode=False)
            return response['message']['content']
        except Exception as e:
            print(f"[LLMClient] Text-mode call failed: {e}")
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
        print("[WARNING] DummyLLMClient used — no API key or provider error")
        return ""

    def complete_text(self, system_prompt: str, user_prompt: str) -> str:
        return self.complete(system_prompt, user_prompt)


# ============================================================================
# COMMENTED OUT: OpenAI and Anthropic clients (use Ollama instead)
# ============================================================================

# class OpenAILLMClient:
#     def __init__(self, model: str = "o4-mini"):
#         self.model = model
#         self.api_key = os.environ.get("OPENAI_API_KEY")
#         if self.api_key is None:
#             print("[WARNING] OPENAI_API_KEY not found, OpenAILLMClient is dead")
#             self._dead = True
#         else:
#             self._dead = False
#             from openai import OpenAI
#             self.client = OpenAI(api_key=self.api_key)
#
#     def complete(self, system_prompt: str, user_prompt: str) -> str:
#         if self._dead:
#             return DummyLLMClient().complete(system_prompt, user_prompt)
#         try:
#             response = self.client.chat.completions.create(
#                 model=self.model,
#                 messages=[
#                     {"role": "system", "content": system_prompt},
#                     {"role": "user", "content": user_prompt},
#                 ],
#             )
#             return response.choices[0].message.content
#         except Exception as e:
#             print(f"[OpenAILLMClient] API call failed: {e}")
#             return DummyLLMClient().complete(system_prompt, user_prompt)
#
#
# class AnthropicLLMClient:
#     def __init__(self, model: str = "claude-sonnet-4-5-20251001"):
#         self.model = model
#         self.api_key = os.environ.get("ANTHROPIC_API_KEY")
#         if self.api_key is None:
#             print("[WARNING] ANTHROPIC_API_KEY not found, AnthropicLLMClient is dead")
#             self._dead = True
#         else:
#             self._dead = False
#             import anthropic
#             self.client = anthropic.Anthropic(api_key=self.api_key)
#
#     def complete(self, system_prompt: str, user_prompt: str) -> str:
#         if self._dead:
#             return DummyLLMClient().complete(system_prompt, user_prompt)
#         try:
#             response = self.client.messages.create(
#                 model=self.model,
#                 max_tokens=1024,
#                 system=system_prompt,
#                 messages=[{"role": "user", "content": user_prompt}],
#             )
#             return response.content[0].text
#         except Exception as e:
#             print(f"[AnthropicLLMClient] API call failed: {e}")
#             return DummyLLMClient().complete(system_prompt, user_prompt)


def create_llm_client(provider: str, model: str | None = None):
    """
    provider: "ollama" | "openai" | "anthropic" | "dummy"
    Returns an object with complete(system_prompt, user_prompt) -> str
    NOTE: OpenAI and Anthropic are commented out. Requests route to Ollama instead.
    """
    if provider == "ollama":
        return LLMClient(model=model or "llama3.1:8b")
    elif provider in ("openai", "anthropic"):
        print(f"[INFO] {provider} not available (commented out), routing to Ollama instead")
        return LLMClient(model=model or "llama3.1:8b")
    elif provider == "dummy":
        return DummyLLMClient()
    else:
        raise ValueError(f"Unknown provider: {provider}")
