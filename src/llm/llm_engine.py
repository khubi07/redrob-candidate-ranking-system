from ollama import chat
from src.config import OLLAMA_MODEL


class LLMEngine:

    def __init__(self):
        self.model = OLLAMA_MODEL

    def chat(self, prompt: str) -> str:

        response = chat(
            model=self.model,
            messages=[
                {
                    "role": "user",
                    "content": prompt
                }
            ]
        )

        return response.message.content or ""