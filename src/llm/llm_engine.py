from ollama import Client

from src.config import (
    OLLAMA_HOST,
    OLLAMA_MODEL,
    OLLAMA_TEMPERATURE,
)


class LLMEngine:

    def __init__(self):

        self.client = Client(host=OLLAMA_HOST)

        self.model = OLLAMA_MODEL

        self.temperature = OLLAMA_TEMPERATURE


    def generate(
        self,
        system_prompt: str,
        user_prompt: str
    ) -> str:

        response = self.client.chat(

            model=self.model,

            messages=[

                {
                    "role": "system",
                    "content": system_prompt
                },

                {
                    "role": "user",
                    "content": user_prompt
                }

            ],

            options={
                "temperature": self.temperature
            }

        )

        return response["message"]["content"]