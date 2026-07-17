# test_llm.py

from llm_engine import LLMEngine

llm = LLMEngine()

response = llm.chat(
    "Say hello in one sentence."
)

print(response)