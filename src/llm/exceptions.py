
class LLMGenerationError(Exception):
    """Raised when the LLM fails to generate a response."""


class LLMParsingError(Exception):
    """Raised when the LLM response cannot be parsed."""