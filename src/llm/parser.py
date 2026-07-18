from pydantic import ValidationError

from src.llm.schemas import CandidateAnalysis
from src.llm.exceptions import LLMParsingError


def parse_analysis(response: str) -> CandidateAnalysis:

    try:

        return CandidateAnalysis.model_validate_json(response)

    except ValidationError as e:

        raise LLMParsingError(
            f"Invalid JSON returned by LLM.\n{e}"
        )