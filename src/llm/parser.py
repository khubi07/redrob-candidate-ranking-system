from pydantic import ValidationError
from src.llm.schemas import CandidateAnalysis
from src.llm.exceptions import LLMParsingError

    
def clean_response(response: str) -> str:
    """
    Removes markdown formatting that the LLM may add.
    """

    response = response.strip()

    response = response.replace("```json", "")
    response = response.replace("```", "")

    return response.strip()

def parse_analysis(response: str) -> CandidateAnalysis:

    response = clean_response(response)

    try:
        return CandidateAnalysis.model_validate_json(response)

    except ValidationError as e:
        raise LLMParsingError(str(e))