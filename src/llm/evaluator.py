from src.llm.llm_engine import LLMEngine
from src.llm.prompts import (
    SYSTEM_PROMPT,
    build_candidate_prompt
)
from src.llm.parser import parse_analysis, LLMParsingError
from src.llm.schemas import (
    CandidateContext,
    CandidateAnalysis
)


class CandidateEvaluator:

    def __init__(self):

        self.llm = LLMEngine()


    def evaluate(self, context):

        user_prompt = build_candidate_prompt(context)

        for attempt in range(2):

            try:

                response = self.llm.generate(
                    SYSTEM_PROMPT,
                    user_prompt
                )
                
                return parse_analysis(response)

            except LLMParsingError as e:

                print(f"Attempt {attempt+1} failed.")
                print(e)

            

            raise LLMParsingError(
                "Unable to parse LLM response."
            )
                
    