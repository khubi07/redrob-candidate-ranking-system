from src.llm.llm_engine import LLMEngine
from src.llm.prompts import (
    SYSTEM_PROMPT,
    build_candidate_prompt
)
from src.llm.parser import parse_analysis
from src.llm.schemas import (
    CandidateContext,
    CandidateAnalysis
)


class CandidateEvaluator:

    def __init__(self):

        self.llm = LLMEngine()


    def evaluate(
        self,
        context: CandidateContext
    ) -> CandidateAnalysis:

        user_prompt = build_candidate_prompt(context)

        response = self.llm.generate(
            system_prompt=SYSTEM_PROMPT,
            user_prompt=user_prompt
        )

        analysis = parse_analysis(response)

        return analysis