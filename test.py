from src.llm.llm_engine import LLMEngine
from src.llm.prompts import SYSTEM_PROMPT, build_candidate_prompt
from src.llm.schemas import CandidateContext
from src.llm.parser import parse_analysis

context = CandidateContext(
    job_description="Need Python and Machine Learning",
    candidate_headline="AI Engineer",
    candidate_skills=["Python", "FastAPI"],
    matched_experiences=[
        "Built an ML model for spam detection."
    ],
    evidence_text="Implemented ML pipeline using sklearn.",
    evidence_score=92,
    skill_score=85,
    experience_score=88
)

prompt = build_candidate_prompt(context)

llm = LLMEngine()

response = llm.generate(
    system_prompt=SYSTEM_PROMPT,
    user_prompt=prompt
)

analysis = parse_analysis(response)

print(analysis)