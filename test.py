# test_evaluator.py

from src.llm.evaluator import CandidateEvaluator
from src.llm.schemas import CandidateContext

context = CandidateContext(

    job_description="""
Looking for an AI Engineer with Python,
Retrieval, RAG and Machine Learning.
""",

    candidate_headline="AI Engineer",

    candidate_skills=[
        "Python",
        "RAG",
        "FastAPI",
        "BM25"
    ],

    matched_experiences=[
        "Built an AI recruiter using BM25 and semantic search."
    ],

    evidence_text="""
Implemented hybrid retrieval using BM25
and Sentence Transformers.
""",

    evidence_score=94,

    skill_score=91,

    experience_score=89
)
evaluator = CandidateEvaluator()

analysis = evaluator.evaluate(context)

print(analysis)