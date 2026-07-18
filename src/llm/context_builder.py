from src.llm.schemas import CandidateContext


def build_candidate_context(
    job,
    candidate,
    ranking
) -> CandidateContext:

    return CandidateContext(

        job_description=job.document,

        candidate_skills=[
            skill.name
            for skill in candidate.skills
        ],

        evidence_score=ranking["evidence_score"],

        skill_score=ranking["skill_score"],

        experience_score=ranking["experience_score"],

        candidate_headline="",

        matched_experiences=[],

        evidence_text=""

    )