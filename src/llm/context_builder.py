from src.llm.schemas import CandidateContext


def build_candidate_context(
    job,
    candidate,
    ranking
) -> CandidateContext:

    return CandidateContext(

        job_description=job.document,

        candidate_headline=candidate.metadata.get(
            "headline",
            ""
        ),

        candidate_skills=[
            skill.name
            for skill in candidate.skills
        ],

        matched_experiences=[
            exp.evidence_text
            for exp in candidate.experiences
        ],

        evidence_text="\n\n".join(
            exp.evidence_text
            for exp in candidate.experiences
        ),

        evidence_score=ranking["evidence_score"],

        skill_score=ranking["skill_score"],

        experience_score=ranking["experience_score"]
    )