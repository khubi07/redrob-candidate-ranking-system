from src.llm.schemas import CandidateContext
from textwrap import dedent

SYSTEM_PROMPT = """
You are a Senior Technical Recruiter specializing in AI and Software Engineering.

Your job is to objectively evaluate candidates against a job description.

Rules:
- Base your evaluation ONLY on the provided evidence.
- Do not assume experience or skills that are not demonstrated.
- If a listed skill is unsupported by the evidence, include it in "unsupported_skill_claims".
- Keep the summary concise (2-3 sentences).
- Return ONLY valid JSON.
- Do not use markdown.
- Do not include any text before or after the JSON.
"""
JSON_TEMPLATE = """
    {
        "experience_relevance": "High",
        "production_readiness": "Medium",

        "strengths": [
            {
                "title": "Python",
                "evidence": "Implemented ML pipeline using Python and scikit-learn."
            }
        ],

        "weaknesses": [
            "Limited production deployment experience."
        ],

        "missing_skills": [
            "Deep Learning"
        ],

        "unsupported_skill_claims": [
            "FastAPI"
        ],

        "risk_factors": [],

        "summary": "Candidate demonstrates solid Python and machine learning skills but lacks evidence of Deep Learning experience."
    }
    """


def build_candidate_prompt(context: CandidateContext) -> str:

    matched_experiences = "\n".join(
        f"- {exp}" for exp in context.matched_experiences
    )
    return  dedent(f"""

    JOB DESCRIPTION
    ================
    {context.job_description}

    ----------------------------

    CANDIDATE HEADLINE
    ==================
    {context.candidate_headline}

    ----------------------------

    CANDIDATE SKILLS
    ================
    {", ".join(context.candidate_skills)}

    ----------------------------

    MATCHED EXPERIENCES
    ===================
    
    {matched_experiences}
    

    ----------------------------

    EVIDENCE TEXT
    =============
    {context.evidence_text}

    ----------------------------

    CURRENT SCORES
    ==============
    Evidence Score: 92/100 (higher means stronger evidence matching the job description)
    Skill Score: 85/100 (higher means more required skills matched)
    Experience Score: 88/100 (higher means more relevant experience)
    

    ----------------------------

    Evaluate this candidate.

    Rules:
    - Base your evaluation only on the provided evidence.
    - Do not assume experience or skills that are not demonstrated.
    - If a section has no applicable items, return an empty array [].
    - Do not omit any fields.
    - Return ONLY valid JSON.
    - Do not include markdown or explanatory text.

    Return the response in this format:
    {JSON_TEMPLATE}
    
    """)
