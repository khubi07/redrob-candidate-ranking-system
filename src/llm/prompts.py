from src.llm.schemas import CandidateContext

SYSTEM_PROMPT = """
You are a Senior Technical Recruiter specializing in AI and Software Engineering.

Your job is to objectively evaluate candidates against a job description.

Rules:
- Evaluate ONLY the supplied evidence.
- Never assume skills that are not demonstrated.
- If a skill appears in the skills list but is not supported by experience, mention it.
- Base every conclusion on evidence.
- Return ONLY valid JSON.
"""


def build_candidate_prompt(context: CandidateContext) -> str:

    return f"""
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
{chr(10).join(context.matched_experiences)}

----------------------------

EVIDENCE TEXT
=============
{context.evidence_text}

----------------------------

CURRENT SCORES
==============
Evidence Score: {context.evidence_score}

Skill Score: {context.skill_score}

Experience Score: {context.experience_score}

----------------------------

Evaluate this candidate.

Return your response in EXACTLY this JSON format.

{
    "experience_relevance": "High",
    "production_readiness": "Medium",

    "strengths": [
        {
            "title": "...",
            "evidence": "..."
        }
    ],

    "weaknesses": [
        "..."
    ],

    "missing_skills": [
        "..."
    ],

    "unsupported_skill_claims": [
        "..."
    ],

    "risk_factors": [
        "..."
    ],

    "summary": "..."
}

Return ONLY valid JSON.
Do not include markdown.
Do not include explanations outside the JSON.
"""