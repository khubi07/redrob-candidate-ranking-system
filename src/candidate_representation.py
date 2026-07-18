from dataclasses import dataclass
from typing import List, Dict, Any

@dataclass
class Experience:
    company: str
    title: str
    duration_months: int
    is_current: bool
    industry: str
    company_size: str
    description: str
    # Used for evidence matching
    evidence_text: str = ""

    # Filled later by Ranker
    embedding: Any = None

@dataclass
class Skill:
    name: str
    proficiency: str
    endorsements: int
    duration_months: int

@dataclass
class Candidate:
    candidate_id: str

    retrieval_document: str

    experiences: List[Experience]

    skills: List[Skill]

    signals: Dict[str, Any]

    metadata: Dict[str, Any]

    document_embedding: Any = None

    evidence_document_embedding: Any = None

ACTION_VERBS = {
    "built",
    "implemented",
    "designed",
    "developed",
    "created",
    "trained",
    "optimized",
    "deployed",
    "engineered",
    "integrated",
    "fine-tuned",
    "improved",
    "architected",
    "migrated",
    "owned",
}

WEAK_PHRASES = [
    "interested in",
    "looking for",
    "open to",
    "excited about",
    "experimenting with",
    "learning",
    "taking online courses",
    "exploring",
]

JD_RELEVANT_SKILLS = {
    "python",
    "machine learning",
    "deep learning",
    "nlp",
    "llms",
    "rag",
    "faiss",
    "recommendation systems",
    "retrieval",
    "bm25",
    "embeddings",
    "vector databases",
    "search",
    "ranking",
    "transformers",
    "pytorch",
    "tensorflow",
    "xgboost",
    "lightgbm",
}

TECH_TERMS = {
    "model",
    "models",
    "pipeline",
    "pipelines",
    "api",
    "apis",
    "python",
    "ml",
    "machine learning",
    "deep learning",
    "ai",
    "llm",
    "embedding",
    "embeddings",
    "retrieval",
    "search",
    "ranking",
    "recommendation",
    "feature",
    "features",
    "training",
    "inference",
    "deploy",
    "deployment",
    "kafka",
    "spark",
    "docker",
    "kubernetes",
    "sql",
    "vector",
    "faiss",
    "bm25",
    "elasticsearch",
    "fastapi",
    "django",
    "transformer",
    "pytorch",
    "tensorflow",
}

def extract_skills(candidate_json):

    skills = []

    for skill in candidate_json.get("skills", []):

        skills.append(
            Skill(
                name=skill.get("name", ""),
                proficiency=skill.get("proficiency", ""),
                endorsements=skill.get("endorsements", 0),
                duration_months=skill.get("duration_months", 0)
            )
        )

    return skills

def create_retrieval_document(candidate_json: dict) -> str:

    sections = []
    experiences = candidate_json.get("career_history", [])

    for exp in experiences:

        company = exp.get("company", "") # type: ignore
        title = exp.get("title", "")
        description = exp.get("description", "")

        sections.append(
            f"Company: {company}"
        )
        sections.append(
            f"Experience Title: {title}"
        )

        sections.append(
            f"Experience Description: {description}"
        )

    profile = candidate_json.get("profile", {})
    headline = profile.get("headline", "")
    
    sections.append(f"Headline: {headline}")

    skills = candidate_json.get("skills", [])

    relevant_skills = []

    for skill in skills:

        name = skill.get("name", "")
        lower = name.lower()

        if lower in JD_RELEVANT_SKILLS:
            relevant_skills.append(name)

    relevant_skills = sorted(set(relevant_skills))

    if relevant_skills:
        sections.append(
            "Skills: " + ", ".join(relevant_skills)
        )

    return "\n".join(sections)

def create_evidence_document(candidate_json: dict) -> str:
    """
    Build a retrieval-friendly document using
    only evidence from work experience.
    """

    parts = []

    career_history = candidate_json.get("career_history", [])

    for exp in career_history:

        title = exp.get("title", "")
        company = exp.get("company", "")
        description = exp.get("description", "")

        section = f"""
            Role: {title}
           
            Work:
            {description}
            """

        parts.append(section.strip())

    return "\n\n".join(parts)


def extract_capability_text(
    candidate_id: str,
    title: str,
    description: str,
) -> str:
    """
    Keep only sentences that describe
    technical engineering work.
    """
    import re
    sentences = re.split(
        r'(?<=[.!?])\s+',
        description
    )

    kept = []

    for sentence in sentences:

        s = sentence.lower()

        has_action = any(
            verb in s
            for verb in ACTION_VERBS
        )

        import re

        has_tech = any(
            re.search(rf"\b{re.escape(term)}\b", s)
            for term in TECH_TERMS
        )

        matched_terms = [
            term
            for term in TECH_TERMS
            if re.search(rf"\b{re.escape(term)}\b", s)
        ]

        # if candidate_id == "CAND_0000083":
        #     print("=" * 60)
        #     print(sentence)
        #     print("Action:", has_action)
        #     print("Matched TECH_TERMS:", matched_terms)

        if has_action and has_tech:
            kept.append(sentence.strip())
            
    if not kept:
        return ""

    return (
        title + ".\n" +
        "\n".join(kept)
    )

def extract_experiences(candidate_json):

    experiences = []

    for exp in candidate_json.get("career_history", []):

        title = exp.get("title") or ""
        description = exp.get("description") or ""
        experiences.append(
            Experience(
                company=exp.get("company", ""),
                title=title,
                duration_months=exp.get("duration_months", 0),
                is_current=exp.get("is_current", False),
                industry=exp.get("industry", ""),
                company_size=exp.get("company_size", ""),
                description=description,
                evidence_text=extract_capability_text(
                    candidate_json["candidate_id"],
                    title,
                    description
                ),
            )
        )

    return experiences

def create_metadata(candidate_json: dict):
    profile = candidate_json.get("profile", {})
    return {

        "num_skills":
            len(candidate_json.get("skills", [])),

        "num_jobs":
            len(candidate_json.get("career_history", [])),

        "num_education":
            len(candidate_json.get("education", [])),

         "years_of_experience": profile.get(
            "years_of_experience",
            0.0
        ),
        "headline": profile.get("headline", "")
    }

def build_candidate(candidate_json: dict) -> Candidate:

    return Candidate(

        candidate_id=
        candidate_json["candidate_id"],

        retrieval_document=
        create_retrieval_document(candidate_json),

        experiences=
        extract_experiences(candidate_json),

        skills=
        extract_skills(candidate_json),

        signals=
        candidate_json.get("redrob_signals", {}),

        metadata=
        create_metadata(candidate_json),

        evidence_document_embedding=
        create_evidence_document(candidate_json),
    )
