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

    profile = candidate_json.get("profile", {})
    headline = profile.get("headline", "")
    summary = profile.get("summary", "")
    
    
    sections.append(f"Summary: {summary}")
    sections.append(f"Headline: {headline}")

    skills = candidate_json.get("skills", [])

    skill_names = [
        skill.get("name", "")
        for skill in skills
    ]

    if skill_names:
        sections.append(
            "Skills: " + ", ".join(skill_names)
        )

    education = candidate_json.get("education", [])

    for edu in education:
        degree = edu.get("degree", "")
        field = edu.get("field_of_study", "")

        sections.append(
            f"Education: {degree} {field}"
        )

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

    return "\n".join(sections)

def extract_experiences(candidate_json):

    experiences = []

    for exp in candidate_json.get("career_history", []):

        title = exp.get("title", "")
        description = exp.get("description", "")
        experiences.append(
            Experience(
                company=exp.get("company", ""),
                title=title,
                duration_months=exp.get("duration_months", 0),
                is_current=exp.get("is_current", False),
                industry=exp.get("industry", ""),
                company_size=exp.get("company_size", ""),
                description=description,
                evidence_text=f"{title}. {description}",
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
        create_metadata(candidate_json)
    )
