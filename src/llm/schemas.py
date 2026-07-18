from pydantic import BaseModel
from enum import Enum

class CandidateContext(BaseModel):

    job_description: str

    candidate_headline: str

    candidate_skills: list[str]

    matched_experiences: list[str]

    evidence_text: str

    evidence_score: float

    skill_score: float

    experience_score: float

class Rating(str, Enum):

    HIGH = "High"

    MEDIUM = "Medium"

    LOW = "Low"

class EvidenceItem(BaseModel):

    title: str

    evidence: str

class CandidateAnalysis(BaseModel):

    experience_relevance: Rating

    production_readiness: Rating

    strengths: list[EvidenceItem]

    weaknesses: list[str]

    missing_skills: list[str]

    unsupported_skill_claims: list[str]

    risk_factors: list[str]

    summary: str