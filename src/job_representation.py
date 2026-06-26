from dataclasses import dataclass
from typing import List, Dict, Any


@dataclass
class JobDescription:

    # Full JD text for retrieval
    document: str

    # Compact query used by embedding retrieval
    semantic_query: str

    # Core requirements
    requirements: Dict[str, float]

    

    # Nice-to-have requirements
    preferences: List[str]

    # Hard constraints
    constraints: Dict

    # Desired behavioral traits
    traits: List[str]

    # Things the JD explicitly dislikes
    negative_signals: List[str]

    # Instructions hidden inside the JD
    ranking_hints: List[str]

    requirement_embeddings: Any = None


def build_job_description(jd_text: str) -> JobDescription:

    # Hard requirements from JD
    requirements = {
        "retrieval systems": 3.0,
        "ranking systems": 3.0,
        "recommendation systems": 3.0,
        "search systems": 3.0,

        "embeddings": 2.5,
        "vector databases": 2.5,

        "python": 1.5,
        "llms": 1.5,
        "fine-tuning": 1.0,
        "bm25": 1.0,
        "ndcg": 1.0,
        "mrr": 1.0,
        "map": 1.0
    }

    # Nice-to-have skills
    preferences = [
        "lora",
        "qlora",
        "peft",
        "learning to rank",
        "xgboost",
        "distributed systems",
        "inference optimization"
    ]

    # Hard constraints
    constraints = {
        "min_years_experience": 5,
        "max_years_experience": 9
    }

    # Behavioral signals
    traits = [
        "product mindset",
        "builder",
        "scrappy",
        "ships fast",
        "pragmatic"
    ]
    #for penalty
    negative_signals = [
        "research_only",
        "langchain_only",
        "consulting_only",
        "inactive_candidate",
        "no_recent_production_code",
        "framework_enthusiast",
        "cv_only",
        "speech_only",
        "robotics_only"
    ]


    ranking_hints = [
        "evidence_over_keywords",
        "behavioral_signals_matter",
        "precision_over_recall",
        "product_company_experience",
        "retrieval_ranking_recommendation_experience",
        "active_job_seeker_bonus"
    ]

    # Compact query used for semantic retrieval
    semantic_query = " ".join(
        list(requirements.keys()) + preferences
    )

    return JobDescription(
        document=jd_text,
        semantic_query=semantic_query,
        requirements=requirements,
        preferences=preferences,
        constraints=constraints,
        traits=traits,
        negative_signals=negative_signals,
        ranking_hints=ranking_hints
    )