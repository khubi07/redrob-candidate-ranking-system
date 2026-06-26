from dataclasses import dataclass
from typing import List, Dict


@dataclass
class JobDescription:

    # Full JD text for retrieval
    document: str

    # Compact query used by embedding retrieval
    semantic_query: str

    # Core requirements
    requirements: List[str]

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


def build_job_description(jd_text: str) -> JobDescription:

    # Hard requirements from JD
    requirements = [
        "python",
        "embeddings",
        "retrieval systems",
        "ranking systems",
        "llms",
        "fine-tuning",
        "bm25",
        "vector databases",
        "ndcg",
        "mrr",
        "map",
        "recommendation systems",
        "search systems"
    ]

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
        requirements + preferences
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