import json
from pathlib import Path

from sentence_transformers import SentenceTransformer
from docx import Document

from src.data_loader import load_candidates
from src.job_representation import build_job_description
from src.retrieval import Retriever
from src.ranking import Ranker


# =====================================================
# Configuration
# =====================================================

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

BASE_DIR = Path(__file__).resolve().parent.parent

CANDIDATES_PATH = BASE_DIR / "data" / "candidates.jsonl"
JOB_DESCRIPTION_PATH = (
    BASE_DIR 
    / "India_Runs_Hackathon"
    / "job_description.docx"
)
TOP_K_RESULTS = 10


# =====================================================
# Helper Functions
# =====================================================

def load_job_description(file_path: Path):
    """
    Read a DOCX job description into plain text.
    """

    doc = Document(str(file_path))

    return "\n".join(
        paragraph.text
        for paragraph in doc.paragraphs
    )


# =====================================================
# Main Pipeline
# =====================================================

def main():

    print("=" * 60)
    print("AI Recruiter")
    print("=" * 60)

    # -------------------------------------------------
    # Load embedding model
    # -------------------------------------------------

    print("\nLoading embedding model...")

    embedding_model = SentenceTransformer(MODEL_NAME)

    # -------------------------------------------------
    # Load candidates
    # -------------------------------------------------

    print("Loading candidates...")

    candidates = load_candidates(
        str(CANDIDATES_PATH),
        limit=100
        ) #converted path -> str

    print(f"Loaded {len(candidates)} candidates.")

    # -------------------------------------------------
    # Load Job Description
    # -------------------------------------------------

    print("\nLoading Job Description...")

    jd_text = load_job_description(JOB_DESCRIPTION_PATH)

    job = build_job_description(jd_text)

    # -------------------------------------------------
    # Initialize components
    # -------------------------------------------------

    print("\nInitializing Retriever and Ranker...")

    retriever = Retriever(embedding_model)
    ranker = Ranker(embedding_model)

    # -------------------------------------------------
    # Build Retrieval Indexes
    # -------------------------------------------------

    print("\nBuilding BM25 Index...")

    retriever.build_bm25_index(candidates)

    print("Building Experience Embedding Index...")

    retriever.build_experience_embedding_index(candidates)

    # -------------------------------------------------
    # Build Ranking Embeddings
    # -------------------------------------------------

    print("\nBuilding Requirement Embeddings...")

    ranker.build_requirement_embeddings(job)

    print("Building Experience Embeddings...")

    ranker.build_experience_embeddings(candidates)

    # -------------------------------------------------
    # Retrieve Candidates
    # -------------------------------------------------

    print("\nRetrieving Candidates...")

    retrieved_candidates = retriever.retrieve_candidates(job)

    print(f"Retrieved {len(retrieved_candidates)} candidates.")

    # -------------------------------------------------
    # Rank Candidates
    # -------------------------------------------------

    print("\nRanking Candidates...")

    rankings = ranker.rank_candidates(
        job,
        retrieved_candidates
    )

    # -------------------------------------------------
    # Display Results
    # -------------------------------------------------

    print("\n" + "=" * 60)
    print("TOP CANDIDATES")
    print("=" * 60)

    for rank, result in enumerate(rankings[:TOP_K_RESULTS], start=1):

        candidate = result["candidate"]

        print(f"\n#{rank}")
        print("-" * 60)
        print(f"Candidate ID : {candidate.candidate_id}")
        print(f"Final Score  : {result['final_score']:.3f}")
        print(f"Evidence     : {result['evidence_score']:.3f}")
        print(f"Skills       : {result['skill_score']:.3f}")
        print(f"Experience   : {result['experience_score']:.3f}")
        print(f"Behavior     : {result['behavior_score']:.3f}")

        analysis = result["analysis"]

        if analysis:

            print("\nSummary")
            print(analysis.summary)

            print("\nStrengths")

            for strength in analysis.strengths:
                print(f"• {strength.title}")

            if analysis.missing_skills:

                print("\nMissing Skills")

                for skill in analysis.missing_skills:
                    print(f"• {skill}")

    print("\nDone.")


if __name__ == "__main__":
    main()