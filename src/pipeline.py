import json
from pathlib import Path

from sentence_transformers import SentenceTransformer
from docx import Document

from src.data_loader import load_candidates
from src.job_representation import build_job_description
from src.retrieval import Retriever
from src.ranking import Ranker
# Configuration
    # =====================================================

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

# -------------------------------------------------
    # Load embedding model
    # -------------------------------------------------

embedding_model = SentenceTransformer(MODEL_NAME)


BASE_DIR = Path(__file__).resolve().parent.parent

CANDIDATES_PATH = BASE_DIR / "data" / "candidates.jsonl"
JOB_DESCRIPTION_PATH = (
        BASE_DIR 
        / "India_Runs_Hackathon"
        / "job_description.docx"
    )



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

def initialize_pipeline():
    """
    Initialize everything that doesn't change.
    """

    candidates = load_candidates(
        str(CANDIDATES_PATH),
        limit=100
    )

    jd_text = load_job_description(JOB_DESCRIPTION_PATH)
    job = build_job_description(jd_text)

    retriever = Retriever(embedding_model)
    ranker = Ranker(embedding_model)

    retriever.build_bm25_index(candidates)
    retriever.build_experience_embedding_index(candidates)

    ranker.build_requirement_embeddings(job)
    ranker.build_experience_embeddings(candidates)

    return retriever, ranker, job

def run_pipeline(retriever, ranker, job):

    retrieved_candidates = retriever.retrieve_candidates(job)

    rankings = ranker.rank_candidates(
        job,
        retrieved_candidates
    )

    return rankings
    

