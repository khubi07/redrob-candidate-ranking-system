# BM25 for keyword-based retrieval
from rank_bm25 import BM25Okapi

# Sentence embeddings for semantic retrieval
from sentence_transformers import SentenceTransformer

import numpy as np

HIGH_PRIORITY_TITLES = {
    "Search Engineer",
    "Recommendation Systems Engineer",
    "Applied ML Engineer",
    "ML Engineer",
    "Machine Learning Engineer",
    "NLP Engineer",
    "Data Engineer",
    "Senior Data Engineer",
}

MEDIUM_PRIORITY_TITLES = {
    "Backend Engineer",
    "Software Engineer",
    "DevOps Engineer",
    "Java Developer",
    "Full Stack Developer",
    "Cloud Engineer",
    "Analytics Engineer",
}
class Retriever:

    def __init__(self, embedding_model):

        self.embedding_model = embedding_model

        # BM25 index
        self.bm25 = None

        # Candidate IDs in index order
        self.candidate_ids = []

        # Candidate embedding matrix
        self.candidate_embeddings = None

        # -----------------------------
        # Experience-level retrieval
        # -----------------------------

        # Embedding for every experience
        self.experience_embeddings = None

        # Candidate ID corresponding to each experience embedding
        self.experience_candidate_ids = []

        # Text document of each experience
        self.experience_documents = []

        # Extra info for debugging
        self.experience_metadata = []

    def build_bm25_index(self, candidates):
        self.candidate_lookup = {
        c.candidate_id: c
        for c in candidates
    }
        # BM25 expects tokenized documents
        corpus = []

        self.candidate_ids = []

        for candidate in candidates:

            corpus.append(
                candidate.retrieval_document.lower().split()
            )

            self.candidate_ids.append(
                candidate.candidate_id
            )

        # Build the searchable BM25 index
        self.bm25 = BM25Okapi(corpus)

    def build_embedding_index(self, candidates):

        # Extract retrieval documents
        documents = [
            candidate.retrieval_document
            for candidate in candidates
        ]

        # Generate embeddings for all candidates
        self.candidate_embeddings = self.embedding_model.encode(
            documents,
            show_progress_bar=True,
            convert_to_numpy=True,
            normalize_embeddings=True
        )

    def build_experience_embedding_index(
        self,
        candidates
    ):

        # Clear previous index
        self.experience_candidate_ids = []
        self.experience_documents = []
        self.experience_metadata = []
        self.experience_titles = []

        # ----------------------------------
        # Collect every experience
        # ----------------------------------

        for candidate in candidates:

            for experience in candidate.experiences:

                # Candidate mapping
                self.experience_candidate_ids.append(
                    candidate.candidate_id
                )

                # Document to embed
                self.experience_documents.append(
                    experience.evidence_text
                )

                # Metadata (for debugging)
                self.experience_metadata.append(
                    {
                        "candidate_id": candidate.candidate_id,
                        "company": experience.company,
                        "title": experience.title,
                        "is_current": experience.is_current
                    }
                )
                self.experience_titles.append(experience.title)

        # ----------------------------------
        # Generate embeddings
        # ----------------------------------

        self.experience_embeddings = self.embedding_model.encode(
            self.experience_documents,
            show_progress_bar=True,
            convert_to_numpy=True,
            normalize_embeddings=True
        )

        print(f"Indexed {len(self.experience_documents)} experiences.")

    def _experience_embedding_search(
        self,
        query,
        top_k=500
    ):

        # -----------------------------
        # Embed the query
        # -----------------------------
        query_embedding = self.embedding_model.encode(
            query,
            normalize_embeddings=True
        )
        

        # -----------------------------
        # Similarity with ALL experiences
        # -----------------------------
        scores = self.experience_embeddings @ query_embedding

        # -----------------------------
        # Sort experiences
        # -----------------------------
        ranked_indices = np.argsort(scores)[::-1][:5000]
        # -----------------------------
        # Group by candidate
        # -----------------------------
        candidate_scores = {}

        for idx in ranked_indices:

            candidate_id = self.experience_candidate_ids[idx]
            score = float(scores[idx])

            title = self.experience_titles[idx]

            if title in HIGH_PRIORITY_TITLES:
                title_prior = 1.0
            elif title in MEDIUM_PRIORITY_TITLES:
                title_prior = 0.60
            else:
                title_prior = 0.20

            score *= title_prior

            # Keep BEST experience only
            if (
                candidate_id not in candidate_scores
                or score > candidate_scores[candidate_id]
            ):
                candidate_scores[candidate_id] = score

        # -----------------------------
        # Candidate ranking
        # -----------------------------
        results = sorted(
            candidate_scores.items(),
            key=lambda x: x[1],
            reverse=True
        )

        return results[:top_k]

    def _bm25_search(
        self,
        query,
        top_k=500
    ):
        """
        Search candidates using BM25 keyword matching.
        """

        # Safety check
        assert self.bm25 is not None, "Build BM25 index first."

        # BM25 expects a list of tokens
        tokenized_query = query.lower().split()

        # Compute BM25 scores
        scores = self.bm25.get_scores(tokenized_query)

        # Get indices of top candidates
        top_idx = np.argsort(scores)[::-1][:top_k]

        # Return (candidate_id, score)
        return [
            (self.candidate_ids[i], float(scores[i]))
            for i in top_idx
        ]   

    def _embedding_search(
        self,
        query,
        top_k=500
    ):

        # Convert query into embedding
        query_embedding = self.embedding_model.encode(
            query,
            normalize_embeddings=True
        )

        # Cosine similarity (because embeddings are normalized)
        scores = self.candidate_embeddings @ query_embedding

        # Get indices of highest scores
        top_idx = np.argsort(scores)[::-1][:top_k]

        return [
            (self.candidate_ids[i], float(scores[i]))
            for i in top_idx
        ]
    
    def _fuse_results(
        self,
        bm25_results,
        embedding_results,
        k=60
    ):
        """
        Combine BM25 and embedding retrieval using
        Reciprocal Rank Fusion (RRF).
        """

        rrf_scores = {}

        embedding_ids = {
            candidate_id
            for candidate_id, _ in embedding_results
        }

        # BM25 contribution
        for rank, (candidate_id, _) in enumerate(bm25_results, start=1):

            # Ignore candidates that semantic search
            # considers completely irrelevant.
            if candidate_id not in embedding_ids:
                continue

            rrf_scores[candidate_id] = (
                rrf_scores.get(candidate_id, 0)
                + 1 / (k + rank)
            )

        # Embedding contribution
        for rank, (candidate_id, _) in enumerate(embedding_results, start=1):

            rrf_scores[candidate_id] = (
                rrf_scores.get(candidate_id, 0)
                + 1 / (k + rank)
            )

        # Sort by highest RRF score
        return sorted(
            rrf_scores.items(),
            key=lambda x: x[1],
            reverse=True
        )[:500]
    
    #encapsulating bm25, embd and rrf into 1 public api which will b eused by ranking.py
    #so it does not hv to call 3 diff func

    def retrieve_candidates(
        self,
        job,
        top_k=5
    ):
        """
        Retrieve the most relevant candidates using
        BM25 + Embedding Search + RRF.
        """

        # -----------------------------
        # Step 1: Keyword Retrieval
        # -----------------------------
        bm25_results = self._bm25_search(
            job.document,
            top_k=top_k
        )
        self.last_bm25_results = bm25_results

        # -----------------------------
        # Step 2: Semantic Retrieval
        # -----------------------------
        
        embedding_results = self._experience_embedding_search(
            job.semantic_query,
            top_k=top_k
        )
        self.last_embedding_results = embedding_results

        # -----------------------------
        # Step 3: Hybrid Fusion
        # -----------------------------
        final_results = self._fuse_results(
            bm25_results,
            embedding_results
        )
        self.last_rrf_results = final_results

        return [
            self.candidate_lookup[candidate_id]
            for candidate_id, _ in final_results[:top_k]
        ]
    def inspect_candidate(
        self,
        candidate_id,
    ):
        """
        Inspect why a candidate was retrieved.
        """

        print("=" * 60)
        print(f"Candidate: {candidate_id}")

        # -------------------------
        # BM25
        # -------------------------
        bm25_rank = None
        bm25_score = None

        for rank, (cid, score) in enumerate(
            self.last_bm25_results,
            start=1
        ):
            if cid == candidate_id:
                bm25_rank = rank
                bm25_score = score
                break

        print("\nBM25")
        print("-----")
        print("Rank :", bm25_rank)
        print("Score:", bm25_score)

        # -------------------------
        # Embedding
        # -------------------------
        emb_rank = None
        emb_score = None

        for rank, (cid, score) in enumerate(
            self.last_embedding_results,
            start=1
        ):
            if cid == candidate_id:
                emb_rank = rank
                emb_score = score
                break

        print("\nEmbedding")
        print("---------")
        print("Rank :", emb_rank)
        print("Score:", emb_score)

        # -------------------------
        # RRF
        # -------------------------
        rrf_score = None

        for cid, score in self.last_rrf_results:
            if cid == candidate_id:
                rrf_score = score
                break

        print("\nRRF")
        print("---")
        print(rrf_score)

        print("=" * 60)

        print("\nExperience Evidence")
        print("-------------------")

        experiences = [
            (meta, doc)
            for cid, meta, doc in zip(
                self.experience_candidate_ids,
                self.experience_metadata,
                self.experience_documents,
            )
            if cid == candidate_id
        ]

        if not experiences:
            print("No experience evidence found.")
            return

        for i, (meta, document) in enumerate(experiences, start=1):
            print(f"\n[{i}] {meta['title']}")
            print(document)
