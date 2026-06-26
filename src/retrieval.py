# BM25 for keyword-based retrieval
from rank_bm25 import BM25Okapi

# Sentence embeddings for semantic retrieval
from sentence_transformers import SentenceTransformer

import numpy as np


class Retriever:

    def __init__(self):

        # Embedding model (loaded only once)
        self.embedding_model = SentenceTransformer(
            "sentence-transformers/all-MiniLM-L6-v2"
        )

        # BM25 index
        self.bm25 = None

        # Candidate IDs in index order
        self.candidate_ids = []

        # Embedding matrix
        self.candidate_embeddings = None

    def build_bm25_index(self, candidates):

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

        # BM25 contribution
        for rank, (candidate_id, _) in enumerate(bm25_results, start=1):

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
        top_k=500
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

        # -----------------------------
        # Step 2: Semantic Retrieval
        # -----------------------------
        embedding_results = self._embedding_search(
            job.semantic_query,
            top_k=top_k
        )

        # -----------------------------
        # Step 3: Hybrid Fusion
        # -----------------------------
        final_results = self._fuse_results(
            bm25_results,
            embedding_results
        )

        return final_results[:top_k]