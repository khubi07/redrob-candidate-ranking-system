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

        corpus = []

        self.candidate_ids = []

        for candidate in candidates:

            corpus.append(
                candidate.retrieval_document.split()
            )

            self.candidate_ids.append(
                candidate.candidate_id
            )

        self.bm25 = BM25Okapi(corpus)

    def build_embedding_index(self, candidates):

        documents = [
            candidate.retrieval_document
            for candidate in candidates
        ]

        self.candidate_embeddings = (
            self.embedding_model.encode(
                documents,
                show_progress_bar=True
            )
        )

    def bm25_search(self, query, top_k=500):

        tokenized_query = query.split()

        scores = self.bm25.get_scores(  # type: ignore
            tokenized_query
        )

        top_idx = np.argsort(scores)[::-1][:top_k]

        return [
            (self.candidate_ids[i], scores[i])
            for i in top_idx
        ]
    
    def embedding_search(
        self,
        query,
        top_k=500
    ):

        query_embedding = (
            self.embedding_model.encode(query)
        )

        scores = (
            self.candidate_embeddings
            @ query_embedding
        )

        top_idx = np.argsort(scores)[::-1][:top_k]

        return [
            (self.candidate_ids[i], scores[i])
            for i in top_idx
        ]