"""
Ranking Philosophy

Evidence > Skills > Metadata

Candidates are rewarded for demonstrating
skills through real work, not merely listing
keywords in their profile.
"""

from sentence_transformers import util
import numpy as np

class Ranker:

    def __init__(self, embedding_model):

        self.embedding_model = embedding_model

    def build_requirement_embeddings(
        self,
        job
    ):
        """
        Precompute embeddings for all job requirements.
        """

        job.requirement_embeddings = (
            self.embedding_model.encode(
                list(job.requirements.keys()),
                convert_to_tensor=True
            )
        )

    def build_experience_embeddings(
        self,
        candidates
    ):
        """
        Precompute embeddings for every experience.
        """

        for candidate in candidates:

            for exp in candidate.experiences:

                exp.evidence_embedding = (
                    self.embedding_model.encode(
                        exp.evidence_text,
                        convert_to_tensor=True
                    )
                )

    def _score_evidence(
        self,
        job,
        candidate
    ):
        """
        Score how well a candidate's experiences
        satisfy the job requirements.
        """

        weighted_score = 0.0
        total_weight = 0.0

        # Iterate over each requirement
        for (requirement, weight), req_embedding in zip(
            job.requirements.items(),
            job.requirement_embeddings
        ):
            best_score = 0.0

            # Compare against every experience
            for exp in candidate.experiences:

                score = util.cos_sim(
                    req_embedding,
                    exp.evidence_embedding
                ).item()
                
                if score > best_score:
                    best_score = score
           
            weighted_score += best_score * weight
            total_weight += weight
           
        return weighted_score / total_weight