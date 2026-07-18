"""
Ranking Philosophy

Evidence > Skills > Metadata

Candidates are rewarded for demonstrating
skills through real work, not merely listing
keywords in their profile.
"""

# ===================== TODOs (Ranking v2) =====================
# 1. Add skill alias matching
#    Example:
#    "vector databases" -> FAISS, Pinecone, Milvus, Qdrant
#
# 2. Store matched experience for each requirement
#    Used for explainable reasoning in submission.csv
#
# 3. Replace linear experience scoring with smoother decay.
#
# 4. Tune scoring weights using validation metrics (NDCG/MRR).
#
# 5. Add negative signal penalties
#    - research_only
#    - consulting_only
#    - langchain_only
#    - no_recent_production_code
#
# 6. Generate recruiter-friendly reasoning for every candidate.
# ==============================================================


from sentence_transformers import util
import numpy as np
from aliases import SKILL_ALIASES
from src.llm.context_builder import build_candidate_context
from src.llm.evaluator import CandidateEvaluator

class Ranker:

    def __init__(self, embedding_model):

        self.embedding_model = embedding_model
        self.evaluator = CandidateEvaluator()

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

                if not exp.evidence_text:
                    exp.evidence_embedding = None
                    continue

                exp.evidence_embedding = self.embedding_model.encode(
                    exp.evidence_text,
                    convert_to_tensor=True
                )

    def build_document_embeddings(
        self,
        candidates
    ):
        """
        Precompute embeddings for each
        candidate retrieval document.
        """

        for candidate in candidates:

            candidate.document_embedding = (
                self.embedding_model.encode(
                    candidate.retrieval_document,
                    convert_to_tensor=True
                )
            )

    def build_evidence_embeddings(
        self,
        candidates
    ):
        """
        Build embeddings from the
        evidence document only.
        """

        for candidate in candidates:

            candidate.evidence_document_embedding = (
                self.embedding_model.encode(
                    candidate.evidence_document,
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
        matched_experiences = {}
        weighted_score = 0.0
        total_weight = 0.0

        # Iterate over each requirement
        for (requirement, weight), req_embedding in zip(
            job.requirements.items(),
            job.requirement_embeddings
            
        ):
            print("Entered _score_evidence")
            best_score = 0.0
            best_experience = None

            # Compare against every experience

            
            for exp in candidate.experiences:

                if exp.evidence_embedding is None:
                    continue

                score = util.cos_sim(
                    req_embedding,
                    exp.evidence_embedding
                ).item()

                if score > best_score:
                    best_score = score
                    best_experience = exp

                                  
            SIMILARITY_FLOOR = 0.13

            adjusted_score = max(
                0.0,
                best_score - SIMILARITY_FLOOR
            )

            weighted_score += adjusted_score * weight

            total_weight += weight

            matched_experiences[requirement] = {
                "experience": best_experience,
                "score": best_score
            }

            print(
                f"{requirement:25}"
                f" Weight={weight:.1f}"
                f" Best={best_score:.3f}"
            )
        
        candidate.matched_experiences = matched_experiences

        print(
            f"\nFinal Evidence Score = "
            f"{weighted_score / total_weight:.3f}"
        )
        print("TOTAL WEIGHT =", total_weight)
        return weighted_score / total_weight
            
    def generate_reasoning(
        self,
        candidate
    ):

        reasons = []

        for requirement, info in candidate.matched_experiences.items():

            exp = info["experience"]

            if exp is None:
                continue

            if info["score"] < 0.30:
                continue

            reasons.append(
                f"{requirement}: {exp.title} at {exp.company}"
            )

        return "; ".join(reasons[:4])

    def _score_skills(
        self,
        job,
        candidate
    ):
        """
        Score how well a candidate's skills
        match the job requirements.
        """

        weighted_score = 0.0
        total_weight = 0.0

        candidate_skills = {
            skill.name.lower()
            for skill in candidate.skills
        }

        for requirement, weight in job.requirements.items():

            aliases = SKILL_ALIASES.get(
                requirement,
                [requirement]
            )

            matched = False

            for skill in candidate_skills:
                if any(
                    alias.lower() in skill
                    for alias in aliases
                ):
                    matched = True
                    break

            if matched:
                weighted_score += weight

            total_weight += weight

        return weighted_score / total_weight
    
    def _score_experience(
        self,
        job,
        candidate
    ):
        """
        Score candidate based on years of experience.
        """

        years = candidate.metadata["years_of_experience"]

        min_years = job.constraints["min_years_experience"]
        max_years = job.constraints["max_years_experience"]

        # Candidate is within the ideal range
        if min_years <= years <= max_years:
            return 1.0

        # Candidate has slightly less experience
        elif years < min_years:
            return years / min_years

        # Candidate has more experience
        else:
            excess = years - max_years

            return max(
                0.5,
                1 - (excess * 0.1)
            )
        
    def _score_behavior(
        self,
        candidate
    ):
        """
        Score candidate using behavioral
        signals from Redrob.
        """

        signals = candidate.signals

        score = 0.0
        
        #open to work 
        if signals.get("open_to_work_flag", False):
            score += 0.25       
        
        #Recruiter Response Rate
        score += (
            signals.get("recruiter_response_rate", 0.0)
            * 0.20
        )

        #Notice Period
        notice = signals.get(
            "notice_period_days",
            90
        )

        if notice <= 30:
            score += 0.20

        elif notice <= 60:
            score += 0.14

        else:
            score += 0.08

        #GitHub Activity
        github = signals.get(
            "github_activity_score",
            0
        )

        score += (
            github / 100
        ) * 0.15

        #Saved by Recruiters
        saved = signals.get(
            "saved_by_recruiters_30d",
            0
        )

        score += (
            min(saved / 10, 1.0)
        ) * 0.10

        #Interview Completion
        score += (
            signals.get(
                "interview_completion_rate",
                0
            )
            * 0.05
        )

        #Profile Completeness
        profile = signals.get(
            "profile_completeness_score",
            0
        )

        score += (
            profile / 100
        ) * 0.05

        return score
    
    def rank_candidates(
        self,
        job,
        candidates
    ):
        """
        Compute final ranking score for
        every candidate.
        """

        rankings = []

        for candidate in candidates:    
            evidence_score = self._score_evidence(
                job,
                candidate
            )

            if evidence_score < 0.01:
                continue

            skill_score = self._score_skills(
                job,
                candidate
            )

            experience_score = self._score_experience(
                job,
                candidate
            )

            behavior_score = self._score_behavior(
                candidate
            )

            EVIDENCE_WEIGHT = 0.60
            SKILL_WEIGHT = 0.15
            EXPERIENCE_WEIGHT = 0.10
            BEHAVIOR_WEIGHT = 0.15


            final_score = (
                EVIDENCE_WEIGHT * evidence_score
                + SKILL_WEIGHT * skill_score
                + EXPERIENCE_WEIGHT * experience_score
                + BEHAVIOR_WEIGHT * behavior_score
            )

            # reasoning = self.generate_reasoning(candidate)
            # if evidence_score < 0.15:
            #     final_score *= 0.70
         
            ranking = {
                "evidence_score": evidence_score,
                "skill_score": skill_score,
                "experience_score": experience_score,
            }

            context = build_candidate_context(
                job,
                candidate,
                ranking
            )

            analysis = self.evaluator.evaluate(context)
            rankings.append(
                {
                    "candidate": candidate,
                    "candidate_id": candidate.candidate_id,
                    "final_score": final_score,
                    "evidence_score": evidence_score,
                    "skill_score": skill_score,
                    "experience_score": experience_score,
                    "behavior_score": behavior_score,
                    "analysis": analysis,
                }
            )     

        rankings.sort(
            key=lambda ranking: ranking["final_score"],
            reverse=True
        )
        return rankings    
    
   