from aliases import SKILL_ALIASES


class ExplanationGenerator:

    def _evidence_reason(
        self,
        job,
        candidate,
        ranking
    ):
        """
        Generate reasoning based on
        candidate experience.
        """

        keywords = []

        experience_text = " ".join(
            exp.evidence_text.lower()
            for exp in candidate.experiences
        )

        for requirement in job.requirements:

            aliases = SKILL_ALIASES.get(
                requirement,
                [requirement]
            )

            if any(
                alias in experience_text
                for alias in aliases
            ):
                keywords.append(requirement)

        if keywords:
            return (
                "Production experience in "
                + ", ".join(keywords[:4])
                + "."
            )

        return "Relevant production engineering experience."
    
    def _skill_reason(
        self,
        job,
        candidate,
        ranking
    ):
        """
        Generate reasoning from
        candidate skills.
        """

        candidate_skills = [
            skill.name
            for skill in candidate.skills
        ]

        if candidate_skills:
            return (
                "Key skills include "
                + ", ".join(candidate_skills[:5])
                + "."
            )

        return ""
    
    def _experience_reason(
        self,
        candidate
    ):
        """
        Generate reasoning from
        years of experience.
        """

        years = candidate.metadata[
            "years_of_experience"
        ]

        if 5 <= years <= 9:
            return (
                f"{years:.1f} years of experience "
                "matches the preferred range."
            )

        elif years < 5:
            return (
                f"{years:.1f} years of experience "
                "with strong growth potential."
            )

        return (
            f"{years:.1f} years of production "
            "engineering experience."
        )
    
    def _behavior_reason(
        self,
        candidate
    ):
        """
        Generate reasoning from
        behavioral signals.
        """

        signals = candidate.signals

        reasons = []

        if signals.get("open_to_work_flag", False):
            reasons.append("actively seeking opportunities")

        if signals.get("recruiter_response_rate", 0) >= 0.8:
            reasons.append("high recruiter responsiveness")

        if signals.get("github_activity_score", 0) >= 60:
            reasons.append("strong GitHub activity")

        if signals.get("notice_period_days", 90) <= 30:
            reasons.append("short notice period")

        if signals.get("saved_by_recruiters_30d", 0) >= 10:
            reasons.append("frequently shortlisted by recruiters")

        if reasons:
            return (
                "Behavioral indicators show "
                + ", ".join(reasons[:3])
                + "."
            )

        return ""
            
    def generate_reasoning(
        self,
        job,
        candidate,
        ranking
    ):

        parts = [
            self._evidence_reason(
                job,
                candidate,
                ranking
            ),

            self._skill_reason(
                job,
                candidate,
                ranking
            ),

            self._experience_reason(
                candidate
            ),

            self._behavior_reason(
                candidate
            )
        ]

        return " ".join(
            part
            for part in parts
            if part
        )