import pandas as pd

def generate_submission(
    job,
    candidates,
    ranker,
    explainer,
    output_path="submission.csv"
):
        rankings = ranker.rank_candidates(
            job,
            candidates
        )

        top_candidates = rankings[:100]

        rows = []

        for rank, result in enumerate(top_candidates, start=1):

            candidate = result["candidate"]

            reasoning = explainer.generate_reasoning(
                job,
                candidate,
                result
            )

            rows.append(
                {
                    "candidate_id": candidate.candidate_id,
                    "rank": rank,
                    "reasoning": reasoning
                }
            )

        df = pd.DataFrame(rows)

        df.to_csv(
            output_path,
            index=False
        )

        return df