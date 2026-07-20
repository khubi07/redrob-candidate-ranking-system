from src.pipeline import (
    initialize_pipeline,
    run_pipeline
)
TOP_K_RESULTS = 10
def display_results(rankings):

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

def main():

    retriever, ranker, job = initialize_pipeline()

    rankings = run_pipeline(
        retriever,
        ranker,
        job
    )

    display_results(rankings)

if __name__ == "__main__":
    main()