import json

from src.candidate_representation import build_candidate

def load_candidates(file_path: str, limit=None):

    # Store all Candidate objects
    candidates = []

    # Open the dataset
    with open(file_path, "r", encoding="utf-8") as f:

        # Read one candidate at a time
        for line in f:

            # Convert JSON string -> Python dictionary
            candidate_json = json.loads(line)

            # Convert dictionary -> Candidate object
            candidate = build_candidate(candidate_json)

            # Save candidate
            candidates.append(candidate)

            # Stop early during testing
            if limit is not None and len(candidates) >= limit:
                break

    return candidates