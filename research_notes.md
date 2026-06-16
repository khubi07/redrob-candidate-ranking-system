Observation 1: Skills Distribution

Skills may not align with actual work experience.
Implication:
Skills should not be trusted in isolation.
seems realistic
Average = 9.6
Median  = 9
Min     = 5
Max     = 23

Observation 2: Career History

Job titles frequently mismatch career descriptions.
Examples:
Mechanical Engineer → Business Analyst description
Civil Engineer → Brand Designer description
Implication:
Titles alone are unreliable ranking signals.
Average = 3
Median  = 3
Max     = 9
most candidates have: 3 jobs 

Observation 3: Signals
signals are nested. That means we actually have more than 18 signals. We have around 23+ usable features once we flatten everything.

Observation #4

offer_acceptance_rate uses -1 to represent missing data.
Must not be treated as a real acceptance rate.

Observation #5

GitHub activity is sparse.
High GitHub activity may strongly differentiate candidates.

Observation #6

The skill distribution appears template-generated rather than organic.

Evidence:
- Several AI-related skills occur in nearly identical frequencies:
  LLMs (~5094), FAISS (~5052), RAG (~4995), Recommendation Systems (~5091)
- Several ML skills also occur in nearly identical frequencies:
  Python (~1378), PyTorch (~1378), TensorFlow (~1381), NLP (~1358)

Implication:
- Skill frequencies alone are unlikely to be strong ranking signals.
- Exact keyword matching may be unreliable.
- Career descriptions and behavioral signals may carry more useful information than raw skill lists.

Observation #7

Randomly sampled descriptions are realistic and profession-specific.
Examples:
Customer Support
Sales
QA
Android Development
Brand Design

Implication:
The dataset contains diverse professions and many poor fits for the target role.

Observation #8

Relevant AI and search-related terminology exists in the dataset.

Keyword frequencies:
- AI: 178,930
- LLM: 50,478
- ML: 31,364
- Search: 25,245
- Ranking: 623
- Recommendation: 445
- Embedding: 164
- Retrieval: 93

Implication:
Generic AI terminology is common and therefore weak as a ranking signal.
Retrieval, ranking, recommendation, and embedding terminology are comparatively rare and may provide stronger evidence of fit for the target role.

Observation #9

The organizers intentionally permit:
- Semantic Search
- Vector Embeddings
- Hybrid Retrieval
- LLM Ranking

However, challenge constraints remain:
- CPU only
- No hosted APIs
- 100k candidates
- 5 minute ranking budget

Implication:
Architecture decisions should be driven by latency and reproducibility, not by the availability of LLM-based approaches.

A retrieval → ranking pipeline remains the most practical design.