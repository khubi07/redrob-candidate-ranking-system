Observation 1: Skills Distribution

seems realistic
Average = 9.6
Median  = 9
Min     = 5
Max     = 23

Observation 2: Career History

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