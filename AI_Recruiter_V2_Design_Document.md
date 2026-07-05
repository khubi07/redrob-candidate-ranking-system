AI Recruiter V2
A Hybrid AI-Powered Recruitment Intelligence Platform

Author: Khubi Sahu

Version: 2.0

Status: In Development

1. Vision
Goal

Build an AI Recruiter that mimics how human recruiters shortlist candidates while remaining scalable, explainable, and cost-effective.

Instead of relying only on keyword matching or only on LLM reasoning, the system combines:

Hybrid Retrieval
Semantic Ranking
Structured Feature Engineering
Local LLM Reasoning
Explainable Decisions

The architecture is designed to process thousands of resumes efficiently while providing recruiter-quality reasoning for shortlisted candidates.

2. Problem Statement

Current recruitment systems face a difficult engineering trade-off.

Keyword Search

Advantages

Extremely fast
Cheap
Scalable

Problems

Misses semantic meaning
Cannot understand equivalent technologies
Cannot infer hidden skills

Example

JD

Vector Databases

Resume

Built Retrieval using FAISS

Keyword search often misses this relationship.

LLM-only Recruitment

Advantages

Deep understanding
Human-like reasoning

Problems

Very expensive
High latency
Difficult to scale
Hard to guarantee consistency

Running an LLM over 100,000 resumes is not production-friendly.

Objective

Design a hybrid architecture that combines

Fast retrieval
Semantic understanding
Explainability
Low inference cost
3. Project Philosophy

The system should answer one question:

Can we reduce 100,000 resumes into the best 30 candidates while preserving recruiter-level reasoning?

4. High-Level Architecture
                Job Description
                      │
                      ▼
          Requirement Extraction
                      │
                      ▼
             Hybrid Retrieval
       BM25 + Embedding Search
                      │
                      ▼
          Reciprocal Rank Fusion
                      │
                      ▼
              Top 30 Candidates
                      │
                      ▼
          Structured Feature Engine
                      │
                      ▼
             Local LLM Recruiter
                      │
                      ▼
            Hybrid Score Generator
                      │
                      ▼
              Explainable Ranking
                      │
                      ▼
         Recruiter Dashboard Output
5. Why Hybrid?

Pure BM25

↓

Misses semantics

Pure Embeddings

↓

Misses recruiter reasoning

Pure LLM

↓

Too expensive

Hybrid

↓

Uses each where it performs best.

6. Offline Retrieval Layer

Purpose

Reduce the search space from thousands of resumes into the most promising candidates.

Components

BM25

Captures

Exact skills
Keywords
Certifications
Dense Embeddings

Captures

Semantic similarity
Hidden relationships
Contextual meaning
Reciprocal Rank Fusion

Combines both retrieval methods into a single robust ranking.

Reason

Dense retrieval and sparse retrieval fail in different situations.

RRF combines their strengths.

7. Ranking Layer

Candidate ranking uses four independent signals.

Evidence Score

Measures

How much evidence exists that the candidate actually performed the required work.

Method

Requirement

↓

Compare against every experience

↓

Take Best Match

↓

Weighted Average

This is the most important score.

Skill Score

Measures

Technology overlap

Examples

Python

BM25

FAISS

LLMs

Ranking

Experience Score

Measures

Career depth

Years

Current role

Relevant titles

Career progression

Behavior Score

Uses recruiter signals

Examples

Github activity
Open to work
Recruiter response rate
Notice period
Profile completeness

Final Score

Evidence

+

Skills

+

Experience

+

Behavior
8. Explainability

Instead of

Candidate Score

0.81

The system stores

Requirement

↓

Matched Experience

↓

Evidence

↓

Similarity

Example

Requirement

Ranking Systems

↓

Matched

Recommendation Systems Engineer

↓

Evidence

Built ranking models using XGBoost...

Every decision becomes explainable.

9. Lessons Learned (Version 1)

Major discoveries

Experience-level embeddings outperform whole-resume embeddings.

Capability extraction reduces retrieval noise.

Similarity calibration is required because sentence embeddings produce non-zero similarity even for unrelated text.

Hybrid retrieval performs better than single retrieval methods.

Semantic similarity alone does not imply recruiter fit.

Explainability is as important as ranking quality.

10. Version 2 Architecture

The offline system retrieves Top 30.

A local LLM analyzes only those candidates.

Purpose

Deep reasoning instead of brute-force inference.

Pipeline

Resume

↓

Retriever

↓

Top30

↓

LLM Recruiter

↓

Structured JSON

↓

Hybrid Score

↓

Ranking
11. LLM Responsibilities

The LLM should never search.

The LLM should reason.

Responsibilities

Production experience detection
Leadership detection
Career trajectory analysis
Ownership analysis
Resume quality
Recruiter reasoning
Hidden skill inference
Risk assessment
12. LLM Output Contract

The LLM never returns free-form text.

It returns structured JSON.

Example

{
 "overall_fit": 8,

 "production": true,

 "leadership": 6,

 "career_growth": 9,

 "ownership": 8,

 "strengths": [],

 "weaknesses": [],

 "reasoning": ""
}

Software computes the final score.

The LLM provides evidence.

13. Match Percentage

The percentage is not guessed.

It is computed.

Offline Features

+

LLM Features

↓

Weighted Aggregation

↓

Overall Match

Breakdown

Technical Skills

Experience

Production

Leadership

Career Growth

Behavior

Resume Quality

Every number is explainable.

14. Metrics

Retrieval

Recall@K

Precision@K

MRR

NDCG

Performance

Latency

Embedding time

Ranking latency

LLM latency

Memory

Index size

Candidates/sec

Explainability

Matched experiences

Reason coverage

Evidence coverage

LLM

Consistency

Inference time

Prompt tokens

Hallucination rate

15. Future Roadmap

Version 2

Hybrid AI Recruiter

Version 3

AI Candidate Feedback

Career Improvement Suggestions

Learning Roadmap

Alternative Job Recommendation

Version 4

Recruiter Copilot

Interactive hiring assistant

Natural language querying

Candidate comparison

Interview question generation

16. Tech Stack

Retrieval

BM25
Sentence Transformers
FAISS

Backend

FastAPI

Database

SQLite

LLM

Ollama
Qwen 3
Gemma 3
Llama 3.2

Frontend

Streamlit (prototype)
React (future)

Deployment

Docker
Render
Hugging Face Spaces
17. Interview Story

Problem

Recruitment requires balancing scalability and understanding.

Solution

Hybrid AI pipeline.

Key Contributions

Designed experience-level semantic retrieval.
Built hybrid retrieval using BM25 + dense embeddings + RRF.
Implemented explainable ranking with evidence matching.
Designed a modular AI Recruiter architecture that integrates local LLM reasoning for production-grade candidate evaluation.
18. Engineering Principles

Every AI decision must be

Explainable
Measurable
Modular
Replaceable
Cost-efficient
Production-oriented
19. Core Philosophy

Use classical IR to find candidates. Use an LLM to think like a recruiter. Never use an LLM where deterministic software is more reliable.