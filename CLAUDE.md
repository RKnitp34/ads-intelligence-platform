# Ads Intelligence Platform

## Project Objective

Build a production-grade Recommendation & Ads Ranking System inspired by companies like Disney Ads, Meta, Google, TikTok, and Amazon.

The objective is to learn and implement an end-to-end recommendation pipeline using real-world datasets.

---

# Development Principles

- Write production-quality code.
- Keep functions small and reusable.
- Prefer readability over clever implementations.
- Avoid duplicate code.
- Use type hints whenever possible.
- Keep notebooks for exploration only.
- Business logic belongs inside `src/`.

---

# Project Structure

data/
    raw/
    interim/
    processed/

docs/

notebooks/

src/
    config/
    data/
    analysis/
    features/
    models/
    evaluation/
    inference/
    pipelines/
    utils/

tests/

dashboard/

api/

---

# Current Dataset

KuaiRand-Pure

Main files:

- user_features_pure.csv
- video_features_basic_pure.csv
- video_features_statistic_pure.csv
- log_standard_4_08_to_4_21_pure.csv
- log_standard_4_22_to_5_08_pure.csv
- log_random_4_22_to_5_08_pure.csv

---

# Roadmap

Phase 1
- Data Understanding

Phase 2
- Data Validation
- Data Profiling

Phase 3
- Recommendation Data Audit

Phase 4
- Feature Engineering

Phase 5
- Popularity Recommender

Phase 6
- Collaborative Filtering

Phase 7
- Candidate Generation

Phase 8
- Learning to Rank

Phase 9
- CTR Prediction

Phase 10
- Streamlit Dashboard

Phase 11
- FastAPI

---

# Coding Rules

- Never hardcode file paths.
- Use config.py for all paths.
- Keep reusable code inside src/.
- Use logging instead of print() in production modules.
- Write docstrings for public functions.
- Ask before introducing new third-party dependencies.

---

# Git

Use Conventional Commits.

Examples:

feat:
fix:
refactor:
docs:
test:
chore:

---

# Before Writing Code

Always:

1. Understand the requirement.
2. Reuse existing modules.
3. Avoid duplicate implementations.
4. Explain major architectural changes before implementing them.

Never rewrite existing modules unless requested.