# Feature Inventory

## Overview

This document classifies every column in the KuaiRand dataset before feature engineering.

The objective is to:

- Define the role of every column.
- Prevent data leakage.
- Separate labels from model features.
- Maintain a production-ready feature catalog.

---

# Category Definitions

| Category | Description |
|----------|-------------|
| Primary Key | Unique identifier for an entity. |
| Foreign Key | Reference to another entity. |
| Safe Feature | Available before recommendation time and safe for modeling. |
| Candidate Feature | Raw column that requires transformation before use. |
| Label | Target variable(s). |
| Potential Leakage | Contains future information or post-interaction behaviour. |
| Metadata | Useful for joins, ordering or auditing, but not model input. |
| Drop | No modeling value (constant or redundant). |

---

# User Features

| Column | Category | Notes |
|---------|----------|------|
| user_id | Primary Key | User identifier |
| user_active_degree | Safe Feature | User activity bucket |
| is_lowactive_period | Drop | Constant value |
| is_live_streamer | Safe Feature | Binary profile attribute |
| is_video_author | Safe Feature | Binary profile attribute |
| follow_user_num | Candidate Feature | Apply log/bucket transformation |
| fans_user_num | Candidate Feature | Apply log/bucket transformation |
| friend_user_num | Candidate Feature | Apply log/bucket transformation |
| register_days | Candidate Feature | Derive account age |
| follow_user_num_range | Safe Feature | Pre-binned feature |
| fans_user_num_range | Safe Feature | Pre-binned feature |
| friend_user_num_range | Safe Feature | Pre-binned feature |
| register_days_range | Safe Feature | Pre-binned feature |
| onehot_feat0–17 | Safe Feature | Encoded categorical features |

---

# Video Features

| Column | Category | Notes |
|---------|----------|------|
| video_id | Primary Key | Video identifier |
| author_id | Candidate Feature | High-cardinality creator ID |
| music_id | Candidate Feature | High-cardinality music ID |
| video_type | Safe Feature | Static attribute |
| upload_type | Safe Feature | Static attribute |
| music_type | Safe Feature | Static attribute |
| server_width | Safe Feature | Static metadata |
| server_height | Safe Feature | Static metadata |
| video_duration | Safe Feature | Duration before recommendation |
| upload_dt | Candidate Feature | Derive video age |
| tag | Candidate Feature | Encode content category |
| visible_status | Drop | Constant value |

---

# Video Statistics

| Column | Category | Notes |
|---------|----------|------|
| video_id | Foreign Key | Join key |
| All engagement counters (51 columns) | Potential Leakage | Lifetime aggregates must never be joined directly to historical interactions |

Examples include:

- play_cnt
- show_cnt
- like_cnt
- comment_cnt
- follow_cnt
- share_cnt
- collect_cnt
- play_progress
- download_cnt
- report_cnt
- ...

These must later be rebuilt as **point-in-time features**.

---

# Interaction Log

## Keys

| Column | Category |
|---------|----------|
| user_id | Foreign Key |
| video_id | Foreign Key |

---

## Labels

| Column |
|---------|
| is_click |
| is_like |
| is_follow |
| is_comment |
| is_forward |
| is_hate |
| long_view |
| is_profile_enter |

---

## Safe Features

| Column | Notes |
|---------|------|
| tab | Recommendation surface |

---

## Candidate Features

| Column | Notes |
|---------|------|
| date | Derive weekday |
| hourmin | Derive hour-of-day |

---

## Potential Leakage

| Column | Reason |
|---------|--------|
| play_time_ms | Post-interaction watch behaviour |
| profile_stay_time | Post-click behaviour |
| comment_stay_time | Post-click behaviour |

---

## Metadata

| Column | Notes |
|---------|------|
| time_ms | Event ordering |
| duration_ms | Duplicate video duration for validation/debugging |
| is_rand | Exposure policy (standard vs random) |

---

# Modeling Principles

## Never use as model features

- play_time_ms
- profile_stay_time
- comment_stay_time
- Lifetime video statistics

---

## Safe initial feature groups

- User profile
- User activity
- Video metadata
- Upload information
- Device/context (future)
- Time-based features
- Encoded categorical features

---

# Recommendation Pipeline

```
Raw Data
      │
      ▼
Validation
      │
      ▼
Feature Inventory
      │
      ▼
Master Dataset
      │
      ▼
Feature Engineering
      │
      ▼
Candidate Generation
      │
      ▼
Ranking
      │
      ▼
CTR Prediction
```