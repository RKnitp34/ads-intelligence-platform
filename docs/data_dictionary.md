# KuaiRand Dataset - Data Dictionary

## Overview

The dataset consists of six files:

| File | Description |
|------|-------------|
| log_standard_4_08_to_4_21_pure.csv | Historical user interactions (Training) |
| log_standard_4_22_to_5_08_pure.csv | Standard recommendation interactions |
| log_random_4_22_to_5_08_pure.csv | Randomly exposed videos for unbiased evaluation |
| user_features_pure.csv | User profile features |
| video_features_basic_pure.csv | Static video metadata |
| video_features_statistic_pure.csv | Historical engagement statistics |

---

# Entity Relationship


```
Users
   │
   │ user_id
   │
Interactions
   │
   │ video_id
   │
Videos
```

---

# 1. Interaction Logs

Primary Key (logical)

```
(user_id, video_id, time_ms)
```

Important columns

| Column | Description |
|---------|-------------|
| user_id | User identifier |
| video_id | Video identifier |
| date | Interaction date |
| hourmin | Interaction time |
| time_ms | Timestamp |
| is_click | Click / Valid play label |
| is_like | User liked video |
| is_follow | User followed author |
| is_comment | Commented |
| is_forward | Shared |
| is_hate | Disliked |
| long_view | Long watch indicator |
| play_time_ms | Watch duration |
| duration_ms | Video duration |
| profile_stay_time | Author profile dwell time |
| comment_stay_time | Comment section dwell time |
| is_profile_enter | Opened author profile |
| is_rand | Random exposure flag |
| tab | App scenario/page |

---

# 2. User Features

Primary Key

```
user_id
```

Contains

- User activity level
- Registration age
- Social graph
- Follow counts
- Fan counts
- Friend counts
- Encrypted categorical features

---

# 3. Video Basic Features

Primary Key

```
video_id
```

Contains

- Author
- Upload date
- Duration
- Resolution
- Music
- Tags
- Video type

---

# 4. Video Statistics

Primary Key

```
video_id
```

Contains historical aggregates

Examples

- Shows
- Plays
- Complete Plays
- Valid Plays
- Long Plays
- Likes
- Comments
- Shares
- Downloads
- Favorites

These should be treated as historical features and checked carefully for leakage depending on the prediction task.

---

# Training Strategy

Historical Logs
↓

Generate User Features

↓

Join User Table

↓

Join Video Metadata

↓

Join Video Statistics

↓

Build Recommendation Dataset

↓

Candidate Generation

↓

Ranking

↓

CTR Prediction