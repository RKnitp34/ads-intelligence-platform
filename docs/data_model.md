# Data Model

## Fact Table

interaction_log

Grain

One row = One user interacting with one video at one timestamp.

Primary Entity

(user_id, video_id, time_ms)

---

## Dimensions

User

↓

Video

↓

Video Metadata

---

Relationship

Users (1)
        |
        |
        | user_id
        |
Interactions (N)
        |
        | video_id
        |
Videos (1)
        |
        |
Video Statistics (1)