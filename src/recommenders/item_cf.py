"""Item-based collaborative filtering recommender."""

import logging
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class ItemBasedCollaborativeFiltering:
    """Item-based collaborative filtering recommender."""

    def __init__(self) -> None:
        self.interaction_matrix: Optional[pd.DataFrame] = None
        self.item_similarity: Optional[pd.DataFrame] = None

    def build_interaction_matrix(self, df: pd.DataFrame) -> pd.DataFrame:
        """Build a binary user-video interaction matrix.

        Args:
            df: Interaction log with `user_id` and `video_id` columns.

        Returns:
            DataFrame indexed by `user_id`, columned by `video_id`; 1
            where an interaction exists, else 0.
        """
        deduped = df.drop_duplicates(subset=["user_id", "video_id"])
        logger.info(
            "Deduplicated interactions: %d -> %d rows", len(df), len(deduped)
        )

        matrix = pd.crosstab(deduped["user_id"], deduped["video_id"])
        matrix = (matrix > 0).astype("int8")

        logger.info(
            "Built interaction matrix: %d users x %d videos", *matrix.shape
        )

        self.interaction_matrix = matrix
        return matrix

    def compute_item_similarity(self, interaction_matrix: pd.DataFrame) -> pd.DataFrame:
        """Compute cosine similarity between every pair of videos.

        Args:
            interaction_matrix: User-video interaction matrix (rows are
                `user_id`, columns are `video_id`).

        Returns:
            Square DataFrame indexed and columned by `video_id`, holding
            cosine similarity values; diagonal is 1.
        """
        logger.info(
            "Computing item similarity for %d videos across %d users",
            interaction_matrix.shape[1],
            interaction_matrix.shape[0],
        )

        values = interaction_matrix.to_numpy(dtype=float)
        norms = np.linalg.norm(values, axis=0)
        norms[norms == 0] = 1.0  # avoid divide-by-zero for videos with no interactions

        normalized = values / norms
        similarity = normalized.T @ normalized
        np.fill_diagonal(similarity, 1.0)

        video_ids = interaction_matrix.columns
        similarity_df = pd.DataFrame(similarity, index=video_ids, columns=video_ids)

        logger.info("Computed %d x %d item similarity matrix", *similarity_df.shape)

        self.item_similarity = similarity_df
        return similarity_df

    def get_similar_videos(self, video_id: int, k: int = 10) -> pd.DataFrame:
        """Return the top-k videos most similar to a given video.

        Args:
            video_id: The video to find similar videos for.
            k: Number of similar videos to return. Defaults to 10.

        Returns:
            DataFrame with `rank`, `video_id`, and `similarity` columns,
            sorted descending by similarity, excluding `video_id` itself.

        Raises:
            RuntimeError: If the similarity matrix has not been computed.
            ValueError: If `video_id` is not found in the similarity matrix.
        """
        if self.item_similarity is None:
            raise RuntimeError(
                "Call compute_item_similarity() before get_similar_videos()."
            )

        if video_id not in self.item_similarity.index:
            raise ValueError(f"video_id {video_id!r} not found in similarity matrix")

        logger.info("Finding top-%d videos similar to video_id=%s", k, video_id)

        similar = (
            self.item_similarity[video_id]
            .drop(index=video_id)
            .sort_values(ascending=False)
            .head(k)
            .rename("similarity")
            .rename_axis("video_id")
            .reset_index()
        )
        similar.insert(0, "rank", range(1, len(similar) + 1))

        logger.info("Found %d similar videos for video_id=%s", len(similar), video_id)

        return similar

    def recommend_for_user(
        self,
        user_id: int,
        interactions: pd.DataFrame,
        n: int = 20,
        k: int = 20,
    ) -> pd.DataFrame:
        """Recommend videos for a user via item-based collaborative filtering.

        For each video the user has watched, retrieves its top-k most
        similar videos, sums similarity scores across watched videos for
        any repeated candidate, excludes already-watched videos, and
        returns the top-n candidates by total similarity.

        Args:
            user_id: The user to recommend for. Must be present in
                `interactions`.
            interactions: Interaction log with `user_id` and `video_id`
                columns, used to determine the user's watch history.
            n: Number of recommendations to return. Defaults to 20.
            k: Number of similar videos to retrieve per watched video.
                Defaults to 20.

        Returns:
            DataFrame with `rank`, `video_id`, and `similarity_score`
            columns, sorted descending by `similarity_score`.

        Raises:
            RuntimeError: If the interaction matrix or similarity matrix
                has not been built/computed.
            ValueError: If `user_id` is not found in `interactions`.
        """
        if self.interaction_matrix is None:
            raise RuntimeError(
                "Call build_interaction_matrix() before recommend_for_user()."
            )

        if self.item_similarity is None:
            raise RuntimeError(
                "Call compute_item_similarity() before recommend_for_user()."
            )

        if not (interactions["user_id"] == user_id).any():
            raise ValueError(f"user_id {user_id!r} not found in interactions")

        watched_videos = interactions.loc[
            interactions["user_id"] == user_id, "video_id"
        ].unique()

        logger.info(
            "Generating item-CF recommendations for user_id=%s "
            "(%d watched videos, k=%d, n=%d)",
            user_id,
            len(watched_videos),
            k,
            n,
        )

        known_watched = [v for v in watched_videos if v in self.item_similarity.index]
        skipped = len(watched_videos) - len(known_watched)
        if skipped:
            logger.info(
                "Skipping %d watched video(s) not present in similarity matrix",
                skipped,
            )

        candidates = [self.get_similar_videos(video_id, k) for video_id in known_watched]

        if not candidates:
            logger.info(
                "No candidates found for user_id=%s; returning empty recommendations",
                user_id,
            )
            return pd.DataFrame(columns=["rank", "video_id", "similarity_score"])

        combined = pd.concat(candidates, ignore_index=True)

        scored = (
            combined.groupby("video_id")["similarity"]
            .sum()
            .rename("similarity_score")
            .reset_index()
        )
        scored = scored[~scored["video_id"].isin(watched_videos)]

        recommendations = (
            scored.sort_values("similarity_score", ascending=False)
            .head(n)
            .reset_index(drop=True)
        )
        recommendations.insert(0, "rank", range(1, len(recommendations) + 1))

        logger.info(
            "Generated %d recommendations for user_id=%s", len(recommendations), user_id
        )

        return recommendations[["rank", "video_id", "similarity_score"]]
