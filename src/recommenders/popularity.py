"""Popularity-based recommendation baseline."""

import logging
from typing import Optional

import pandas as pd

logger = logging.getLogger(__name__)


class PopularityRecommender:
    """Recommend globally popular videos."""

    def __init__(self) -> None:
        self._popularity: Optional[pd.DataFrame] = None

    def fit(self, df: pd.DataFrame) -> "PopularityRecommender":
        """Compute global video popularity."""

        required_columns = {"video_id"}
        missing = required_columns - set(df.columns)

        if missing:
            raise ValueError(f"Missing required columns: {missing}")

        logger.info("Fitting PopularityRecommender on %,d interactions", len(df))

        popularity = (
            df.groupby("video_id")
            .size()
            .rename("interaction_count")
            .sort_values(ascending=False)
            .reset_index()
        )

        popularity.insert(
            0,
            "rank",
            range(1, len(popularity) + 1),
        )

        self._popularity = popularity

        logger.info(
            "Ranked %,d unique videos",
            len(self._popularity),
        )

        return self

    def recommend(self, n: int = 20) -> pd.DataFrame:
        """Return the top-N most popular videos."""

        self._check_is_fitted()

        return self._popularity.head(n).reset_index(drop=True)

    def recommend_for_user(
        self,
        user_id: int,
        n: int = 20,
    ) -> pd.DataFrame:
        """Return popularity recommendations for a user.

        The user_id is ignored because popularity recommendations
        are identical for every user.
        """

        logger.info("Generating recommendations for user %s", user_id)

        return self.recommend(n)

    def recommend_unseen(
        self,
        user_id: int,
        interactions: pd.DataFrame,
        n: int = 20,
    ) -> pd.DataFrame:
        """Return the top-N popular videos a user has not already interacted with.

        Args:
            user_id: The user to recommend for. Must be present in
                `interactions`.
            interactions: Interaction log containing `user_id` and
                `video_id` columns, used to determine which videos this
                user has already seen.
            n: Number of videos to recommend. Defaults to 20.

        Returns:
            A dataframe with `rank`, `video_id`, and `interaction_count`
            columns for the top-N most popular videos the user has not
            yet interacted with, re-ranked starting from 1.

        Raises:
            RuntimeError: If called before `fit`.
            ValueError: If `user_id` is not present in `interactions`.
        """
        self._check_is_fitted()

        if not (interactions["user_id"] == user_id).any():
            raise ValueError(f"user_id {user_id!r} not found in interactions")

        seen_videos = interactions.loc[
            interactions["user_id"] == user_id, "video_id"
        ].unique()

        logger.info(
            "Generating top-%d unseen recommendations for user %s (%d videos seen)",
            n,
            user_id,
            len(seen_videos),
        )

        unseen = self._popularity[~self._popularity["video_id"].isin(seen_videos)]
        unseen = unseen.head(n).reset_index(drop=True)
        unseen["rank"] = range(1, len(unseen) + 1)

        return unseen[["rank", "video_id", "interaction_count"]]

    def _check_is_fitted(self) -> None:
        """Ensure the recommender has been trained."""

        if self._popularity is None:
            raise RuntimeError(
                "Call fit() before requesting recommendations."
            )