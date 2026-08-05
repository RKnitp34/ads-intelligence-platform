"""Evaluation utilities for popularity-based recommendation baselines."""

import pandas as pd


class PopularityEvaluation:
    """Evaluation utilities for popularity-based recommendation baselines."""

    @staticmethod
    def top_k_coverage(df: pd.DataFrame, k: int) -> pd.DataFrame:
        """Measure how much of total engagement the top-k videos account for.

        Args:
            df: Interaction log containing a `video_id` column, one row
                per interaction.
            k: Number of top videos (by interaction count) to evaluate.

        Returns:
            A two-column (`Metric`, `Value`) dataframe with the total
            interactions accounted for by the top-k videos, that total as
            a percentage of all interactions, and the top-k videos'
            coverage of the overall video catalog, as a percentage.
        """
        video_counts = df.groupby("video_id").size().sort_values(ascending=False)

        total_interactions = len(df)
        total_videos = video_counts.shape[0]
        top_k_interactions = int(video_counts.head(k).sum())

        summary = {
            "Top K Interactions": top_k_interactions,
            "Percentage of Total Interactions": round(
                top_k_interactions / total_interactions * 100, 2
            ),
            "Coverage %": round(min(k, total_videos) / total_videos * 100, 2),
        }

        return pd.DataFrame(summary.items(), columns=["Metric", "Value"])

    @staticmethod
    def cumulative_popularity(df: pd.DataFrame) -> pd.DataFrame:
        """Build a cumulative popularity curve over videos, ranked by interactions.

        Args:
            df: Interaction log containing a `video_id` column, one row
                per interaction.

        Returns:
            A dataframe with one row per video, ranked descending by
            interaction count, containing `rank`, `video_id`,
            `interactions`, `cumulative_interactions`,
            `cumulative_interaction_pct`, and `cumulative_video_pct`
            columns. Useful for plotting or inspecting how quickly
            interactions concentrate in a small share of the catalog.
        """
        popularity = (
            df.groupby("video_id")
            .size()
            .rename("interactions")
            .sort_values(ascending=False)
            .reset_index()
        )

        total_interactions = popularity["interactions"].sum()
        total_videos = len(popularity)

        popularity.insert(0, "rank", range(1, total_videos + 1))
        popularity["cumulative_interactions"] = popularity["interactions"].cumsum()
        popularity["cumulative_interaction_pct"] = round(
            popularity["cumulative_interactions"] / total_interactions * 100, 2
        )
        popularity["cumulative_video_pct"] = round(
            popularity["rank"] / total_videos * 100, 2
        )

        return popularity
