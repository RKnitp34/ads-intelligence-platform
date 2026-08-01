import pandas as pd


class RecommendationAudit:
    """Reusable audit utilities for interaction-log level recommendation analysis."""

    @staticmethod
    def interaction_summary(df: pd.DataFrame) -> pd.DataFrame:
        """Summarize headline interaction metrics for an interaction log.

        Args:
            df: Interaction log containing `user_id`, `video_id`, `is_click`,
                `is_like`, `is_follow`, and `long_view` columns.

        Returns:
            A two-column (`Metric`, `Value`) dataframe with total interactions,
            unique users, unique videos, and click/like/follow/long-view rates
            expressed as percentages.
        """
        summary = {
            "Total Interactions": len(df),
            "Unique Users": df["user_id"].nunique(),
            "Unique Videos": df["video_id"].nunique(),
            "Click Rate": round(df["is_click"].mean() * 100, 2),
            "Like Rate": round(df["is_like"].mean() * 100, 2),
            "Follow Rate": round(df["is_follow"].mean() * 100, 2),
            "Long View Rate": round(df["long_view"].mean() * 100, 2),
        }

        return pd.DataFrame(summary.items(), columns=["Metric", "Value"])

    @staticmethod
    def user_activity(df: pd.DataFrame) -> pd.DataFrame:
        """Compute descriptive statistics of interactions per user.

        Args:
            df: Interaction log containing a `user_id` column.

        Returns:
            Descriptive statistics (count, mean, std, min, quartiles, max)
            of the number of interactions per user.
        """
        return (
            df.groupby("user_id")
            .size()
            .rename("interactions")
            .describe()
            .to_frame()
        )

    @staticmethod
    def video_popularity(df: pd.DataFrame) -> pd.DataFrame:
        """Compute descriptive statistics of interactions per video.

        Args:
            df: Interaction log containing a `video_id` column.

        Returns:
            Descriptive statistics (count, mean, std, min, quartiles, max)
            of the number of interactions per video.
        """
        return (
            df.groupby("video_id")
            .size()
            .rename("interactions")
            .describe()
            .to_frame()
        )

    @staticmethod
    def daily_activity(df: pd.DataFrame) -> pd.DataFrame:
        """Count interactions per day.

        Args:
            df: Interaction log containing a `date` column.

        Returns:
            A dataframe with one row per date, sorted chronologically,
            containing `date` and `interactions` columns.
        """
        return (
            df.groupby("date")
            .size()
            .rename("interactions")
            .reset_index()
            .sort_values("date")
            .reset_index(drop=True)
        )

    @staticmethod
    def hourly_activity(df: pd.DataFrame) -> pd.DataFrame:
        """Count interactions per hour of day.

        The hour is derived from the `hourmin` column (e.g. 2350 -> hour 23)
        rather than requiring a precomputed `hour` column.

        Args:
            df: Interaction log containing an `hourmin` column.

        Returns:
            A dataframe with one row per hour (0-23), sorted by hour,
            containing `hour` and `interactions` columns.
        """
        hour = (df["hourmin"] // 100).rename("hour")

        return (
            hour.to_frame()
            .groupby("hour")
            .size()
            .rename("interactions")
            .reset_index()
            .sort_values("hour")
            .reset_index(drop=True)
        )

    @staticmethod
    def top_users(df: pd.DataFrame, n: int = 20) -> pd.DataFrame:
        """Return the top n most active users by interaction count.

        Args:
            df: Interaction log containing a `user_id` column.
            n: Number of top users to return. Defaults to 20.

        Returns:
            A dataframe with `user_id` and `interactions` columns, sorted
            descending by interaction count.
        """
        return (
            df.groupby("user_id")
            .size()
            .rename("interactions")
            .sort_values(ascending=False)
            .head(n)
            .reset_index()
        )

    @staticmethod
    def top_videos(df: pd.DataFrame, n: int = 20) -> pd.DataFrame:
        """Return the top n most popular videos by interaction count.

        Args:
            df: Interaction log containing a `video_id` column.
            n: Number of top videos to return. Defaults to 20.

        Returns:
            A dataframe with `video_id` and `interactions` columns, sorted
            descending by interaction count.
        """
        return (
            df.groupby("video_id")
            .size()
            .rename("interactions")
            .sort_values(ascending=False)
            .head(n)
            .reset_index()
        )
