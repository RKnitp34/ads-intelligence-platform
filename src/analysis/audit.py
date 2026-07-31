import pandas as pd


class RecommendationAudit:

    @staticmethod
    def interaction_summary(df: pd.DataFrame):

        summary = {
            "Total Interactions": len(df),
            "Unique Users": df["user_id"].nunique(),
            "Unique Videos": df["video_id"].nunique(),
            "Date Range": f"{df['date'].min()} - {df['date'].max()}",
            "Click Rate": round(df["is_click"].mean() * 100, 2),
            "Like Rate": round(df["is_like"].mean() * 100, 2),
            "Follow Rate": round(df["is_follow"].mean() * 100, 2),
            "Long View Rate": round(df["long_view"].mean() * 100, 2),
        }

        return pd.DataFrame(summary.items(), columns=["Metric", "Value"])


    @staticmethod
    def user_activity(df):

        return (
            df.groupby("user_id")
              .size()
              .rename("interactions")
              .describe()
              .to_frame()
        )


    @staticmethod
    def video_popularity(df):

        return (
            df.groupby("video_id")
              .size()
              .rename("interactions")
              .describe()
              .to_frame()
        )