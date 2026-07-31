import pandas as pd


class DataProfiler:

    @staticmethod
    def overview(df: pd.DataFrame) -> pd.DataFrame:

        return pd.DataFrame({
            "Rows": [len(df)],
            "Columns": [df.shape[1]],
            "Duplicate Rows": [df.duplicated().sum()],
            "Missing Cells": [df.isna().sum().sum()],
            "Memory (MB)": [round(df.memory_usage(deep=True).sum() / 1024**2, 2)]
        })


    @staticmethod
    def missing_summary(df: pd.DataFrame):

        summary = pd.DataFrame({
            "Missing": df.isna().sum(),
            "Missing %": (df.isna().mean() * 100).round(2)
        })

        return (
            summary
            .query("Missing > 0")
            .sort_values("Missing %", ascending=False)
        )


    @staticmethod
    def cardinality(df: pd.DataFrame):

        return (
            pd.DataFrame({
                "Unique Values": df.nunique(),
                "Unique %": (
                    df.nunique() / len(df) * 100
                ).round(2)
            })
            .sort_values("Unique Values", ascending=False)
        )