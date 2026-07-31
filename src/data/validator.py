import pandas as pd


class DataValidator:

    @staticmethod
    def duplicate_rows(df: pd.DataFrame):
        return df.duplicated().sum()

    @staticmethod
    def duplicate_primary_key(df: pd.DataFrame, cols):
        return df.duplicated(subset=cols).sum()

    @staticmethod
    def missing_values(df: pd.DataFrame):
        return df.isna().sum()