import pandas as pd


class DataExplorer:

    @staticmethod
    def dataset_summary(df: pd.DataFrame) -> pd.DataFrame:
        return pd.DataFrame({
            "Column": df.columns,
            "Data Type": df.dtypes.astype(str).values,
            "Missing Values": df.isna().sum().values,
            "Missing %": (df.isna().mean() * 100).round(2).values,
            "Unique Values": df.nunique().values,
        })

    @staticmethod
    def dataset_shape(df: pd.DataFrame):
        print(f"Rows    : {df.shape[0]:,}")
        print(f"Columns : {df.shape[1]}")