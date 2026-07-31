import pandas as pd

from src.config.config import RAW_DATA_DIR
from src.data.schema import FILES


def load_csv(file_name: str) -> pd.DataFrame:
    return pd.read_csv(RAW_DATA_DIR / file_name)


def load_user_features():
    return load_csv(FILES["user_features"])


def load_video_features_basic():
    return load_csv(FILES["video_features_basic"])


def load_video_features_statistics():
    return load_csv(FILES["video_features_statistic"])


def load_train_logs():
    return load_csv(FILES["train_log"])


def load_test_logs():
    return load_csv(FILES["test_log"])


def load_random_logs():
    return load_csv(FILES["random_log"])