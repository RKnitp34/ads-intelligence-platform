"""Build the master training dataset."""

import logging

import pandas as pd

from src.config.config import DATA_DIR
from src.data.loader import (
    load_train_logs,
    load_user_features,
    load_video_features_basic,
)

logger = logging.getLogger(__name__)

INTERIM_DATA_DIR = DATA_DIR / "interim"
MASTER_TRAIN_PATH = INTERIM_DATA_DIR / "master_train.parquet"


class RowCountMismatchError(Exception):
    """Raised when a join changes the number of interaction rows."""


def _validate_duplicates(users: pd.DataFrame, videos: pd.DataFrame) -> None:
    """Validate primary keys in dimension tables."""

    if users["user_id"].duplicated().any():
        raise ValueError("Duplicate user_id found in user_features.")

    if videos["video_id"].duplicated().any():
        raise ValueError("Duplicate video_id found in video_features.")


def _validate_row_count(expected: int, actual: int, step: str) -> None:
    """Ensure joins preserve the interaction table."""

    if expected != actual:
        raise RowCountMismatchError(
            f"{step}: expected {expected:,} rows but found {actual:,}."
        )


def build_master_dataset(save: bool = True) -> pd.DataFrame:
    """Build the master dataset."""

    logger.info("Loading datasets...")

    interactions = load_train_logs()
    users = load_user_features()
    videos = load_video_features_basic()

    logger.info(
        "Interactions: %,d | Users: %,d | Videos: %,d",
        len(interactions),
        len(users),
        len(videos),
    )

    _validate_duplicates(users, videos)

    missing_users = (~interactions["user_id"].isin(users["user_id"])).sum()
    missing_videos = (~interactions["video_id"].isin(videos["video_id"])).sum()

    logger.info(
        "Missing Users: %d | Missing Videos: %d",
        missing_users,
        missing_videos,
    )

    expected_rows = len(interactions)

    master = interactions.merge(users, on="user_id", how="left")
    _validate_row_count(expected_rows, len(master), "User Join")

    master = master.merge(videos, on="video_id", how="left")
    _validate_row_count(expected_rows, len(master), "Video Join")

    if save:
        INTERIM_DATA_DIR.mkdir(parents=True, exist_ok=True)
        master.to_parquet(MASTER_TRAIN_PATH, index=False)
        logger.info("Saved: %s", MASTER_TRAIN_PATH)

    logger.info(
        "Master Dataset -> Rows: %,d | Columns: %d | Memory: %.2f MB",
        master.shape[0],
        master.shape[1],
        master.memory_usage(deep=True).sum() / 1024**2,
    )

    return master


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )

    build_master_dataset()