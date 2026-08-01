"""Build the master training dataset for the recommendation/ranking pipeline."""

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
    """Raised when a join unexpectedly changes the interaction fact table's row count."""


def _validate_row_count(expected: int, actual: int, step: str) -> None:
    """Raise RowCountMismatchError if a join step changed the fact table's row count.

    Args:
        expected: Row count before the join.
        actual: Row count after the join.
        step: Human-readable name of the join step, used in the error message.
    """
    if expected != actual:
        raise RowCountMismatchError(
            f"Row count changed during '{step}': expected {expected}, got {actual}. "
            "A left join must preserve every interaction row."
        )
    logger.info("Row count validated after '%s': %d rows", step, actual)


def build_master_dataset(save: bool = True) -> pd.DataFrame:
    """Build the master training dataset.

    The interaction log (train split) is the fact table: one row per
    user-video interaction. User features and video basic features are
    joined onto it as dimensions using LEFT joins, so no interaction row
    is ever dropped. Video statistics are intentionally excluded, since
    they are lifetime-aggregated and not point-in-time safe.

    Row counts are validated before and after every join; any mismatch
    raises RowCountMismatchError. The resulting dataframe is saved to
    data/interim/master_train.parquet.

    Returns:
        The interaction log enriched with user and video basic features,
        one row per original interaction.
    """
    logger.info("Loading train interaction logs")
    interactions = load_train_logs()
    expected_row_count = len(interactions)
    logger.info("Loaded %d interaction rows", expected_row_count)

    logger.info("Loading user features")
    users = load_user_features()

    logger.info("Loading video basic features")
    videos = load_video_features_basic()

    logger.info("Left joining user features on 'user_id'")
    master = interactions.merge(users, on="user_id", how="left")
    _validate_row_count(expected_row_count, len(master), "join user features")

    logger.info("Left joining video basic features on 'video_id'")
    master = master.merge(videos, on="video_id", how="left")
    _validate_row_count(expected_row_count, len(master), "join video features")

    logger.info(
        "Master dataset built: %d rows, %d columns", master.shape[0], master.shape[1]
    )

    INTERIM_DATA_DIR.mkdir(parents=True, exist_ok=True)
    logger.info("Saving master dataset to %s", MASTER_TRAIN_PATH)
    master.to_parquet(MASTER_TRAIN_PATH, index=False)
    logger.info("Saved master dataset to %s", MASTER_TRAIN_PATH)

    return master


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )
    build_master_dataset()
