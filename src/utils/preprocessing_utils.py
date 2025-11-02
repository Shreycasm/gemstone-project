import numpy as np
from src.logger import get_logger
from src.exception import SpaceshipTitanicException
import sys

logger = get_logger(__name__)

def clip_outliers_array(X: np.ndarray) -> np.ndarray:
    try:
        logger.info("Clipping Outliers between 1% and 99%.")
        lower = np.quantile(X, 0.01, axis=0)
        upper = np.quantile(X, 0.99, axis=0)
        logger.info("Outliers clipped successfully.")
        return np.clip(X, lower, upper)
    except Exception as e:
        raise SpaceshipTitanicException(e, sys)
