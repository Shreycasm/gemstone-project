import logging
from pathlib import Path
from datetime import datetime

LOG_DIR = Path("logs")
Path.mkdir(LOG_DIR,parents=True,exist_ok=True)

LOG_FILE = f"{datetime.now().strftime('%Y_%m_%d_%H_%M_%S')}.log"
LOG_FILE_PATH = LOG_DIR / LOG_FILE

logging.basicConfig(
    filename=LOG_FILE_PATH,
    format="[%(asctime)s] %(name)s | %(filename)s:%(lineno)d | %(funcName)s | %(levelname)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
)

def get_logger(name: str) -> logging.Logger:

    logger = logging.getLogger(name)
    
  
    if not logger.handlers:
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        formatter = logging.Formatter(
            "[ %(asctime)s ] %(name)s - %(levelname)s - %(message)s"
        )
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
    
    return logger


if __name__ == "__main__":
    logger = get_logger(__name__)
    logger.info("Logger has been configured.")
