import logging
from logging.handlers import RotatingFileHandler
import os


def setup_logging():
    # Detect Lambda
    if os.environ.get("AWS_LAMBDA_FUNCTION_NAME"):
        log_dir = "/tmp/logs"
    else:
        log_dir = "logs"

    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, "app.log")

    handler = RotatingFileHandler(
        log_path, maxBytes=5 * 1024 * 1024, backupCount=3
    )
    formatter = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
    handler.setFormatter(formatter)

    logger = logging.getLogger()
    logger.setLevel(logging.INFO)  # or DEBUG if you want
    # Avoid duplicate handlers if called multiple times
    if not any(isinstance(h, RotatingFileHandler) for h in logger.handlers):
        logger.addHandler(handler)

    # Also log to stdout for Lambda CloudWatch
    console = logging.StreamHandler()
    console.setFormatter(formatter)
    if not any(isinstance(h, logging.StreamHandler) for h in logger.handlers):
        logger.addHandler(console)
