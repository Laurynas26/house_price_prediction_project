import logging
import os


def setup_logging():
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    # Clear existing handlers
    logger.handlers = []

    # On Lambda → ONLY log to stdout (CloudWatch)
    if os.environ.get("AWS_LAMBDA_FUNCTION_NAME"):
        handler = logging.StreamHandler()
        formatter = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
        handler.setFormatter(formatter)
        logger.addHandler(handler)

    # Local dev → also log to file
    else:
        os.makedirs("logs", exist_ok=True)
        fh = logging.FileHandler("logs/app.log")
        formatter = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
        fh.setFormatter(formatter)
        logger.addHandler(fh)

        # Also console
        console = logging.StreamHandler()
        console.setFormatter(formatter)
        logger.addHandler(console)
