"""Logging setup, done once by each entry script"""

import logging


def setup_logging(log_path: str) -> None:
    """Send INFO and above to the console and to log_path.
    Modules then log through logging.getLogger(name), with no handlers of their own.
    A second call does nothing, so log lines are never duplicated.
    """
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[logging.StreamHandler(), logging.FileHandler(log_path)],
    )
