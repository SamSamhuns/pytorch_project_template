import subprocess
import sys


def test_setup_logging_writes_each_line_once(tmp_path):
    """A second setup_logging call must not add handlers, which would duplicate every log line.
    It runs in a fresh interpreter, as pytest attaches its own handlers to the root logger.
    """
    log_path = tmp_path / "train.log"
    code = (
        "import logging; from src.loggers import setup_logging; "
        f"setup_logging({str(log_path)!r}); setup_logging({str(log_path)!r}); logging.getLogger('demo').info('hello')"
    )
    subprocess.run([sys.executable, "-c", code], check=True)
    assert log_path.read_text().count("hello") == 1
