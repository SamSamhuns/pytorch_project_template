import os
import logging
from unittest.mock import patch
from src.loggers import get_logger


def test_logger_initialization(gen_basic_config, logger_test_dir):
    """Test logger initialization"""
    with patch('os.makedirs') as mocked_makedirs, \
            patch('logging.FileHandler') as mocked_file_handler, \
            patch('logging.StreamHandler') as mocked_stream_handler:
        # Configure MagicMock to return specific values for 'level'
        mocked_file_handler.return_value.level = logging.INFO
        mocked_stream_handler.return_value.level = logging.ERROR

        basic_config = gen_basic_config("test_logger_initialization", logger_test_dir)
        logger = get_logger(**basic_config)

        # Check directory creation
        mocked_makedirs.assert_called_with(
            basic_config["logger_dir"], exist_ok=True)

        # Check that both handlers are added
        assert len(logger.handlers) == 2
        # Check logger level
        assert logger.level == basic_config["logger_level"]

        # Check handlers' levels and formatter types
        assert logger.handlers[0].level == basic_config["file_level"]
        assert logger.handlers[1].level == basic_config["console_level"]


def test_logger_file_creation(gen_basic_config, logger_test_dir):
    """Test logger file creation"""
    with patch('os.makedirs'), \
            patch('logging.FileHandler') as mocked_file_handler:
        basic_config = gen_basic_config("test_logger_file_creation", logger_test_dir)
        get_logger(**basic_config)
        mocked_file_handler.assert_called_with(os.path.join(
            basic_config["logger_dir"], basic_config["logger_name"] + ".txt"))


def test_logger_logging_output(gen_basic_config, logger_test_dir, caplog):
    """Test logger output levels"""
    basic_config = gen_basic_config("test_logger_output", logger_test_dir)
    logger = get_logger(**basic_config)
    debug_msg = "Debug message"
    info_msg = "Info message"
    warning_msg = "Warning message"
    error_msg = "Error message"
    critical_msg = "Critical message"

    logger.debug(debug_msg)
    logger.info(info_msg)
    logger.warning(warning_msg)
    logger.error(error_msg)
    logger.critical(critical_msg)

    # Capture logs and test
    assert debug_msg not in caplog.text
    assert info_msg not in caplog.text
    assert warning_msg in caplog.text
    assert error_msg in caplog.text
    assert critical_msg in caplog.text


def test_logger_no_duplicate_logs(gen_basic_config, logger_test_dir):
    """Test logger output duplication"""
    with patch('os.makedirs'), \
            patch('logging.FileHandler'), \
            patch('logging.StreamHandler'):
        basic_config = gen_basic_config("test_logger_no_duplicate_logs", logger_test_dir)
        logger1 = get_logger(**basic_config)
        # Attempt to create the same logger
        logger2 = get_logger(**basic_config)

        assert logger1 is logger2  # Should be the same logger due to logger's internal caching
        # Should not add more handlers if already initialized
        # each looger has two handlers
        assert len(logger1.handlers) == 2
