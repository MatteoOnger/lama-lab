import logging
from pathlib import Path


def setup_logger(
    log_path: Path,
    name: str = "ExperimentLogger",
    level: int | str = logging.INFO,
    capture_loggers: list[str] | None = None,
) -> logging.Logger:
    """Sets up experiment logging to both console and file.

    Parameters
    ----------
    log_path : Path
        Path to the destination log file.
    name : str, optional
        Name of the logger.
    level : int or str, optional
        Logging threshold level (e.g., logging.INFO, logging.DEBUG, "INFO").
    capture_loggers : list of str, optional
        Names of additional loggers to capture (e.g., ["lama_lab"]).

    Returns
    -------
    logger : logging.Logger
        Main logger instance for the script.
    """
    if isinstance(level, str):
        level = getattr(logging, level.upper(), logging.INFO)

    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(name)s:%(funcName)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Shared Handlers
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    console_handler.setLevel(level)

    file_handler = logging.FileHandler(log_path)
    file_handler.setFormatter(formatter)
    file_handler.setLevel(level)

    # Define all logger names to attach handlers to
    targets = [name]
    if capture_loggers:
        targets.extend(capture_loggers)

    for logger_name in targets:
        logger = logging.getLogger(logger_name)
        logger.setLevel(level)
        if logger.hasHandlers():
            logger.handlers.clear()
        logger.addHandler(console_handler)
        logger.addHandler(file_handler)
    return logging.getLogger(name)
