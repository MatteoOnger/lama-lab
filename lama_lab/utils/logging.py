import logging
from pathlib import Path


def setup_logger(
    log_path: Path | str | None = None,
    name: str = "ExperimentLogger",
    level: int | str = logging.INFO,
    capture_loggers: list[str] | None = None,
) -> logging.Logger:
    """Sets up experiment logging to console and optionally to a file.

    Parameters
    ----------
    log_path : pathlib.Path or str, optional
        Path to the destination log file. If ``None``, logs are directed
        only to the console (terminal).
    name : str, optional
        Name of the logger instance.
    level : int or str, optional
        Logging threshold level.
    capture_loggers : list of str, optional
        Names of additional :class:`logging.Logger` instances to attach the same
        handlers to (e.g., ``["lama_lab"]``).

    Returns
    -------
    logger : logging.Logger
        Configured main logger instance.
    """
    if isinstance(level, str):
        level = getattr(logging, level.upper(), logging.INFO)

    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(name)s:%(funcName)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    handlers: list[logging.Handler] = []

    # Console handler (always attached)
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    console_handler.setLevel(level)
    handlers.append(console_handler)

    # File handler (attached only if log_path is provided)
    if log_path is not None:
        destination_path = Path(log_path)
        destination_path.parent.mkdir(parents=True, exist_ok=True)

        file_handler = logging.FileHandler(destination_path)
        file_handler.setFormatter(formatter)
        file_handler.setLevel(level)
        handlers.append(file_handler)

    # Define all logger names to attach handlers to
    targets = [name]
    if capture_loggers:
        targets.extend(capture_loggers)

    for logger_name in targets:
        logger = logging.getLogger(logger_name)
        logger.setLevel(level)

        # Properly close existing handlers before clearing to avoid resource leaks
        for handler in logger.handlers:
            handler.close()
        logger.handlers.clear()

        for handler in handlers:
            logger.addHandler(handler)

        # Prevent duplicate log messages in root logger
        logger.propagate = False
    return logging.getLogger(name)
