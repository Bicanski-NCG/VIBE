# vibe/utils/logger.py
import logging
from contextvars import ContextVar
from contextlib import contextmanager


_INDENT: ContextVar[int] = ContextVar("_INDENT", default=0)


class IndentFilter(logging.Filter):
    """Prepend four spaces per nesting level to every log record."""
    def filter(self, record: logging.LogRecord) -> bool:
        indent = "    " * _INDENT.get()
        record.msg = f"{indent}{record.msg}"
        return True


_logger = logging.getLogger("vibe")
_logger.setLevel(logging.INFO)
_handler = logging.StreamHandler()
_handler.setFormatter(
    logging.Formatter("%(asctime)s | %(levelname)-7s | %(message)s", "%H:%M:%S")
)
_handler.addFilter(IndentFilter())
_logger.addHandler(_handler)
_logger.propagate = False


def info(msg: str, *args, **kwargs):
    _logger.info(msg, *args, **kwargs)


def warning(msg: str, *args, **kwargs):
    _logger.warning(msg, *args, **kwargs)


def error(msg: str, *args, **kwargs):
    _logger.error(msg, *args, **kwargs)


def debug(msg: str, *args, **kwargs):
    _logger.debug(msg, *args, **kwargs)


def critical(msg: str, *args, **kwargs):
    _logger.critical(msg, *args, **kwargs)


def open_step(msg: str):
    _logger.info(msg)
    _INDENT.set(_INDENT.get() + 1)


def close_step(msg: str = "☑️ Done"):
    depth = _INDENT.get()
    if depth == 0:
        _logger.warning("close_step called without matching open_step")
        return
    _logger.info(msg)
    _INDENT.set(depth - 1)


@contextmanager
def step(msg: str, closing_msg: str = "☑️ Done"):
    """Context-manager wrapper so you can write `with logger.step("Epoch"):`."""
    open_step(msg)
    try:
        yield
    finally:
        close_step(closing_msg)