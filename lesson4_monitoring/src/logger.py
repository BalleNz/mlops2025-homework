"""Цветной вывод в консоль, JSON-логи и JSONL-файл метрик."""

from __future__ import annotations

import json
import logging
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .config import LoggingConfig

RESET = "\033[0m"
BOLD = "\033[1m"
DIM = "\033[2m"
GREEN = "\033[32m"
YELLOW = "\033[33m"
RED = "\033[31m"

STATUS_COLORS = {"OK": GREEN, "WARNING": YELLOW, "CRITICAL": RED + BOLD}
LEVEL_COLORS = {
    logging.DEBUG: DIM,
    logging.INFO: "",
    logging.WARNING: YELLOW,
    logging.ERROR: RED,
    logging.CRITICAL: RED + BOLD,
}

# Стандартные атрибуты LogRecord: всё остальное в record.__dict__ пришло через extra
_RESERVED = set(vars(logging.makeLogRecord({}))) | {"message", "asctime"}


def timestamp(created: float | None = None) -> str:
    """Время в ISO 8601 с часовым поясом, до миллисекунд."""
    moment = (
        datetime.fromtimestamp(created, tz=timezone.utc)
        if created is not None
        else datetime.now(timezone.utc)
    )
    return moment.astimezone().isoformat(timespec="milliseconds")


class ConsoleFormatter(logging.Formatter):
    """Строка вида `2025-11-13 20:40:01 [WARNING ] текст` с цветом статуса."""

    def __init__(self, use_colors: bool):
        super().__init__()
        self.use_colors = use_colors

    def format(self, record: logging.LogRecord) -> str:
        moment = datetime.fromtimestamp(record.created).strftime("%Y-%m-%d %H:%M:%S")
        status = getattr(record, "status", None)
        label = getattr(record, "label", None) or status or record.levelname
        line = f"{moment} [{label:<8}] {record.getMessage()}"
        if not self.use_colors:
            return line
        color = STATUS_COLORS.get(status, "") if status else ""
        if getattr(record, "label", None):
            color = BOLD
        color = color or LEVEL_COLORS.get(record.levelno, "")
        return f"{color}{line}{RESET}" if color else line


class JsonFormatter(logging.Formatter):
    """Одна JSON-запись на строку: время, уровень, событие и поля из extra."""

    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "timestamp": timestamp(record.created),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        for key, value in record.__dict__.items():
            if key not in _RESERVED and not key.startswith("_"):
                payload[key] = value
        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)
        return json.dumps(payload, ensure_ascii=False, default=str)


def colors_enabled(config: LoggingConfig, force: bool | None = None) -> bool:
    if force is not None:
        return force
    if not config.console_colors or os.environ.get("NO_COLOR"):
        return False
    return bool(os.environ.get("FORCE_COLOR")) or sys.stdout.isatty()


def setup_logging(
    config: LoggingConfig, use_colors: bool | None = None
) -> logging.Logger:
    """Логгер `monitoring`: цветная консоль плюс JSON-строки в config.log_file."""
    logger = logging.getLogger("monitoring")
    logger.setLevel(logging.INFO)
    logger.propagate = False
    for handler in list(logger.handlers):
        logger.removeHandler(handler)
        handler.close()

    console = logging.StreamHandler(sys.stdout)
    console.setFormatter(ConsoleFormatter(colors_enabled(config, use_colors)))
    logger.addHandler(console)

    config.log_file.parent.mkdir(parents=True, exist_ok=True)
    file_handler = logging.FileHandler(config.log_file, encoding="utf-8")
    file_handler.setFormatter(JsonFormatter())
    logger.addHandler(file_handler)
    return logger


class MetricsWriter:
    """Дописывает по одной JSON-строке на проверку в metrics_file."""

    def __init__(self, path: Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def write(self, record: dict[str, Any]) -> None:
        with open(self.path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")
