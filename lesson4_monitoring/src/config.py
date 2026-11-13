"""Загрузка и проверка конфигурации мониторинга."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

DEFAULT_THRESHOLDS = {
    "response_time_ms": {"warning": 2000, "critical": 5000},
    "p95_latency_ms": {"warning": 3000, "critical": 6000},
    "error_rate_percent": {"warning": 10, "critical": 25},
    "consecutive_failures": {"warning": 3, "critical": 5},
}

DEFAULT_START_COMMAND = [
    "poetry",
    "run",
    "uvicorn",
    "src.api:app",
    "--host",
    "127.0.0.1",
    "--port",
    "8000",
]


class ConfigError(ValueError):
    """Ошибка в файле конфигурации."""


@dataclass(frozen=True)
class Threshold:
    warning: float
    critical: float


@dataclass(frozen=True)
class ServiceConfig:
    host: str = "localhost"
    port: int = 8000
    base_url: str = "http://localhost:8000"
    working_dir: Path | None = None
    start_command: list[str] = field(default_factory=lambda: DEFAULT_START_COMMAND)
    startup_timeout_seconds: float = 300.0


@dataclass(frozen=True)
class MonitoringConfig:
    check_interval_seconds: float = 30.0
    samples_per_check: int = 3
    request_timeout_seconds: float = 10.0
    metrics_window: int = 50
    images: list[Path] = field(default_factory=list)


@dataclass(frozen=True)
class ThresholdsConfig:
    response_time_ms: Threshold
    p95_latency_ms: Threshold
    error_rate_percent: Threshold
    consecutive_failures: Threshold

    def as_dict(self) -> dict[str, Threshold]:
        return {
            "response_time_ms": self.response_time_ms,
            "p95_latency_ms": self.p95_latency_ms,
            "error_rate_percent": self.error_rate_percent,
            "consecutive_failures": self.consecutive_failures,
        }


@dataclass(frozen=True)
class AlertsConfig:
    enabled: bool = True
    cooldown_minutes: float = 5.0


@dataclass(frozen=True)
class LoggingConfig:
    console_colors: bool = True
    log_file: Path = Path("logs/monitoring.log")
    metrics_file: Path = Path("logs/metrics.jsonl")


@dataclass(frozen=True)
class Config:
    service: ServiceConfig
    monitoring: MonitoringConfig
    thresholds: ThresholdsConfig
    alerts: AlertsConfig
    logging: LoggingConfig


def _section(raw: dict[str, Any], name: str) -> dict[str, Any]:
    value = raw.get(name) or {}
    if not isinstance(value, dict):
        raise ConfigError(f"секция '{name}' должна быть словарём")
    return value


def _positive(value: Any, name: str, allow_zero: bool = False) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ConfigError(f"{name}: ожидается число, получено {value!r}") from exc
    if number < 0 or (number == 0 and not allow_zero):
        raise ConfigError(f"{name}: значение должно быть больше нуля")
    return number


def _resolve(path: str | Path, base_dir: Path) -> Path:
    path = Path(path).expanduser()
    return path if path.is_absolute() else (base_dir / path).resolve()


def _parse_thresholds(raw: dict[str, Any]) -> ThresholdsConfig:
    parsed = {}
    for metric, defaults in DEFAULT_THRESHOLDS.items():
        values = {**defaults, **(raw.get(metric) or {})}
        warning = _positive(values["warning"], f"thresholds.{metric}.warning", True)
        critical = _positive(values["critical"], f"thresholds.{metric}.critical", True)
        if warning > critical:
            raise ConfigError(
                f"thresholds.{metric}: warning ({warning}) больше critical ({critical})"
            )
        parsed[metric] = Threshold(warning=warning, critical=critical)
    unknown = set(raw) - set(DEFAULT_THRESHOLDS)
    if unknown:
        raise ConfigError(f"неизвестные пороги: {', '.join(sorted(unknown))}")
    return ThresholdsConfig(**parsed)


def load_config(path: str | Path, base_dir: str | Path | None = None) -> Config:
    """Читает YAML и возвращает проверенную конфигурацию.

    Относительные пути (картинки, логи, папка сервиса) считаются от base_dir,
    по умолчанию от папки проекта, в которой лежит config/.
    """
    path = Path(path)
    if not path.exists():
        raise ConfigError(f"файл конфигурации не найден: {path}")
    base = Path(base_dir) if base_dir else path.resolve().parent.parent

    with open(path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}
    if not isinstance(raw, dict):
        raise ConfigError("корень конфигурации должен быть словарём")

    svc = _section(raw, "service")
    host = str(svc.get("host", "localhost"))
    port = int(svc.get("port", 8000))
    base_url = str(svc.get("base_url") or f"http://{host}:{port}").rstrip("/")
    command = svc.get("start_command", DEFAULT_START_COMMAND)
    if isinstance(command, str):
        command = command.split()
    service = ServiceConfig(
        host=host,
        port=port,
        base_url=base_url,
        working_dir=(
            _resolve(svc["working_dir"], base) if svc.get("working_dir") else None
        ),
        start_command=[str(part) for part in command],
        startup_timeout_seconds=_positive(
            svc.get("startup_timeout_seconds", 300), "service.startup_timeout_seconds"
        ),
    )

    mon = _section(raw, "monitoring")
    samples = int(_positive(mon.get("samples_per_check", 3), "samples_per_check"))
    window = int(_positive(mon.get("metrics_window", 50), "metrics_window"))
    monitoring = MonitoringConfig(
        check_interval_seconds=_positive(
            mon.get("check_interval_seconds", 30), "check_interval_seconds", True
        ),
        samples_per_check=samples,
        request_timeout_seconds=_positive(
            mon.get("request_timeout_seconds", 10), "request_timeout_seconds"
        ),
        metrics_window=window,
        images=[_resolve(p, base) for p in mon.get("images", [])],
    )

    alerts_raw = _section(raw, "alerts")
    alerts = AlertsConfig(
        enabled=bool(alerts_raw.get("enabled", True)),
        cooldown_minutes=_positive(
            alerts_raw.get("cooldown_minutes", 5), "alerts.cooldown_minutes", True
        ),
    )

    log_raw = _section(raw, "logging")
    logging_cfg = LoggingConfig(
        console_colors=bool(log_raw.get("console_colors", True)),
        log_file=_resolve(log_raw.get("log_file", "logs/monitoring.log"), base),
        metrics_file=_resolve(log_raw.get("metrics_file", "logs/metrics.jsonl"), base),
    )

    return Config(
        service=service,
        monitoring=monitoring,
        thresholds=_parse_thresholds(_section(raw, "thresholds")),
        alerts=alerts,
        logging=logging_cfg,
    )
