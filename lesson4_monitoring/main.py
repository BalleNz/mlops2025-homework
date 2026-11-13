"""Точка входа: мониторинг FastAPI-сервиса инференса ONNX-модели (урок 4, шаг 2)."""

from __future__ import annotations

import argparse
import shlex
import sys
from dataclasses import replace
from pathlib import Path

from src.config import ConfigError, load_config
from src.logger import MetricsWriter, setup_logging
from src.monitor import ServiceMonitor, ServiceProcess

PROJECT_DIR = Path(__file__).resolve().parent


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Мониторинг FastAPI-сервиса инференса ONNX-модели"
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=PROJECT_DIR / "config" / "monitoring_config.yaml",
        help="путь к YAML-конфигу",
    )
    parser.add_argument(
        "--checks",
        type=int,
        help="сколько проверок выполнить (по умолчанию бесконечно, Ctrl+C)",
    )
    parser.add_argument("--interval", type=float, help="пауза между проверками, с")
    parser.add_argument("--samples", type=int, help="запросов к /predict за проверку")
    parser.add_argument("--base-url", help="адрес уже запущенного сервиса")
    parser.add_argument(
        "--start-service",
        action="store_true",
        help="запустить сервис шага 2 самому и остановить после мониторинга",
    )
    parser.add_argument(
        "--service-dir", type=Path, help="папка step2_fastapi_inference"
    )
    parser.add_argument("--service-cmd", help="команда запуска сервиса")
    parser.add_argument("--no-color", action="store_true", help="без цветов в консоли")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        config = load_config(args.config, base_dir=PROJECT_DIR)
    except ConfigError as exc:
        print(f"Ошибка конфигурации: {exc}", file=sys.stderr)
        return 2

    service = config.service
    if args.base_url:
        service = replace(service, base_url=args.base_url.rstrip("/"))
    if args.service_dir:
        service = replace(service, working_dir=args.service_dir.resolve())
    if args.service_cmd:
        service = replace(service, start_command=shlex.split(args.service_cmd))
    monitoring = config.monitoring
    if args.interval is not None:
        monitoring = replace(monitoring, check_interval_seconds=args.interval)
    if args.samples is not None:
        monitoring = replace(monitoring, samples_per_check=args.samples)
    config = replace(config, service=service, monitoring=monitoring)

    logger = setup_logging(config.logging, use_colors=False if args.no_color else None)
    monitor = ServiceMonitor(config, logger, MetricsWriter(config.logging.metrics_file))
    try:
        if args.start_service:
            service_log = config.logging.log_file.parent / "service.log"
            with ServiceProcess(config.service, logger, service_log):
                monitor.run(checks=args.checks)
        else:
            monitor.run(checks=args.checks)
    except KeyboardInterrupt:
        logger.info("Мониторинг остановлен (Ctrl+C)", extra={"event": "stopped"})
    except RuntimeError as exc:
        logger.error(f"Не удалось запустить сервис: {exc}", extra={"event": "error"})
        return 3
    return 0


if __name__ == "__main__":
    sys.exit(main())
