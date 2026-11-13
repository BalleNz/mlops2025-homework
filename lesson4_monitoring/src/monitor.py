"""Мониторинг FastAPI-сервиса инференса ONNX-модели из шага 2 урока 4."""

from __future__ import annotations

import io
import logging
import math
import os
import re
import signal
import subprocess
import time
from collections import deque
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Iterable

import requests
from PIL import Image

from .config import AlertsConfig, Config, ServiceConfig, Threshold
from .logger import MetricsWriter, timestamp

OK = "OK"
WARNING = "WARNING"
CRITICAL = "CRITICAL"
SEVERITY = {OK: 0, WARNING: 1, CRITICAL: 2}
LOG_LEVELS = {OK: logging.INFO, WARNING: logging.WARNING, CRITICAL: logging.ERROR}

METRIC_NAMES = {
    "response_time_ms": "response time",
    "p95_latency_ms": "P95 latency",
    "error_rate_percent": "error rate",
    "consecutive_failures": "consecutive failures",
    "health": "health",
}
UNITS = {
    "response_time_ms": " ms",
    "p95_latency_ms": " ms",
    "error_rate_percent": "%",
    "consecutive_failures": "",
}
MIME_TYPES = {
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".png": "image/png",
    ".bmp": "image/bmp",
    ".gif": "image/gif",
    ".webp": "image/webp",
}


def percentile(values: Iterable[float], q: float) -> float | None:
    """Перцентиль с линейной интерполяцией, как numpy.percentile по умолчанию."""
    data = sorted(values)
    if not data:
        return None
    position = (len(data) - 1) * q / 100
    lower, upper = math.floor(position), math.ceil(position)
    return data[lower] + (data[upper] - data[lower]) * (position - lower)


def classify(value: float | None, threshold: Threshold) -> str:
    """OK, WARNING или CRITICAL: порог считается сработавшим при value >= порога."""
    if value is None:
        return OK
    if value >= threshold.critical:
        return CRITICAL
    if value >= threshold.warning:
        return WARNING
    return OK


def worst(statuses: Iterable[str]) -> str:
    return max(statuses, key=SEVERITY.__getitem__, default=OK)


def _fmt(value: float | None, unit: str = "") -> str:
    if value is None:
        return "n/a"
    return f"{value:.0f}{unit}" if unit == " ms" else f"{value:g}{unit}"


@dataclass(frozen=True)
class ProbeImage:
    """Тестовое изображение для запросов к /predict."""

    path: Path
    content: bytes
    mime: str
    size: tuple[int, int]

    @classmethod
    def load(cls, path: str | Path) -> ProbeImage:
        path = Path(path)
        content = path.read_bytes()
        with Image.open(io.BytesIO(content)) as img:
            size = img.size
            fmt = img.format or ""
        mime = MIME_TYPES.get(path.suffix.lower()) or Image.MIME.get(
            fmt, "application/octet-stream"
        )
        return cls(path=path, content=content, mime=mime, size=size)


@dataclass
class RequestResult:
    endpoint: str
    ok: bool
    status_code: int | None
    response_time_ms: float
    image: str | None = None
    prediction: str | None = None
    server_time_ms: float | None = None
    error: str | None = None


def validate_health(payload: Any) -> str | None:
    """Проверяет ответ /health. None, если сервис здоров."""
    if not isinstance(payload, dict):
        return "ответ /health не JSON-объект"
    if payload.get("status") != "healthy":
        return f"status = {payload.get('status')!r}"
    if payload.get("model_loaded") is False:
        return "модель не загружена"
    return None


def validate_prediction(payload: Any, expected_size: tuple[int, int]) -> str | None:
    """Проверяет ответ /predict сервиса шага 2. None, если ответ корректен."""
    if not isinstance(payload, dict):
        return "ответ /predict не JSON-объект"
    if payload.get("success") is not True:
        return "success != true"
    result = payload.get("result")
    if not isinstance(result, dict):
        return "в ответе нет поля result"
    prediction = result.get("prediction")
    if not isinstance(prediction, str) or not prediction.strip():
        return "пустое предсказание"
    if result.get("success") is False:
        return f"модель не сгенерировала описание: {prediction[:80]}"
    total_ms = (result.get("timing") or {}).get("total_ms")
    if not isinstance(total_ms, (int, float)) or total_ms < 0:
        return "нет корректного result.timing.total_ms"
    size = payload.get("image_size")
    if size is not None and list(size) != list(expected_size):
        return f"image_size {list(size)} не совпадает с {list(expected_size)}"
    return None


def _http_error(response: requests.Response) -> str:
    try:
        detail = response.json().get("detail")
    except (ValueError, AttributeError):
        detail = None
    text = detail or response.text[:200] or response.reason
    return f"HTTP {response.status_code}: {text}"


def _connection_error(exc: Exception) -> str:
    text = str(exc)
    match = re.search(r"\[Errno -?\d+\] ([^'\")]+)", text)
    if match:
        return f"нет соединения с сервисом ({match.group(1).strip()})"
    if "RemoteDisconnected" in text or "Connection aborted" in text:
        return "сервис разорвал соединение"
    return f"нет соединения с сервисом ({type(exc).__name__})"


class AlertManager:
    """Алерты по метрикам: смена уровня сразу, повтор того же уровня после cooldown."""

    def __init__(
        self,
        config: AlertsConfig,
        logger: logging.Logger,
        clock: Callable[[], float] = time.monotonic,
    ):
        self.enabled = config.enabled
        self.cooldown_seconds = config.cooldown_minutes * 60
        self.logger = logger
        self.clock = clock
        self.active: dict[str, str] = {}
        self.last_sent: dict[str, float] = {}
        self.sent = 0

    def update(self, metric: str, status: str, message: str) -> dict | None:
        previous = self.active.get(metric)
        if status == OK:
            if previous is None:
                return None
            del self.active[metric]
            self.last_sent.pop(metric, None)
            event = {"metric": metric, "event": "recovered", "status": OK}
            event["previous_status"] = previous
            if self.enabled:
                self.logger.info(
                    f"RECOVERED {METRIC_NAMES[metric]}: снова в норме (был {previous})",
                    extra={**event, "event": "alert_recovered"},
                )
            return event

        now = self.clock()
        last = self.last_sent.get(metric)
        same_level = previous == status
        if same_level and last is not None and now - last < self.cooldown_seconds:
            return None
        self.active[metric] = status
        self.last_sent[metric] = now
        event = {"metric": metric, "event": "alert", "status": status}
        if self.enabled:
            self.sent += 1
            self.logger.log(LOG_LEVELS[status], f"ALERT {message}", extra=event)
        return {**event, "message": message}


class ServiceMonitor:
    """Периодически проверяет /health и /predict, считает метрики и шлёт алерты."""

    def __init__(
        self,
        config: Config,
        logger: logging.Logger,
        metrics_writer: MetricsWriter | None = None,
        session: requests.Session | None = None,
        clock: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], None] = time.sleep,
    ):
        self.config = config
        self.logger = logger
        self.metrics_writer = metrics_writer
        self.session = session or requests.Session()
        self.clock = clock
        self.sleep = sleep
        self.base_url = config.service.base_url
        self.timeout = config.monitoring.request_timeout_seconds
        self.thresholds = config.thresholds.as_dict()
        self.images = [ProbeImage.load(path) for path in config.monitoring.images]
        if not self.images:
            raise ValueError("не задано ни одного тестового изображения")

        window = config.monitoring.metrics_window
        self.latencies: deque[float] = deque(maxlen=window)
        self.outcomes: deque[bool] = deque(maxlen=window)
        self.consecutive_failures = 0
        self.checks_done = 0
        self.alerts = AlertManager(config.alerts, logger, clock)
        self._next_image = 0

        self.total_requests = 0
        self.total_failures = 0
        self.all_latencies: list[float] = []
        self.status_counts = {OK: 0, WARNING: 0, CRITICAL: 0}

    def _request(self, method: str, endpoint: str, **kwargs: Any):
        start = time.perf_counter()
        try:
            response = self.session.request(
                method, f"{self.base_url}{endpoint}", timeout=self.timeout, **kwargs
            )
            return response, (time.perf_counter() - start) * 1000, None
        except requests.Timeout:
            error = f"таймаут {self.timeout:g} с"
        except requests.ConnectionError as exc:
            error = _connection_error(exc)
        except requests.RequestException as exc:
            error = f"ошибка запроса: {exc}"
        return None, (time.perf_counter() - start) * 1000, error

    def _record(self, result: RequestResult) -> RequestResult:
        self.outcomes.append(result.ok)
        self.total_requests += 1
        if result.ok:
            self.consecutive_failures = 0
            if result.endpoint == "/predict":
                self.latencies.append(result.response_time_ms)
                self.all_latencies.append(result.response_time_ms)
        else:
            self.consecutive_failures += 1
            self.total_failures += 1
        self._log_request(result)
        return result

    def check_health(self) -> RequestResult:
        response, elapsed, error = self._request("GET", "/health")
        if response is not None:
            if response.status_code != 200:
                error = _http_error(response)
            else:
                try:
                    error = validate_health(response.json())
                except ValueError:
                    error = "ответ /health не JSON"
        return self._record(
            RequestResult(
                endpoint="/health",
                ok=error is None,
                status_code=response.status_code if response is not None else None,
                response_time_ms=round(elapsed, 1),
                error=error,
            )
        )

    def predict(self, image: ProbeImage) -> RequestResult:
        files = {"file": (image.path.name, image.content, image.mime)}
        response, elapsed, error = self._request("POST", "/predict", files=files)
        prediction = server_ms = None
        if response is not None:
            if response.status_code != 200:
                error = _http_error(response)
            else:
                try:
                    payload = response.json()
                except ValueError:
                    payload = None
                error = validate_prediction(payload, image.size)
                if isinstance(payload, dict) and isinstance(
                    payload.get("result"), dict
                ):
                    prediction = payload["result"].get("prediction")
                    server_ms = (payload["result"].get("timing") or {}).get("total_ms")
        return self._record(
            RequestResult(
                endpoint="/predict",
                ok=error is None,
                status_code=response.status_code if response is not None else None,
                response_time_ms=round(elapsed, 1),
                image=image.path.name,
                prediction=prediction,
                server_time_ms=(
                    round(server_ms, 1) if isinstance(server_ms, (int, float)) else None
                ),
                error=error,
            )
        )

    def _log_request(self, result: RequestResult) -> None:
        method = "GET" if result.endpoint == "/health" else "POST"
        target = f"{method} {result.endpoint}"
        if result.image:
            target += f" {result.image}"
        extra = {"event": "request", **asdict(result)}
        if result.ok:
            text = f"{target} -> {result.status_code}, {result.response_time_ms:.0f} ms"
            if result.server_time_ms is not None:
                text += f" (на сервере {result.server_time_ms:.0f} ms)"
            if result.prediction:
                text += f': "{result.prediction}"'
            self.logger.info(text, extra=extra)
        else:
            text = f"{target} -> ошибка за {result.response_time_ms:.0f} ms: {result.error}"
            self.logger.error(text, extra=extra)

    def _metric_values(self, predictions: list[RequestResult]) -> dict[str, Any]:
        ok_times = [r.response_time_ms for r in predictions if r.ok]
        p95 = percentile(self.latencies, 95)
        failures = self.outcomes.count(False)
        return {
            "response_time_ms": (
                round(sum(ok_times) / len(ok_times), 1) if ok_times else None
            ),
            "p95_latency_ms": round(p95, 1) if p95 is not None else None,
            "error_rate_percent": (
                round(100 * failures / len(self.outcomes), 1) if self.outcomes else 0.0
            ),
            "consecutive_failures": self.consecutive_failures,
        }

    def _alert_message(self, metric: str, value: Any, status: str) -> str:
        threshold = self.thresholds[metric]
        level = "critical" if status == CRITICAL else "warning"
        limit = threshold.critical if status == CRITICAL else threshold.warning
        unit = UNITS[metric]
        return (
            f"{METRIC_NAMES[metric]} {_fmt(value, unit)} >= {level} {_fmt(limit, unit)}"
        )

    def run_check(self) -> dict[str, Any]:
        """Одна проверка: /health, затем samples_per_check запросов к /predict."""
        self.checks_done += 1
        check_id = self.checks_done
        health = self.check_health()
        predictions = []
        for _ in range(self.config.monitoring.samples_per_check):
            image = self.images[self._next_image % len(self.images)]
            self._next_image += 1
            predictions.append(self.predict(image))

        values = self._metric_values(predictions)
        statuses = {
            metric: classify(value, self.thresholds[metric])
            for metric, value in values.items()
        }
        statuses["health"] = OK if health.ok else CRITICAL
        overall = worst(statuses.values())
        self.status_counts[overall] += 1

        failures = self.outcomes.count(False)
        line = (
            f"check #{check_id} | health {'OK' if health.ok else 'FAIL'}"
            f" | response time {_fmt(values['response_time_ms'], ' ms')}"
            f" | p95 {_fmt(values['p95_latency_ms'], ' ms')}"
            f" | errors {values['error_rate_percent']:.1f}%"
            f" ({failures}/{len(self.outcomes)})"
            f" | consecutive failures {values['consecutive_failures']}"
        )
        self.logger.log(
            LOG_LEVELS[overall],
            line,
            extra={
                "event": "check",
                "status": overall,
                "check_id": check_id,
                "metrics": values,
                "statuses": statuses,
            },
        )

        alerts = []
        for metric, value in values.items():
            message = self._alert_message(metric, value, statuses[metric])
            event = self.alerts.update(metric, statuses[metric], message)
            if event:
                alerts.append(event)
        event = self.alerts.update(
            "health", statuses["health"], f"health: {health.error}"
        )
        if event:
            alerts.append(event)

        record = {
            "timestamp": timestamp(),
            "check_id": check_id,
            "status": overall,
            "metrics": values,
            "statuses": statuses,
            "window": {"requests": len(self.outcomes), "failures": failures},
            "requests": [asdict(r) for r in [health, *predictions]],
            "alerts": alerts,
        }
        if self.metrics_writer:
            self.metrics_writer.write(record)
        return record

    def summary(self) -> dict[str, Any]:
        p95 = percentile(self.all_latencies, 95)
        mean = (
            sum(self.all_latencies) / len(self.all_latencies)
            if self.all_latencies
            else None
        )
        rate = (
            100 * self.total_failures / self.total_requests
            if self.total_requests
            else 0
        )
        worst_status = worst(s for s, n in self.status_counts.items() if n)
        return {
            "checks": self.checks_done,
            "requests": self.total_requests,
            "failures": self.total_failures,
            "error_rate_percent": round(rate, 1),
            "mean_latency_ms": round(mean, 1) if mean is not None else None,
            "p95_latency_ms": round(p95, 1) if p95 is not None else None,
            "status_counts": dict(self.status_counts),
            "worst_status": worst_status,
            "alerts_sent": self.alerts.sent,
        }

    def run(self, checks: int | None = None) -> dict[str, Any]:
        """Проверки каждые check_interval_seconds, пока не выполнено checks штук."""
        interval = self.config.monitoring.check_interval_seconds
        names = ", ".join(image.path.name for image in self.images)
        self.logger.info(
            f"Мониторинг {self.base_url}: проверка каждые {interval:g} с, "
            f"запросов к /predict за проверку: "
            f"{self.config.monitoring.samples_per_check}, изображения: {names}",
            extra={"event": "start", "base_url": self.base_url},
        )
        try:
            while checks is None or self.checks_done < checks:
                started = self.clock()
                self.run_check()
                if checks is not None and self.checks_done >= checks:
                    break
                self.sleep(max(0.0, interval - (self.clock() - started)))
        finally:
            result = self.summary()
            counts = result["status_counts"]
            self.logger.info(
                f"Итого: проверок {result['checks']}, запросов {result['requests']}, "
                f"ошибок {result['error_rate_percent']:.1f}%, "
                f"среднее время /predict {_fmt(result['mean_latency_ms'], ' ms')}, "
                f"P95 {_fmt(result['p95_latency_ms'], ' ms')}, "
                f"статусы OK/WARNING/CRITICAL: {counts[OK]}/{counts[WARNING]}/"
                f"{counts[CRITICAL]}, алертов {result['alerts_sent']}",
                extra={"event": "summary", "label": "SUMMARY", **result},
            )
        return result


class ServiceProcess:
    """Запускает сервис шага 2 подпроцессом и ждёт успешного ответа /health."""

    def __init__(
        self,
        service: ServiceConfig,
        logger: logging.Logger,
        log_path: Path,
        poll_seconds: float = 2.0,
    ):
        self.service = service
        self.logger = logger
        self.log_path = Path(log_path)
        self.poll_seconds = poll_seconds
        self.process: subprocess.Popen | None = None
        self._log_file = None

    def __enter__(self) -> ServiceProcess:
        workdir = self.service.working_dir
        if workdir is None or not workdir.is_dir():
            raise RuntimeError(f"папка сервиса не найдена: {workdir}")
        command = self.service.start_command
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        self._log_file = open(self.log_path, "ab")
        self.logger.info(
            f"Запускаю сервис: {' '.join(command)} (папка {workdir}, лог {self.log_path})",
            extra={"event": "service_start", "command": command, "cwd": str(workdir)},
        )
        self.process = subprocess.Popen(
            command,
            cwd=workdir,
            stdout=self._log_file,
            stderr=subprocess.STDOUT,
            start_new_session=os.name != "nt",
        )
        started = time.monotonic()
        url = f"{self.service.base_url}/health"
        while time.monotonic() - started < self.service.startup_timeout_seconds:
            if self.process.poll() is not None:
                self.stop()
                raise RuntimeError(
                    f"сервис завершился с кодом {self.process.returncode}, "
                    f"см. {self.log_path}"
                )
            try:
                if requests.get(url, timeout=5).status_code == 200:
                    self.logger.info(
                        f"Сервис готов через {time.monotonic() - started:.0f} с",
                        extra={"event": "service_ready", "status": OK},
                    )
                    return self
            except requests.RequestException:
                pass
            time.sleep(self.poll_seconds)
        self.stop()
        raise RuntimeError(
            f"сервис не ответил на /health за {self.service.startup_timeout_seconds:g} с"
        )

    def stop(self) -> None:
        if self.process is not None and self.process.poll() is None:
            if os.name == "nt":
                self.process.terminate()
            else:
                os.killpg(self.process.pid, signal.SIGTERM)
            try:
                self.process.wait(timeout=15)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait()
        if self._log_file is not None:
            self._log_file.close()
            self._log_file = None

    def __exit__(self, *exc_info: Any) -> None:
        self.stop()
        self.logger.info("Сервис остановлен", extra={"event": "service_stop"})
