import json
import logging
import threading
from dataclasses import replace
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest

from src.config import AlertsConfig, ConfigError, Threshold, load_config
from src.logger import MetricsWriter
from src.monitor import (
    CRITICAL,
    OK,
    WARNING,
    AlertManager,
    ServiceMonitor,
    classify,
    percentile,
    validate_prediction,
)

PROJECT_DIR = Path(__file__).resolve().parents[1]
CONFIG_PATH = PROJECT_DIR / "config" / "monitoring_config.yaml"


def test_percentile_matches_linear_interpolation():
    values = list(range(1, 101))
    assert percentile(values, 95) == pytest.approx(95.05)
    assert percentile([42.0], 95) == 42.0
    assert percentile([], 95) is None


def test_classify_uses_warning_and_critical_thresholds():
    threshold = Threshold(warning=10, critical=20)
    assert classify(None, threshold) == OK
    assert classify(9.9, threshold) == OK
    assert classify(10, threshold) == WARNING
    assert classify(19.9, threshold) == WARNING
    assert classify(20, threshold) == CRITICAL


def test_validate_prediction():
    good = {
        "success": True,
        "image_size": [612, 408],
        "result": {"prediction": "a dog", "success": True, "timing": {"total_ms": 5}},
    }
    assert validate_prediction(good, (612, 408)) is None
    assert "image_size" in validate_prediction(good, (100, 100))
    failed = {**good, "result": {**good["result"], "success": False}}
    assert "не сгенерировала" in validate_prediction(failed, (612, 408))
    empty = {**good, "result": {**good["result"], "prediction": " "}}
    assert validate_prediction(empty, (612, 408)) == "пустое предсказание"
    assert validate_prediction({"success": False}, (1, 1)) == "success != true"


def test_alert_manager_cooldown_escalation_and_recovery():
    now = [0.0]
    alerts = AlertManager(
        AlertsConfig(enabled=True, cooldown_minutes=5),
        logging.getLogger("test"),
        clock=lambda: now[0],
    )
    assert alerts.update("error_rate_percent", WARNING, "w")["event"] == "alert"
    now[0] = 60
    assert alerts.update("error_rate_percent", WARNING, "w") is None
    assert alerts.update("error_rate_percent", CRITICAL, "c")["status"] == CRITICAL
    now[0] = 60 + 5 * 60
    assert alerts.update("error_rate_percent", CRITICAL, "c") is not None
    recovered = alerts.update("error_rate_percent", OK, "")
    assert recovered["event"] == "recovered"
    assert recovered["previous_status"] == CRITICAL
    assert alerts.update("error_rate_percent", OK, "") is None


def test_load_config_and_validation(tmp_path):
    config = load_config(CONFIG_PATH, base_dir=PROJECT_DIR)
    assert config.service.base_url == "http://localhost:8000"
    assert config.thresholds.p95_latency_ms == Threshold(3000, 6000)
    assert [p.name for p in config.monitoring.images] == ["img.jpg", "img.png"]

    bad = tmp_path / "bad.yaml"
    bad.write_text("thresholds:\n  response_time_ms: {warning: 10, critical: 5}\n")
    with pytest.raises(ConfigError):
        load_config(bad, base_dir=tmp_path)


class FakeService(BaseHTTPRequestHandler):
    """Имитация сервиса шага 2: /health и /predict, режим поломки через флаг."""

    broken = False

    def _send(self, code, payload):
        body = json.dumps(payload).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.broken:
            self._send(503, {"detail": "Модель не загружена"})
        else:
            self._send(200, {"status": "healthy", "model_loaded": True})

    def do_POST(self):
        self.rfile.read(int(self.headers["Content-Length"]))
        if self.broken:
            self._send(500, {"detail": "Ошибка инференса"})
        else:
            result = {"prediction": "a test caption", "success": True}
            result["timing"] = {"total_ms": 1.0}
            self._send(200, {"success": True, "result": result})

    def log_message(self, *args):
        pass


@pytest.fixture
def fake_service():
    server = ThreadingHTTPServer(("127.0.0.1", 0), FakeService)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield server
    FakeService.broken = False
    server.shutdown()


def test_monitor_detects_failures_and_recovery(fake_service, tmp_path):
    config = load_config(CONFIG_PATH, base_dir=PROJECT_DIR)
    config = replace(
        config,
        service=replace(
            config.service, base_url=f"http://127.0.0.1:{fake_service.server_port}"
        ),
        monitoring=replace(config.monitoring, samples_per_check=2, metrics_window=6),
    )
    metrics_file = tmp_path / "metrics.jsonl"
    monitor = ServiceMonitor(
        config, logging.getLogger("test"), MetricsWriter(metrics_file)
    )

    first = monitor.run_check()
    assert first["status"] == OK
    assert first["metrics"]["error_rate_percent"] == 0.0

    FakeService.broken = True
    second = monitor.run_check()
    assert second["statuses"]["health"] == CRITICAL
    assert second["metrics"]["consecutive_failures"] == 3
    assert second["statuses"]["consecutive_failures"] == WARNING
    assert second["metrics"]["error_rate_percent"] == 50.0

    FakeService.broken = False
    monitor.run_check()
    fourth = monitor.run_check()
    assert fourth["status"] == OK
    assert fourth["metrics"]["consecutive_failures"] == 0

    lines = metrics_file.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 4
    assert json.loads(lines[1])["requests"][0]["status_code"] == 503
