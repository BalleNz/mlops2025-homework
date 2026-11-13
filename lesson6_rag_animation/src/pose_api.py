"""Клиент Pose Visualization API (шаг 2 урока 6, порт 8001)."""

from __future__ import annotations

import base64
import io

import requests
from PIL import Image


class PoseAPIError(RuntimeError):
    pass


class PoseAPIClient:
    def __init__(self, base_url: str = "http://localhost:8001", timeout: float = 30.0):
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.session = requests.Session()

    def health(self) -> bool:
        try:
            response = self.session.get(f"{self.base_url}/health", timeout=5)
            return response.status_code == 200
        except requests.RequestException:
            return False

    def render(self, pose: dict) -> Image.Image:
        """PNG-кадр с человечком в позе pose."""
        try:
            response = self.session.post(
                f"{self.base_url}/visualize", json={"pose": pose}, timeout=self.timeout
            )
            response.raise_for_status()
            payload = response.json()
        except (requests.RequestException, ValueError) as exc:
            raise PoseAPIError(
                f"Pose API недоступен или ответил ошибкой: {exc}"
            ) from exc
        if not payload.get("success") or not payload.get("image"):
            raise PoseAPIError(f"Pose API не вернул изображение: {payload}")
        image = Image.open(io.BytesIO(base64.b64decode(payload["image"])))
        image.load()
        return image
