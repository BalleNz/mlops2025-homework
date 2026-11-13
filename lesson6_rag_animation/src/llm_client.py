"""LLM через OpenAI-совместимый API Ollama (шаг 1 урока 6): описания поз по-русски."""

from __future__ import annotations

import re

from openai import OpenAI

ITEM_RE = re.compile(r"^\s*(?:\d+\s*[.):-]|[-*•])\s*(.+?)\s*$")

STEP_SYSTEM_PROMPT = (
    "Ты переводишь шаг танца в короткое описание позы человечка для поиска по базе поз. "
    "Одна фраза на русском: где правая рука, где левая рука, при необходимости ноги "
    "и корпус. Без названия танца, номеров и пояснений."
)
# Примеры не из Макарены, чтобы модель не подглядывала ответ
STEP_EXAMPLES = [
    (
        "Шаг: поднять обе руки высоко над головой и встать на носки",
        "Обе руки подняты вверх, ноги вместе",
    ),
    (
        "Шаг: правую руку упереть в пояс, левую опустить вдоль тела",
        "Правая рука на бедре, левая опущена",
    ),
    (
        "Шаг: присесть, вытянув руки перед собой",
        "Колени согнуты, обе руки вытянуты вперед",
    ),
]

PLAN_SYSTEM_PROMPT = """Ты помогаешь делать анимацию человечка по описанию движения.
Разбей движение на позы и опиши каждую одной короткой фразой на русском: где правая рука, где левая рука, при необходимости ноги и корпус.
Пиши как в примерах: «Обе руки подняты вверх, ноги вместе», «Правая рука на бедре, левая опущена».
Отвечай нумерованным списком, ровно столько строк, сколько попросили. Без пояснений."""


def build_step_messages(step: str) -> list[dict[str, str]]:
    messages = [{"role": "system", "content": STEP_SYSTEM_PROMPT}]
    for question, answer in STEP_EXAMPLES:
        messages.append({"role": "user", "content": question})
        messages.append({"role": "assistant", "content": answer})
    messages.append({"role": "user", "content": f"Шаг: {step}"})
    return messages


def build_plan_prompt(request: str, frames: int) -> str:
    return (
        f"Запрос пользователя: {request}\n"
        f"Разбей это движение на {frames} последовательных поз, ровно {frames} строк."
    )


TRIM = "\"'«»*.;: "


def clean_line(text: str) -> str:
    """Первая непустая строка ответа без кавычек и точки в конце."""
    for line in text.splitlines():
        line = line.strip(TRIM)
        if line:
            return line
    return ""


def parse_items(text: str) -> list[str]:
    """Пункты нумерованного или маркированного списка из ответа модели."""
    items = []
    for line in text.splitlines():
        match = ITEM_RE.match(line)
        if match:
            item = match.group(1).strip(TRIM)
            if item:
                items.append(item)
    return items


class LLMClient:
    def __init__(
        self,
        base_url: str = "http://localhost:11434/v1",
        model: str = "qwen2.5:1.5b",
        timeout: float = 300.0,
        seed: int = 42,
    ):
        self.base_url = base_url
        self.model = model
        self.seed = seed
        self.client = OpenAI(base_url=base_url, api_key="ollama", timeout=timeout)

    def _chat(
        self, messages: list[dict[str, str]], max_tokens: int, temperature: float
    ):
        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            seed=self.seed,
        )
        return response.choices[0].message.content or ""

    def rewrite_step(self, step: str) -> str:
        """Шаг хореографии -> описание позы в словаре базы поз.

        По одному запросу на шаг: маленькая модель (1.5B) на длинном списке
        начинает повторять одну и ту же строку, а на одном шаге работает стабильно.
        """
        answer = clean_line(self._chat(build_step_messages(step), 60, 0.0))
        return answer or step

    def plan_poses(self, request: str, frames: int) -> tuple[str, list[str]]:
        """Описания поз для движения, которого нет в базе хореографии."""
        messages = [
            {"role": "system", "content": PLAN_SYSTEM_PROMPT},
            {"role": "user", "content": build_plan_prompt(request, frames)},
        ]
        raw = self._chat(messages, 40 * frames + 100, 0.3)
        return raw, parse_items(raw)
