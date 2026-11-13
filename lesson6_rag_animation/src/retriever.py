"""TF-IDF поиск: проверенные позы по описанию и хореография по запросу."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

POSE_KEYS = ("Torso", "Head", "RH", "LH", "RK", "LK")


def normalize(text: str) -> str:
    """Нижний регистр, ё -> е, только буквы, цифры и пробелы."""
    text = text.lower().replace("ё", "е")
    text = re.sub(r"[^\w\s]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def _vectorizer() -> TfidfVectorizer:
    # Символьные n-граммы устойчивы к русским окончаниям: рука, руки, руками
    return TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5), sublinear_tf=True)


def pose_key(pose: dict[str, Any]) -> tuple:
    return tuple(tuple(pose[key]) for key in POSE_KEYS)


@dataclass(frozen=True)
class PoseMatch:
    index: int
    score: float
    description: str
    pose: dict[str, list[float]]


class PoseRetriever:
    """Находит в базе проверенную позу, ближайшую к текстовому описанию."""

    def __init__(self, records: list[dict[str, Any]]):
        if not records:
            raise ValueError("база поз пуста")
        for i, record in enumerate(records):
            missing = [k for k in POSE_KEYS if k not in record.get("pose", {})]
            if missing or not record.get("description"):
                raise ValueError(f"поза #{i}: нет полей {missing or ['description']}")
        self.records = records
        self.vectorizer = _vectorizer()
        self.matrix = self.vectorizer.fit_transform(
            [normalize(r["description"]) for r in records]
        )

    @classmethod
    def from_json(cls, path: str | Path) -> PoseRetriever:
        with open(path, "r", encoding="utf-8") as f:
            return cls(json.load(f))

    def scores(
        self, query: str | list[str], context: str | None = None, weight: float = 0.1
    ):
        """Косинусная близость запроса ко всем позам (векторы TF-IDF нормированы).

        Несколько формулировок одного шага (например, ответ LLM и исходный шаг
        хореографии) усредняются: так ошибка одной формулировки меньше влияет на
        выбор позы. context (исходный запрос пользователя) добавляет небольшой
        бонус позам, в описании которых упомянут тот же танец.
        """
        queries = [query] if isinstance(query, str) else list(query)
        vectors = self.vectorizer.transform([normalize(q) for q in queries])
        result = np.asarray((self.matrix @ vectors.T).todense()).mean(axis=1)
        if context:
            extra = self.matrix @ self.vectorizer.transform([normalize(context)]).T
            result = result + weight * extra.toarray().ravel()
        return result

    def search(
        self, query: str | list[str], top_k: int = 5, context: str | None = None
    ) -> list[PoseMatch]:
        scores = self.scores(query, context)
        order = np.argsort(-scores, kind="stable")[:top_k]
        return [
            PoseMatch(
                index=int(i),
                score=float(scores[i]),
                description=self.records[i]["description"],
                pose=self.records[i]["pose"],
            )
            for i in order
        ]

    def sequence(
        self,
        descriptions: list[str | list[str]],
        context: str | None = None,
        avoid_repeats: bool = True,
        tolerance: float = 0.97,
    ) -> list[PoseMatch]:
        """Поза на каждое описание.

        Если лучшая поза совпадает с предыдущим кадром, берём следующую по
        близости, но только если она почти так же хороша (score >= tolerance от
        лучшего): иначе повтор оставляем, он означает паузу в движении.
        """
        result: list[PoseMatch] = []
        previous = None
        for description in descriptions:
            candidates = self.search(description, top_k=8, context=context)
            best = candidates[0]
            if (
                avoid_repeats
                and previous is not None
                and pose_key(best.pose) == previous
            ):
                for candidate in candidates[1:]:
                    if candidate.score < tolerance * best.score:
                        break
                    if pose_key(candidate.pose) != previous:
                        best = candidate
                        break
            result.append(best)
            previous = pose_key(best.pose)
        return result


@dataclass(frozen=True)
class Dance:
    name: str
    slug: str
    steps: list[str]
    score: float


class DanceKnowledgeBase:
    """Небольшая база хореографии: что делать в танце, без координат."""

    def __init__(self, records: list[dict[str, Any]], min_score: float = 0.3):
        self.records = records
        self.min_score = min_score
        self.vectorizer = _vectorizer()
        self.matrix = self.vectorizer.fit_transform(
            [normalize(" ".join([r["name"], *r.get("aliases", [])])) for r in records]
        )

    @classmethod
    def from_json(cls, path: str | Path, min_score: float = 0.3) -> DanceKnowledgeBase:
        with open(path, "r", encoding="utf-8") as f:
            return cls(json.load(f), min_score=min_score)

    def find(self, request: str) -> Dance | None:
        """Танец, о котором просит пользователь, или None, если такого нет в базе."""
        scores = (
            (self.matrix @ self.vectorizer.transform([normalize(request)]).T)
            .toarray()
            .ravel()
        )
        best = int(np.argmax(scores))
        if scores[best] < self.min_score:
            return None
        record = self.records[best]
        return Dance(
            name=record["name"],
            slug=record.get("slug", "animation"),
            steps=list(record["steps"]),
            score=float(scores[best]),
        )
