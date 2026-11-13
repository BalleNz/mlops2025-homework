"""RAG-конвейер: запрос -> хореография -> LLM -> позы из базы -> кадры -> GIF."""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Callable

from .animation import add_caption, normalize_frames, save_gif
from .llm_client import LLMClient
from .pose_api import PoseAPIClient
from .retriever import Dance, DanceKnowledgeBase, PoseRetriever

DEFAULT_FRAMES = 8


@dataclass
class Frame:
    number: int
    step: str | None
    llm_description: str
    pose_description: str
    similarity: float
    pose_index: int
    pose: dict


@dataclass
class AnimationResult:
    request: str
    dance: str | None
    llm_model: str
    llm_mode: str
    frames: list[Frame]
    gif_path: str
    llm_raw_output: str | None = None
    timings_s: dict[str, float] = field(default_factory=dict)

    def save_json(self, path: str | Path) -> Path:
        path = Path(path)
        text = json.dumps(asdict(self), ensure_ascii=False, indent=2)
        path.write_text(text + "\n", encoding="utf-8")
        return path


def _display_path(path: Path) -> str:
    """Путь относительно текущей папки, чтобы в трассировку не попадали чужие пути."""
    try:
        return str(Path(path).resolve().relative_to(Path.cwd().resolve()))
    except ValueError:
        return Path(path).name


class RagAnimator:
    def __init__(
        self,
        llm: LLMClient,
        retriever: PoseRetriever,
        dances: DanceKnowledgeBase,
        pose_api: PoseAPIClient,
        output_dir: str | Path = "output",
        log: Callable[[str], None] = print,
    ):
        self.llm = llm
        self.retriever = retriever
        self.dances = dances
        self.pose_api = pose_api
        self.output_dir = Path(output_dir)
        self.log = log

    def describe(
        self, request: str, frames: int | None = None
    ) -> tuple[Dance | None, list[str | None], list[str], str | None]:
        """Описания поз от LLM.

        Танец есть в базе хореографии: LLM переводит каждый его шаг в описание позы.
        Танца нет: LLM сама раскладывает движение на позы.
        """
        dance = self.dances.find(request)
        if dance:
            steps: list[str | None] = list(dance.steps)
            if frames:
                steps = (steps * (frames // len(steps) + 1))[:frames]
            self.log(
                f"База хореографии: «{dance.name}», шагов {len(dance.steps)}; "
                f"LLM ({self.llm.model}) описывает позу на каждом шаге"
            )
            descriptions = []
            for i, step in enumerate(steps, start=1):
                descriptions.append(self.llm.rewrite_step(step))
                self.log(f"  {i:>2}. {step} -> {descriptions[-1]}")
            return dance, steps, descriptions, None

        count = frames or DEFAULT_FRAMES
        self.log(
            f"В базе хореографии такого нет: LLM ({self.llm.model}) "
            f"раскладывает движение на {count} поз"
        )
        raw, items = self.llm.plan_poses(request, count)
        if not items:
            raise RuntimeError(f"LLM не вернула список поз, ответ: {raw[:300]!r}")
        if len(items) != count:
            self.log(f"LLM вернула {len(items)} строк вместо {count}, выравниваю")
            items = (items * (count // len(items) + 1))[:count]
        for i, item in enumerate(items, start=1):
            self.log(f"  {i:>2}. {item}")
        return None, [None] * count, items, raw

    def run(
        self,
        request: str,
        output_path: str | Path | None = None,
        frames: int | None = None,
        duration_ms: int = 500,
        captions: bool = True,
    ) -> AnimationResult:
        if not self.pose_api.health():
            raise RuntimeError(
                f"Pose API не отвечает на {self.pose_api.base_url}/health, "
                "запустите его (см. README)"
            )
        timings = {}
        started = time.perf_counter()
        dance, steps, descriptions, raw = self.describe(request, frames)
        timings["llm"] = round(time.perf_counter() - started, 2)

        started = time.perf_counter()
        queries = [
            [text, step] if step else text
            for text, step in zip(descriptions, steps, strict=False)
        ]
        matches = self.retriever.sequence(queries, context=request)
        timings["retrieval"] = round(time.perf_counter() - started, 3)
        self.log(f"Поиск проверенных поз (TF-IDF, {len(self.retriever.records)} поз):")
        for i, (text, match) in enumerate(
            zip(descriptions, matches, strict=False), start=1
        ):
            self.log(f"  {i:>2}. {text} -> {match.description} (sim {match.score:.2f})")

        started = time.perf_counter()
        images = normalize_frames([self.pose_api.render(m.pose) for m in matches])
        if captions:
            total = len(images)
            images = [
                add_caption(image, f"{i}/{total}  {text}", f"база: {match.description}")
                for i, (image, text, match) in enumerate(
                    zip(images, descriptions, matches, strict=False), start=1
                )
            ]
        timings["render"] = round(time.perf_counter() - started, 2)

        slug = dance.slug if dance else "animation"
        gif_path = save_gif(
            images, output_path or self.output_dir / f"{slug}.gif", duration_ms
        )
        self.log(
            f"GIF: {_display_path(gif_path)} ({len(images)} кадров по {duration_ms} мс)"
        )

        result = AnimationResult(
            request=request,
            dance=dance.name if dance else None,
            llm_model=self.llm.model,
            llm_mode="rewrite_steps" if dance else "plan_poses",
            frames=[
                Frame(
                    number=i,
                    step=step,
                    llm_description=text,
                    pose_description=match.description,
                    similarity=round(match.score, 3),
                    pose_index=match.index,
                    pose=match.pose,
                )
                for i, (step, text, match) in enumerate(
                    zip(steps, descriptions, matches, strict=False), start=1
                )
            ],
            gif_path=_display_path(gif_path),
            llm_raw_output=raw,
            timings_s=timings,
        )
        result.save_json(Path(gif_path).with_suffix(".json"))
        return result
