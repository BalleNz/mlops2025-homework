"""RAG-анимация: текстовый запрос -> GIF. Пример: python main.py "танец макарена"."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from src.llm_client import LLMClient
from src.pipeline import RagAnimator
from src.pose_api import PoseAPIClient
from src.retriever import DanceKnowledgeBase, PoseRetriever

PROJECT_DIR = Path(__file__).resolve().parent


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="RAG-система анимаций: запрос -> GIF")
    parser.add_argument("request", nargs="?", default="танец макарена")
    parser.add_argument(
        "--frames", type=int, help="число кадров (по умолчанию из базы)"
    )
    parser.add_argument("--output", type=Path, help="путь к GIF")
    parser.add_argument("--duration", type=int, default=500, help="мс на кадр")
    parser.add_argument("--no-captions", action="store_true", help="без подписей")
    parser.add_argument("--llm-url", default="http://localhost:11434/v1")
    parser.add_argument("--model", default="qwen2.5:1.5b")
    parser.add_argument("--pose-api-url", default="http://localhost:8001")
    parser.add_argument(
        "--poses", type=Path, default=PROJECT_DIR / "data" / "poses_database.json"
    )
    parser.add_argument(
        "--dances", type=Path, default=PROJECT_DIR / "data" / "dances.json"
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    animator = RagAnimator(
        llm=LLMClient(base_url=args.llm_url, model=args.model),
        retriever=PoseRetriever.from_json(args.poses),
        dances=DanceKnowledgeBase.from_json(args.dances),
        pose_api=PoseAPIClient(args.pose_api_url),
        output_dir=PROJECT_DIR / "output",
    )
    output = args.output
    if output is not None and not output.is_absolute():
        output = Path.cwd() / output
    print(f"Запрос: «{args.request}»")
    try:
        result = animator.run(
            args.request,
            output_path=output,
            frames=args.frames,
            duration_ms=args.duration,
            captions=not args.no_captions,
        )
    except RuntimeError as exc:
        print(f"Ошибка: {exc}", file=sys.stderr)
        return 1
    gif = Path(result.gif_path)
    print(f"Готово: {gif}, трассировка: {gif.with_suffix('.json')}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
