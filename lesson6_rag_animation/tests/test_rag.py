import json
from pathlib import Path

import pytest
from PIL import Image

from src.animation import add_caption, normalize_frames, save_gif
from src.llm_client import (
    build_plan_prompt,
    build_step_messages,
    clean_line,
    parse_items,
)
from src.pipeline import RagAnimator
from src.retriever import DanceKnowledgeBase, PoseRetriever, normalize, pose_key

DATA = Path(__file__).resolve().parents[1] / "data"


@pytest.fixture(scope="module")
def retriever():
    return PoseRetriever.from_json(DATA / "poses_database.json")


@pytest.fixture(scope="module")
def dances():
    return DanceKnowledgeBase.from_json(DATA / "dances.json")


def test_normalize():
    assert (
        normalize("Обе руки: на затылке, Макарена!") == "обе руки на затылке макарена"
    )
    assert normalize("Ёлка") == "елка"


def test_database_has_96_poses(retriever):
    assert len(retriever.records) == 96


@pytest.mark.parametrize(
    "query, expected",
    [
        ("обе руки на затылке", "затылке"),
        ("обе руки на бедрах", "бедрах"),
        ("руки подняты высоко вверх", "вверх"),
        ("корпус смещен вправо", "смещен вправо"),
    ],
)
def test_search_finds_relevant_pose(retriever, query, expected):
    assert expected in retriever.search(query)[0].description.lower()


def test_context_prefers_macarena_poses(retriever):
    plain = retriever.search("обе руки вперед ладони вниз")[0]
    boosted = retriever.search("обе руки вперед ладони вниз", context="танец макарена")
    assert "Макарена" in boosted[0].description
    assert boosted[0].score >= plain.score


def test_sequence_repeat_handling(retriever):
    queries = ["руки вперед", "руки вперед", "руки вперед"]
    strict = retriever.sequence(queries, tolerance=1.01)
    assert len({pose_key(frame.pose) for frame in strict}) == 1
    relaxed = retriever.sequence(queries, tolerance=0.0)
    assert pose_key(relaxed[0].pose) != pose_key(relaxed[1].pose)
    assert pose_key(relaxed[1].pose) != pose_key(relaxed[2].pose)


def test_dance_knowledge_base(dances):
    dance = dances.find("танец макарена")
    assert dance.name == "Макарена"
    assert dance.slug == "macarena"
    assert len(dance.steps) == 12
    assert dances.find("сварить борщ") is None


def test_parse_items_handles_numbered_and_bulleted_lists():
    text = (
        "Вот позы:\n1. Обе руки вверх.\n2) «Правая рука на бедре»\n- Ноги вместе;\n\n"
    )
    assert parse_items(text) == [
        "Обе руки вверх",
        "Правая рука на бедре",
        "Ноги вместе",
    ]


def test_step_prompt_is_few_shot_and_ends_with_the_step():
    messages = build_step_messages("обе руки на бедрах")
    assert messages[0]["role"] == "system"
    assert [m["role"] for m in messages[1:-1]] == ["user", "assistant"] * 3
    assert messages[-1] == {"role": "user", "content": "Шаг: обе руки на бедрах"}
    assert "макарен" not in json.dumps(messages, ensure_ascii=False).lower()


def test_plan_prompt_and_clean_line():
    assert "ровно 6 строк" in build_plan_prompt("приседания", 6)
    assert clean_line("\n«Обе руки на затылке».\nлишнее") == "Обе руки на затылке"


def test_gif_is_looped_and_frames_have_same_size(tmp_path):
    frames = [
        Image.new("RGBA", (100, 120), "red"),
        Image.new("RGBA", (90, 110), "blue"),
    ]
    frames = normalize_frames(frames)
    assert {frame.size for frame in frames} == {(100, 120)}
    frames = [add_caption(frame, "1/2 тест", "подпись") for frame in frames]
    path = save_gif(frames, tmp_path / "test.gif", duration_ms=300)
    with Image.open(path) as gif:
        assert gif.n_frames == 2
        assert gif.info.get("loop") == 0


class FakeLLM:
    model = "fake"

    def rewrite_step(self, step):
        return step


class FakePoseAPI:
    base_url = "http://fake"

    def health(self):
        return True

    def render(self, pose):
        return Image.new("RGBA", (120, 160), "white")


def test_pipeline_without_services(tmp_path, retriever, dances):
    animator = RagAnimator(
        FakeLLM(),
        retriever,
        dances,
        FakePoseAPI(),
        output_dir=tmp_path,
        log=lambda _: None,
    )
    result = animator.run("танец макарена")
    assert result.dance == "Макарена"
    assert len(result.frames) == 12
    assert result.gif_path.endswith("macarena.gif")
    assert (tmp_path / "macarena.gif").exists()
    trace = json.loads((tmp_path / "macarena.json").read_text(encoding="utf-8"))
    assert trace["frames"][9]["pose_description"].startswith("Обе руки на бедрах")
    with Image.open(tmp_path / "macarena.gif") as gif:
        assert gif.n_frames == 12
