"""Сборка GIF из кадров Pose API: общий кадр, подписи, зацикливание."""

from __future__ import annotations

from pathlib import Path

import matplotlib
from PIL import Image, ImageChops, ImageDraw, ImageFont

FONT_PATH = Path(matplotlib.get_data_path()) / "fonts" / "ttf" / "DejaVuSans.ttf"
CAPTION_HEIGHT = 96
PADDING = 12


def _font(size: int) -> ImageFont.ImageFont:
    try:
        return ImageFont.truetype(str(FONT_PATH), size)
    except OSError:
        return ImageFont.load_default()


def _flatten(image: Image.Image) -> Image.Image:
    """RGBA -> RGB на белом фоне."""
    if image.mode in ("RGBA", "LA", "P"):
        image = image.convert("RGBA")
        background = Image.new("RGB", image.size, "white")
        background.paste(image, mask=image.getchannel("A"))
        return background
    return image.convert("RGB")


def _wrap(draw: ImageDraw.ImageDraw, text: str, font, max_width: int, lines: int):
    """Перенос по словам по ширине в пикселях, не больше lines строк."""
    result, current = [], ""
    for word in text.split():
        candidate = f"{current} {word}".strip()
        if draw.textlength(candidate, font=font) <= max_width or not current:
            current = candidate
        else:
            result.append(current)
            current = word
    if current:
        result.append(current)
    if len(result) > lines:
        result = result[:lines]
        result[-1] = result[-1].rstrip(",.;") + "…"
    return result


def add_caption(image: Image.Image, title: str, text: str) -> Image.Image:
    """Под кадром: номер и описание позы от LLM, ниже найденная поза из базы."""
    canvas = Image.new("RGB", (image.width, image.height + CAPTION_HEIGHT), "white")
    canvas.paste(image, (0, 0))
    draw = ImageDraw.Draw(canvas)
    width = image.width - 2 * PADDING
    y = image.height + 6
    title_font, text_font = _font(15), _font(13)
    for line in _wrap(draw, title, title_font, width, 2):
        draw.text((PADDING, y), line, fill="#222222", font=title_font)
        y += 20
    y += 4
    for line in _wrap(draw, text, text_font, width, 2):
        draw.text((PADDING, y), line, fill="#666666", font=text_font)
        y += 17
    return canvas


def normalize_frames(
    frames: list[Image.Image], margin: int = 30, min_width: int = 440
) -> list[Image.Image]:
    """Все кадры на один холст и обрезка пустых полей по общей рамке.

    Pose API рисует человечка в центре большого белого поля, а размер PNG может
    немного отличаться от кадра к кадру. Рамка общая для всех кадров, поэтому
    человечек не прыгает внутри GIF.
    """
    if not frames:
        return []
    frames = [_flatten(frame) for frame in frames]
    width = max(frame.width for frame in frames)
    height = max(frame.height for frame in frames)
    aligned = []
    for frame in frames:
        canvas = Image.new("RGB", (width, height), "white")
        canvas.paste(frame, ((width - frame.width) // 2, (height - frame.height) // 2))
        aligned.append(canvas)

    white = Image.new("RGB", (width, height), "white")
    boxes = [ImageChops.difference(frame, white).getbbox() for frame in aligned]
    boxes = [box for box in boxes if box]
    if not boxes:
        return aligned
    left = max(0, min(b[0] for b in boxes) - margin)
    top = max(0, min(b[1] for b in boxes) - margin)
    right = min(width, max(b[2] for b in boxes) + margin)
    bottom = min(height, max(b[3] for b in boxes) + margin)
    if right - left < min_width:
        # оставляем место под подписи: расширяем рамку симметрично
        extra = min_width - (right - left)
        left = max(0, left - extra // 2)
        right = min(width, left + min_width)
    return [frame.crop((left, top, right, bottom)) for frame in aligned]


def save_gif(
    frames: list[Image.Image], path: str | Path, duration_ms: int = 500
) -> Path:
    """Сохраняет зацикленный GIF (loop=0)."""
    if not frames:
        raise ValueError("нет кадров для GIF")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    frames[0].save(
        path,
        format="GIF",
        save_all=True,
        append_images=frames[1:],
        duration=duration_ms,
        loop=0,
        optimize=True,
    )
    return path
