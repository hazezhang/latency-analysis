#!/usr/bin/env python3
"""Generate the EACL cross-fitting pipeline figure."""

from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


OUTPUT = Path(__file__).resolve().parent / "eacl27_paper_staging/figures/fig_method_pipeline.png"


def font(size: int, bold: bool = False):
    candidates = [
        "/System/Library/Fonts/Supplemental/Arial Bold.ttf"
        if bold
        else "/System/Library/Fonts/Supplemental/Arial.ttf",
        "/System/Library/Fonts/Helvetica.ttc",
    ]
    for path in candidates:
        try:
            return ImageFont.truetype(path, size)
        except OSError:
            continue
    return ImageFont.load_default()


def centered(draw, xy, text, text_font, fill=(0, 0, 0), line_gap=4):
    x0, y0, x1, y1 = xy
    lines = text.split("\n")
    heights = [draw.textbbox((0, 0), line, font=text_font)[3] for line in lines]
    total = sum(heights) + line_gap * (len(lines) - 1)
    y = y0 + (y1 - y0 - total) / 2
    for line, height in zip(lines, heights):
        box = draw.textbbox((0, 0), line, font=text_font)
        x = x0 + (x1 - x0 - (box[2] - box[0])) / 2
        draw.text((x, y), line, font=text_font, fill=fill)
        y += height + line_gap


def rect(draw, xy, text, fill, text_font):
    draw.rounded_rectangle(xy, radius=8, fill=fill, outline=(0, 0, 0), width=2)
    centered(draw, xy, text, text_font)


def arrow(draw, start, end):
    draw.line((start, end), fill=(0, 0, 0), width=4)
    x0, _ = start
    x1, y1 = end
    if x1 >= x0:
        points = [(x1, y1), (x1 - 18, y1 - 10), (x1 - 18, y1 + 10)]
    else:
        points = [(x1, y1), (x1 + 18, y1 - 10), (x1 + 18, y1 + 10)]
    draw.polygon(points, fill=(0, 0, 0))


def generate():
    width, height = 2200, 1120
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    title_font, box_font, note_font = font(31, True), font(23), font(21)

    def panel(y0, title, entity, remaining, inner):
        draw.text((35, y0), title, font=title_font, anchor="la", fill=(38, 86, 130))
        train_y, test_y = y0 + 58, y0 + 292
        labels = [
            (35, train_y, 345, train_y + 122, remaining, (232, 244, 253)),
            (450, train_y, 815, train_y + 122, inner, (232, 244, 253)),
            (920, train_y, 1280, train_y + 122, "OOF quality +\nstructure + delay", (223, 243, 228)),
            (1385, train_y, 1735, train_y + 122, "Fit promptness\nregressor", (223, 243, 228)),
            (35, test_y, 345, test_y + 122, entity, (255, 243, 205)),
            (450, test_y, 815, test_y + 122, "Final quality model\nexcludes outer entity", (232, 244, 253)),
            (920, test_y, 1280, test_y + 122, "Outer quality +\nstructure + delay", (223, 243, 228)),
            (1385, test_y, 1735, test_y + 122, "Apply frozen\npromptness regressor", (223, 243, 228)),
            (1840, test_y, 2165, test_y + 122, "Evaluate held-out\npromptness labels", (252, 228, 236)),
        ]
        for x0, box_y0, x1, box_y1, label, fill in labels:
            rect(draw, (x0, box_y0, x1, box_y1), label, fill, box_font)
        for start, end in [
            ((345, train_y + 61), (450, train_y + 61)),
            ((815, train_y + 61), (920, train_y + 61)),
            ((1280, train_y + 61), (1385, train_y + 61)),
            ((345, test_y + 61), (450, test_y + 61)),
            ((815, test_y + 61), (920, test_y + 61)),
            ((1280, test_y + 61), (1385, test_y + 61)),
            ((1735, test_y + 61), (1840, test_y + 61)),
        ]:
            arrow(draw, start, end)
        draw.line(
            (190, train_y + 122, 190, test_y - 22, 635, test_y - 22, 635, test_y),
            fill=(90, 130, 165),
            width=4,
        )
        draw.polygon(
            [(635, test_y), (625, test_y - 18), (645, test_y - 18)],
            fill=(90, 130, 165),
        )
        draw.line((1560, train_y + 122, 1560, test_y), fill=(65, 115, 85), width=4)
        draw.polygon(
            [(1560, test_y), (1550, test_y - 18), (1570, test_y - 18)],
            fill=(65, 115, 85),
        )

    panel(
        25,
        "(a) Speech-disjoint cross-fitting",
        "Held-out speech group",
        "Other 15 speech groups",
        "Inner speech folds\ngenerate OOF quality",
    )
    panel(
        575,
        "(b) Interpreter-disjoint cross-fitting",
        "Held-out interpreter",
        "Other 6 interpreters",
        "Inner speech folds\ngenerate OOF quality",
    )
    draw.text(
        (width / 2, 1090),
        "In both settings, the outer entity is excluded from quality supervision, checkpoint selection, and second-stage promptness training.",
        font=note_font,
        anchor="mm",
        fill=(60, 60, 60),
    )
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    image.save(OUTPUT)


if __name__ == "__main__":
    generate()
