#!/usr/bin/env python3
"""Generate the manuscript's main Pearson-r figure from canonical results."""

from pathlib import Path
import json

from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "experiments" / "eacl_paper_canonical_20260728" / "paper_results.json"
OUT = ROOT / "eacl27_paper_staging" / "figures" / "fig_main_model_comparison.png"
FONT_DIR = Path("/System/Library/Fonts/Supplemental")

COLORS = {
    "delay": "#6B7280",
    "quality": "#2E6F9E",
    "structural": "#C06628",
    "full": "#24785A",
    "human": "#1F2937",
}


def font(size: int, bold: bool = False):
    name = "Arial Bold.ttf" if bold else "Arial.ttf"
    return ImageFont.truetype(str(FONT_DIR / name), size)


def main() -> None:
    results = json.loads(RESULTS.read_text(encoding="utf-8"))
    source = results["source_speech_group"]
    loio = results["interpreter_disjoint"]
    rows = [
        (
            "(a) Source-speech-group-held-out",
            [
                ("Segment-onset delay", source["main_models"]["delay_piecewise"]["point"]["pearson"], None, "delay"),
                ("Predicted LQ+EXP + delay", source["seed_metrics"]["auto_pred_LQ_EXP_piecewise_delay"]["pearson"]["mean"], source["seed_metrics"]["auto_pred_LQ_EXP_piecewise_delay"]["pearson"]["sd"], "quality"),
                ("Structural + delay", source["structural_delay"]["pearson"], None, "structural"),
                ("Full model", source["full_model"]["pearson"]["mean"], source["full_model"]["pearson"]["sd"], "full"),
                ("Human quality reference", source["seed_metrics"]["human_LQ_EXP"]["pearson"]["mean"], None, "human"),
            ],
        ),
        (
            "(b) Interpreter-disjoint",
            [
                ("Segment-onset delay", float(loio["delay_piecewise"]["pearson_mean"]), None, "delay"),
                ("Predicted LQ+EXP + delay", float(loio["auto_pred_LQ_EXP_piecewise_delay"]["pearson_mean"]), float(loio["auto_pred_LQ_EXP_piecewise_delay"]["pearson_sd"]), "quality"),
                ("Structural + delay", float(loio["lexical_structural_piecewise_delay"]["pearson_mean"]), None, "structural"),
                ("Full model", float(loio["auto_pred_LQ_EXP_piecewise_delay_lexical_structural"]["pearson_mean"]), float(loio["auto_pred_LQ_EXP_piecewise_delay_lexical_structural"]["pearson_sd"]), "full"),
            ],
        ),
    ]

    image = Image.new("RGB", (1400, 1290), "white")
    draw = ImageDraw.Draw(image)
    x0, x1 = 480, 1320
    label_x = 48
    axis_font = font(25)
    label_font = font(26)
    value_font = font(24, bold=True)
    title_font = font(30, bold=True)

    for panel_index, (title, panel_rows) in enumerate(rows):
        top = 55 + panel_index * 635
        row_start = top + 86
        row_step = 74
        axis_y = row_start + row_step * len(panel_rows) + 2
        draw.text((label_x, top), title, fill="#111827", font=title_font)
        for tick in range(5):
            value = .2 * tick
            x = x0 + int((x1 - x0) * value / .9)
            draw.line((x, row_start - 15, x, axis_y), fill="#E5E7EB", width=2)
            label = f"{value:.1f}"
            box = draw.textbbox((0, 0), label, font=axis_font)
            draw.text((x - (box[2] - box[0]) / 2, axis_y + 12), label, fill="#4B5563", font=axis_font)
        draw.line((x0, axis_y, x1, axis_y), fill="#374151", width=3)
        axis_label = "Pearson r"
        box = draw.textbbox((0, 0), axis_label, font=axis_font)
        draw.text(((x0 + x1 - (box[2] - box[0])) / 2, axis_y + 48), axis_label, fill="#374151", font=axis_font)

        for row_index, (label, value, error, color_key) in enumerate(panel_rows):
            y = row_start + row_step * row_index
            color = COLORS[color_key]
            draw.text((label_x, y - 15), label, fill="#1F2937", font=label_font)
            x = x0 + int((x1 - x0) * value / .9)
            if error is not None:
                delta = int((x1 - x0) * error / .9)
                draw.line((x - delta, y, x + delta, y), fill=color, width=4)
                draw.line((x - delta, y - 9, x - delta, y + 9), fill=color, width=3)
                draw.line((x + delta, y - 9, x + delta, y + 9), fill=color, width=3)
            radius = 9
            if color_key == "human":
                draw.ellipse((x - radius, y - radius, x + radius, y + radius), outline=color, width=4)
            else:
                draw.ellipse((x - radius, y - radius, x + radius, y + radius), fill=color, outline="white", width=2)
            shown = f"{value:.3f}" if error is None else f"{value:.3f} ± {error:.3f}"
            offset = int((x1 - x0) * error / .9) if error is not None else 0
            text_x = x + offset + 20
            text_box = draw.textbbox((text_x, y - 16), shown, font=value_font)
            if text_box[2] > 1380:
                text_x = x - offset - 20 - (text_box[2] - text_box[0])
            draw.text((text_x, y - 16), shown, fill=color, font=value_font)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    image.save(OUT, dpi=(300, 300))


if __name__ == "__main__":
    main()
