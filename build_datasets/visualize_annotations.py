#!/usr/bin/env python3
"""Visualize dataset_raw.jsonl — image center, object cards surrounding it.

Layout:
    ┌──────────────────── title bar ─────────────────────┐
    │ [card L1] │       IMAGE        │ [card R1]          │
    │ [card L2] │   tags + scene     │ [card R2]          │
    │ [card L3] │                    │                    │
    └────────────────────────────────────────────────────┘

Usage:
    python visualize_annotations.py \\
        --jsonl /path/to/dataset_raw.jsonl \\
        --image_dir /path/to/images \\
        --output_dir ./vis_out --n 5
"""

from __future__ import annotations

import argparse
import json
import math
import textwrap
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from PIL import Image

matplotlib.use("Agg")

# ─────────────────────────────────── palette ─────────────────────────────────

_CARD_THEMES: List[Tuple[str, str, str]] = [
    ("#3A6BC9", "#EDF3FD", "#2A52A0"),
    ("#D4763B", "#FDF1E8", "#A85520"),
    ("#3D9C5E", "#EAFAF2", "#276B3E"),
    ("#B84040", "#FDEAEA", "#8B2626"),
    ("#7155B0", "#F2EDFC", "#513C8C"),
    ("#B09030", "#FCF8E8", "#856C1A"),
    ("#2E90AC", "#E7F6FB", "#1A6B80"),
    ("#8B6244", "#F6EDE4", "#634428"),
]

_TAG_COLOURS = ["#3A6BC9", "#3D9C5E", "#B84040", "#7155B0",
                "#D4763B", "#2E90AC", "#B09030", "#8B6244"]

_FIELDS: List[Tuple[str, str, str]] = [
    ("positive_texts",           "Positive", "#1B7A2F"),
    ("weak_positives",           "Weak +",   "#6A7A00"),
    ("hard_negatives_attribute", "HN attr",  "#B52020"),
    ("hard_negatives_scene",     "HN scene", "#9E1060"),
    ("relational_texts",         "Spatial",  "#1040A8"),
    ("ocr_texts",                "OCR",      "#C04000"),
]

# ─────────────────────────────────── params ──────────────────────────────────

_FS            = 10               # base font size (pt)
_FONT          = "DejaVu Sans Mono"
_WRAP_W        = 30               # value wrap width (chars)
_LBL_W         = 8                # label column width (chars, right-aligned)

_FIG_W         = 22.0             # total figure width (inches)
_CENTER_FRAC   = 0.40             # centre column width fraction
_SIDE_FRAC     = (1.0 - _CENTER_FRAC) / 2.0

# Card internal columns (axes-fraction x positions)
_X_LBL_R       = 0.27            # label right edge (right-aligned)
_X_BULL        = 0.29            # bullet centre
_X_VAL         = 0.33            # value left edge  (~67 % of card for text)

_LINE_H        = _FS * 1.65 / 72  # inches per text line
_HDR_H         = 0.42             # card header (inches)
_CARD_PAD      = 0.18             # card top+bottom padding (inches)
_MIN_CARD_H    = 1.6              # minimum card height (inches)
_TITLE_H       = 0.42             # title bar height (inches)
_INFO_H        = 2.8              # info panel fixed height (inches)
_MAX_FIG_H     = 16.0             # cap figure height (inches)


# ─────────────────────────────── sizing ──────────────────────────────────────

def _card_lines(obj: Dict[str, Any]) -> int:
    n = 0
    for key, *_ in _FIELDS:
        vals = obj.get(key, [])
        if not vals:
            continue
        n += 1  # one label-separator row per field
        for v in vals:
            n += len(textwrap.wrap(v, _WRAP_W) or [""])
    return max(n, 1)


def _card_h(obj: Dict[str, Any]) -> float:
    return max(_card_lines(obj) * _LINE_H + _HDR_H + _CARD_PAD, _MIN_CARD_H)


# ─────────────────────────────────── card ────────────────────────────────────

def _draw_card(ax: plt.Axes, obj: Dict[str, Any],
               theme: Tuple[str, str, str]) -> None:
    hdr_col, body_col, accent = theme
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    # drop-shadow
    ax.add_patch(mpatches.FancyBboxPatch(
        (0.025, 0.005), 0.965, 0.965,
        boxstyle="round,pad=0.015",
        facecolor="#00000020", edgecolor="none",
        transform=ax.transAxes, zorder=0,
    ))
    # card body
    ax.add_patch(mpatches.FancyBboxPatch(
        (0.015, 0.018), 0.970, 0.968,
        boxstyle="round,pad=0.015",
        facecolor=body_col, edgecolor=accent, linewidth=1.0,
        transform=ax.transAxes, zorder=1,
    ))

    name     = obj.get("object_name", "?")
    salience = obj.get("salience", "secondary")
    sal_icon = "★" if salience == "primary" else "◇"

    HDR_TOP  = 0.986 - 0.015          # top of header (axes frac)
    HDR_FRAC = 0.105                   # header height fraction

    ax.add_patch(mpatches.FancyBboxPatch(
        (0.015, HDR_TOP - HDR_FRAC), 0.970, HDR_FRAC,
        boxstyle="round,pad=0.010",
        facecolor=hdr_col, edgecolor="none",
        transform=ax.transAxes, zorder=2,
    ))
    ax.text(0.048, HDR_TOP - HDR_FRAC / 2,
            f"{sal_icon}  {name.upper()}",
            transform=ax.transAxes, va="center", ha="left",
            fontsize=_FS + 1.5, fontweight="bold",
            color="white", fontfamily=_FONT, zorder=3)
    ax.text(0.960, HDR_TOP - HDR_FRAC / 2,
            salience,
            transform=ax.transAxes, va="center", ha="right",
            fontsize=_FS - 2, color="#FFFFFF99",
            fontfamily=_FONT, zorder=3)

    # collect rows: (label, bullet, value, bullet_colour)
    rows: List[Tuple[str, str, str, str]] = []
    for key, label, bcol in _FIELDS:
        vals = obj.get(key, [])
        if not vals:
            continue
        first_field = True
        for v in vals:
            wrapped = textwrap.wrap(v, _WRAP_W) or [""]
            for j, line in enumerate(wrapped):
                lbl  = label.rjust(_LBL_W) if (first_field and j == 0) else ""
                bull = "•" if j == 0 else " "
                rows.append((lbl, bull, line, bcol))
            first_field = False

    if not rows:
        return

    y_top = HDR_TOP - HDR_FRAC - 0.025
    y_bot = 0.030
    rh    = (y_top - y_bot) / len(rows)

    for i, (lbl, bull, text, bcol) in enumerate(rows):
        ym   = y_top - (i + 0.5) * rh
        yrow = y_top - (i + 1) * rh

        if i % 2 == 1:
            ax.add_patch(mpatches.FancyBboxPatch(
                (0.020, yrow), 0.960, rh,
                boxstyle="square,pad=0",
                facecolor="#0000000B", edgecolor="none",
                transform=ax.transAxes, zorder=2,
            ))

        if lbl:
            ax.text(_X_LBL_R, ym, lbl,
                    transform=ax.transAxes, va="center", ha="right",
                    fontsize=_FS - 1.5, color="#999999",
                    fontfamily=_FONT, zorder=3)

        ax.text(_X_BULL, ym, bull,
                transform=ax.transAxes, va="center", ha="center",
                fontsize=_FS + 1, color=bcol,
                fontfamily=_FONT, zorder=3)

        ax.text(_X_VAL, ym, text,
                transform=ax.transAxes, va="center", ha="left",
                fontsize=_FS, color="#111111",
                fontfamily=_FONT, zorder=3)


# ─────────────────────────────── info panel ──────────────────────────────────

def _draw_info(ax: plt.Axes, record: Dict[str, Any]) -> None:
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    ax.add_patch(mpatches.FancyBboxPatch(
        (0.018, 0.015), 0.964, 0.978,
        boxstyle="round,pad=0.012",
        facecolor="white", edgecolor="#DDDDDD", linewidth=0.8,
        transform=ax.transAxes,
    ))

    tags         = record.get("challenge_tags", [])
    global_texts = record.get("global_texts", [])

    # ── challenge tags section ────────────────────────────────────────────
    ax.text(0.05, 0.95, "Challenge Tags",
            transform=ax.transAxes, va="top", ha="left",
            fontsize=_FS, fontweight="bold", color="#333333",
            fontfamily=_FONT)

    # Draw tag chips in a flow layout
    # Convert axes coords to figure inches to estimate chip widths properly.
    # info panel width ≈ _FIG_W * _CENTER_FRAC inches
    PANEL_W_IN  = _FIG_W * _CENTER_FRAC * 0.95  # usable width in inches
    CHIP_H_IN   = 0.22                            # chip height in inches
    CHIP_PAD_IN = 0.08                            # horizontal gap between chips
    CHAR_W_IN   = (_FS - 2) * 0.55 / 72          # approx char width at (FS-2)pt

    chip_ax_h = CHIP_H_IN / _INFO_H              # chip height in axes fraction
    gap_ax    = CHIP_PAD_IN / _INFO_H

    # Start below the "Challenge Tags" header
    y_chip = 0.95 - (0.05 + chip_ax_h)
    x      = 0.05
    row_start_y = y_chip

    for ti, tag in enumerate(tags):
        col   = _TAG_COLOURS[ti % len(_TAG_COLOURS)]
        label = tag.replace("_", " ")
        chip_w_in = len(label) * CHAR_W_IN + 0.18
        chip_ax_w = chip_w_in / (PANEL_W_IN / 0.96)   # normalise to axes frac

        if x + chip_ax_w > 0.95:   # wrap to next row
            x          = 0.05
            y_chip    -= chip_ax_h + gap_ax * 2
            row_start_y = y_chip

        ax.add_patch(mpatches.FancyBboxPatch(
            (x, y_chip), chip_ax_w, chip_ax_h,
            boxstyle="round,pad=0.006",
            facecolor=col, edgecolor="none",
            transform=ax.transAxes, zorder=2,
        ))
        ax.text(x + chip_ax_w / 2, y_chip + chip_ax_h / 2,
                label,
                transform=ax.transAxes, va="center", ha="center",
                fontsize=_FS - 2.5, color="white",
                fontfamily="DejaVu Sans", fontweight="bold", zorder=3)
        x += chip_ax_w + gap_ax

    # ── global scene section ──────────────────────────────────────────────
    y_gs = row_start_y - chip_ax_h - 0.06
    ax.text(0.05, y_gs, "Global Scene",
            transform=ax.transAxes, va="top", ha="left",
            fontsize=_FS, fontweight="bold", color="#333333",
            fontfamily=_FONT)

    DY = ((_FS + 0.5) * 1.6 / 72) / _INFO_H   # one line height in axes frac
    y_gs -= DY * 1.6

    for gt in global_texts:
        wrapped = textwrap.wrap(gt, 42) or [""]
        for j, seg in enumerate(wrapped):
            prefix = "•  " if j == 0 else "   "
            ax.text(0.06, y_gs, prefix + seg,
                    transform=ax.transAxes, va="top", ha="left",
                    fontsize=_FS - 0.5, color="#222222",
                    fontfamily=_FONT)
            y_gs -= DY


# ─────────────────────────────────── figure ──────────────────────────────────

def render_one(
    record: Dict[str, Any],
    image_dir: Path,
    output_dir: Optional[Path],
    show: bool,
    dpi: int = 150,
) -> None:
    image_id = record.get("image_id", "unknown")
    img: Optional[Image.Image] = None
    for cand in [Path(record.get("image_path", "")), image_dir / image_id]:
        if cand.exists():
            img = Image.open(cand).convert("RGB")
            break

    objects = record.get("objects", [])
    n_obj   = len(objects)
    n_left  = math.ceil(n_obj / 2)
    objs_L  = objects[:n_left]
    objs_R  = objects[n_left:]

    def _col_h(objs: List) -> float:
        return sum(_card_h(o) for o in objs) if objs else 0.0

    # Image height from aspect ratio
    CENTER_W_IN = _FIG_W * _CENTER_FRAC
    if img is not None:
        iw, ih = img.size
        img_h_in = min(CENTER_W_IN * ih / iw, 8.0)
    else:
        img_h_in = 5.0
    center_h = img_h_in + _INFO_H

    raw_side_h = max(_col_h(objs_L), _col_h(objs_R))
    content_h  = max(raw_side_h, center_h)
    fig_h      = content_h + _TITLE_H + 0.25

    # Scale down card heights if figure would be too tall
    if fig_h > _MAX_FIG_H:
        avail = _MAX_FIG_H - _TITLE_H - 0.25
        # Only scale cards, not the center image+info
        scale = avail / raw_side_h if raw_side_h > avail else 1.0
        fig_h = _MAX_FIG_H
    else:
        scale = 1.0

    def _scaled_card_h(obj: Dict[str, Any]) -> float:
        return _card_h(obj) * scale

    fig: Figure = plt.figure(figsize=(_FIG_W, fig_h), dpi=dpi)
    fig.patch.set_facecolor("#E4E4E4")

    # ── outer grid: title + content ───────────────────────────────────────
    gs_outer = fig.add_gridspec(
        2, 1,
        height_ratios=[_TITLE_H, fig_h - _TITLE_H],
        left=0.005, right=0.995, top=0.998, bottom=0.005,
        hspace=0.010,
    )

    # title bar
    ax_t = fig.add_subplot(gs_outer[0])
    ax_t.axis("off")
    ax_t.add_patch(mpatches.FancyBboxPatch(
        (0, 0), 1, 1, boxstyle="square,pad=0",
        facecolor="#2C3E50", edgecolor="none",
        transform=ax_t.transAxes,
    ))
    ax_t.text(0.010, 0.50,
              f"  {image_id}   │   {n_obj} objects",
              transform=ax_t.transAxes, va="center", ha="left",
              fontsize=_FS + 1, fontweight="bold", color="white",
              fontfamily=_FONT)
    tags_str = "  ·  ".join(record.get("challenge_tags", []))
    if tags_str:
        ax_t.text(0.990, 0.50, tags_str,
                  transform=ax_t.transAxes, va="center", ha="right",
                  fontsize=_FS - 1, color="#AAD4FF",
                  fontfamily=_FONT)

    # ── content: left | center | right ───────────────────────────────────
    gs_c = gs_outer[1].subgridspec(
        1, 3,
        width_ratios=[_SIDE_FRAC, _CENTER_FRAC, _SIDE_FRAC],
        wspace=0.022,
    )

    # center column: image (top) + info panel (bottom)
    gs_mid = gs_c[1].subgridspec(
        2, 1,
        height_ratios=[img_h_in, _INFO_H],
        hspace=0.018,
    )
    ax_img  = fig.add_subplot(gs_mid[0])
    ax_info = fig.add_subplot(gs_mid[1])

    # image axes — thin spine border, no ticks
    ax_img.set_xticks([])
    ax_img.set_yticks([])
    for sp in ax_img.spines.values():
        sp.set_edgecolor("#AAAAAA")
        sp.set_linewidth(1.0)
    ax_img.set_facecolor("#CCCCCC")
    if img is not None:
        ax_img.imshow(img)
    else:
        ax_img.axis("off")
        ax_img.text(0.5, 0.5, f"Not found\n{image_id}",
                    ha="center", va="center", transform=ax_img.transAxes,
                    color="#CC0000", fontsize=11)

    _draw_info(ax_info, record)

    # left cards
    if objs_L:
        gs_L = gs_c[0].subgridspec(len(objs_L), 1,
                                    height_ratios=[_scaled_card_h(o) for o in objs_L],
                                    hspace=0.022)
        for i, obj in enumerate(objs_L):
            ax = fig.add_subplot(gs_L[i])
            ax.set_facecolor("#E4E4E4")
            _draw_card(ax, obj, _CARD_THEMES[i % len(_CARD_THEMES)])

    # right cards
    if objs_R:
        gs_R = gs_c[2].subgridspec(len(objs_R), 1,
                                    height_ratios=[_scaled_card_h(o) for o in objs_R],
                                    hspace=0.022)
        for i, obj in enumerate(objs_R):
            ax = fig.add_subplot(gs_R[i])
            ax.set_facecolor("#E4E4E4")
            _draw_card(ax, obj, _CARD_THEMES[(n_left + i) % len(_CARD_THEMES)])

    # ── output ────────────────────────────────────────────────────────────
    if output_dir is not None:
        output_dir.mkdir(parents=True, exist_ok=True)
        out_path = output_dir / f"{Path(image_id).stem}_annotation.png"
        fig.savefig(out_path, dpi=dpi, bbox_inches="tight",
                    facecolor=fig.get_facecolor())
        print(f"Saved: {out_path}")
    if show:
        plt.show()
    plt.close(fig)


# ─────────────────────────────────────── CLI ─────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--jsonl",      required=True)
    p.add_argument("--image_dir",  required=True)
    p.add_argument("--n",          type=int, default=None)
    p.add_argument("--output_dir", default=None)
    p.add_argument("--show",       action="store_true")
    p.add_argument("--dpi",        type=int, default=150)
    return p.parse_args()


def main() -> int:
    args       = parse_args()
    output_dir = Path(args.output_dir) if args.output_dir else None
    show       = args.show or (output_dir is None)
    if show:
        matplotlib.use("TkAgg")

    jsonl_path = Path(args.jsonl)
    if not jsonl_path.exists():
        print(f"Error: {jsonl_path} not found.")
        return 1

    records: List[Dict[str, Any]] = []
    with jsonl_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as e:
                print(f"Skipping malformed line: {e}")

    if args.n is not None:
        records = records[: args.n]

    print(f"Visualizing {len(records)} record(s) …")
    for record in records:
        render_one(record,
                   image_dir=Path(args.image_dir),
                   output_dir=output_dir,
                   show=show,
                   dpi=args.dpi)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
