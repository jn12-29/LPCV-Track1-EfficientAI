"""
Visualize LPCV annotations as HTML galleries.

Layout is computed adaptively from the actual image size and object count:
  - Image occupies the center at its natural aspect ratio (max 520px)
  - Objects are distributed to top / left / right / bottom zones
  - Landscape images (wide) → more objects on top/bottom
  - Portrait images (tall)  → more objects on sides
  - Each object block shows positives AND negatives together

  python visualize_annotations.py \
      --jsonl out/dataset_raw.jsonl \
      --image_dir /data/VG_100K \
      --output_dir out/vis_raw --n 200

  python visualize_annotations.py \
      --jsonl out/dataset_raw_contrastive.jsonl \
      --image_dir /data/VG_100K \
      --output_dir out/vis_con --n 200 --mode contrastive
"""
import os, json, argparse, base64, mimetypes, html, struct
from pathlib import Path

# ── image loading (reads file once → URI + dimensions) ───────────────────────

def _png_size(data):
    if data[:8] == b'\x89PNG\r\n\x1a\n':
        return struct.unpack('>II', data[16:24])
    return None

def _jpeg_size(data):
    if data[:2] != b'\xff\xd8':
        return None
    i = 2
    while i < len(data) - 8:
        if data[i] != 0xff:
            break
        marker = data[i + 1]
        if marker in (0xC0, 0xC1, 0xC2, 0xC3, 0xC5, 0xC6, 0xC7,
                      0xC9, 0xCA, 0xCB, 0xCD, 0xCE, 0xCF):
            h, w = struct.unpack('>HH', data[i + 5:i + 9])
            return w, h
        seg_len = struct.unpack('>H', data[i + 2:i + 4])[0]
        i += 2 + seg_len
    return None

def load_image(path):
    """
    Returns (data_uri, width, height).
    Reads the file once; parses PNG / JPEG headers without PIL.
    Falls back to aspect 1:1 on parse failure.
    """
    mime = mimetypes.guess_type(path)[0] or "image/jpeg"
    try:
        with open(path, "rb") as f:
            data = f.read()
    except Exception:
        return "", 1, 1

    uri = f"data:{mime};base64,{base64.b64encode(data).decode()}"

    size = _png_size(data) or _jpeg_size(data)
    if size:
        return uri, size[0], size[1]

    # PIL fallback (handles WEBP, GIF, BMP, TIFF, …)
    try:
        from PIL import Image as PILImage
        import io
        with PILImage.open(io.BytesIO(data)) as img:
            return uri, img.width, img.height
    except Exception:
        pass

    return uri, 1, 1   # unknown size → treat as square

# ── adaptive distribution ─────────────────────────────────────────────────────

def distribute(n, img_w, img_h):
    """
    Distribute n object blocks to top / left / right / bottom zones.

    Strategy:
    - Image is displayed at most MAX_IMG px on its longer side.
    - Side columns (left/right) stack objects vertically →
        capacity ∝ displayed image height.
    - Top/bottom rows lay objects horizontally →
        capacity is roughly fixed (~3 per row, wraps if needed).
    - Aim for ~1/4 to each zone; cap sides by their estimated capacity;
      overflow goes to bottom.

    Returns (n_top, n_left, n_right, n_bottom).
    """
    if n == 0:
        return 0, 0, 0, 0

    MAX_IMG    = 520.0
    OBJ_H      = 145.0   # estimated height of one object block (px)
    OBJ_W_TB   = 260.0   # estimated width of one object block in top/bottom row (px)
    CARD_W     = 960.0   # estimated usable card width (px)

    aspect = img_w / max(img_h, 1)

    # Displayed image dimensions
    if aspect >= 1.0:
        disp_w, disp_h = MAX_IMG, MAX_IMG / aspect
    else:
        disp_h, disp_w = MAX_IMG, MAX_IMG * aspect

    # Capacity per zone
    cap_side   = max(1, int(disp_h / OBJ_H))           # per side column
    cap_topbot = max(2, int(CARD_W / OBJ_W_TB))         # top/bottom row (wraps)

    # Target: spread objects ~evenly across four zones (T, L, R, B)
    target_T = (n + 3) // 4
    target_L = (n + 2) // 4
    target_R = (n + 1) // 4
    # rest goes to bottom

    n_top    = min(cap_topbot, target_T)
    n_left   = min(cap_side,   target_L)
    n_right  = min(cap_side,   target_R)
    n_bottom = n - n_top - n_left - n_right   # absorbs any overflow

    return n_top, n_left, n_right, n_bottom

# ── CSS ───────────────────────────────────────────────────────────────────────

CSS = """
*{box-sizing:border-box;margin:0;padding:0}
body{font-family:system-ui,'Segoe UI',Arial,sans-serif;background:#0c0c11;
     color:#bbb;padding:24px;line-height:1.5;
     font-size:clamp(13px,1.15vw,17px)}
h2{color:#778;font-size:clamp(13px,1.1vw,16px);margin-bottom:20px;
   font-weight:400;letter-spacing:.05em}

/* ── card ──────────────────────────────────────────────────────── */
.card{background:#13131a;border:1px solid #1f1f2c;border-radius:12px;margin:20px 0;
      overflow:hidden}

/* ── header ─────────────────────────────────────────────────────── */
.card-head{padding:9px 15px 7px;border-bottom:1px solid #1a1a26;
           display:flex;flex-wrap:wrap;align-items:baseline;gap:5px 10px}
.fname{font-size:clamp(13px,1.2vw,16px);color:#6a9;font-weight:600;flex-shrink:0}
.global{font-size:clamp(12px,1.1vw,15px);color:#47474f;font-style:italic}
.tag{display:inline-block;background:#111e2a;color:#3d7a96;border:1px solid #1a3040;
     padding:1px 6px;border-radius:7px;font-size:clamp(11px,1vw,14px)}

/* ── three-column center: L col | image | R col ─────────────────── */
.mid{display:grid;grid-template-columns:minmax(150px,1fr) auto minmax(150px,1fr);
     align-items:start}
.col{padding:11px 10px;display:flex;flex-direction:column;gap:7px}
.img-cell{padding:14px;background:#08080f;display:flex;align-items:flex-start;
          justify-content:center;
          border-left:1px solid #1a1a26;border-right:1px solid #1a1a26}
.img-cell img{max-width:520px;max-height:520px;width:100%;height:auto;
              border-radius:8px;object-fit:contain;display:block}

/* ── top row ─────────────────────────────────────────────────────── */
.top{display:flex;flex-wrap:wrap;gap:8px;
     padding:8px 10px;border-bottom:1px solid #1a1a26}
.top .obj{flex:1;min-width:200px}

/* ── bottom row ──────────────────────────────────────────────────── */
.bottom{display:flex;flex-wrap:wrap;gap:8px;
        padding:8px 10px;border-top:1px solid #1a1a26}

/* ── footer: relational + text_on_object ────────────────────────── */
.foot{display:flex;flex-wrap:wrap;gap:4px;
      padding:7px 14px 10px;border-top:1px solid #1a1a26}
.foot-lbl{width:100%;font-size:clamp(10px,0.9vw,13px);color:#282830;
          text-transform:uppercase;letter-spacing:.05em;margin-bottom:2px}

/* ── per-object block ────────────────────────────────────────────── */
.obj{background:#0f0f18;border:1px solid #1b1b27;border-radius:9px;padding:8px 10px}
/* in bottom row: expand to fill space, min width so they don't collapse */
.bottom .obj{flex:1;min-width:200px}
.obj-name{font-size:clamp(13px,1.2vw,16px);font-weight:600;color:#c8954a;margin-bottom:5px}
.sal{font-size:clamp(10px,0.9vw,13px);color:#3a3a48;background:#161620;padding:1px 5px;
     border-radius:4px;margin-left:5px;font-weight:400}
.chips{display:flex;flex-direction:column;gap:3px}
.sep{height:1px;background:#1c1c28;margin:4px 0}

/* ── chips ────────────────────────────────────────────────────────── */
.chip{display:inline-flex;flex-wrap:wrap;align-items:baseline;
      padding:3px 8px;border-radius:8px;font-size:clamp(12px,1.15vw,15px);
      gap:3px;line-height:1.5}
.pos {background:#0a1c13;color:#5c5;border:1px solid #162e1e}
.weak{background:#1a190a;color:#aa8;border:1px solid #2a280f}
.neg-a{background:#1e0b0b;color:#d55;border:1px solid #361212}
.neg-s{background:#1e1108;color:#d73;border:1px solid #362010}
.neg {background:#1e0b0b;color:#d55;border:1px solid #361212}
.rel {background:#0a1726;color:#59b;border:1px solid #102035}
.txt {background:#120e22;color:#97c;border:1px solid #1e143a}

/* ── inline badges ───────────────────────────────────────────────── */
.b-axis{font-size:clamp(10px,0.9vw,13px);background:#071220;color:#38899e;
        border:1px solid #0d1e30;padding:0 4px;border-radius:4px}
.b-edit{font-size:clamp(10px,0.9vw,13px);background:#1e0808;color:#a44;
        border:1px solid #300c0c;padding:0 4px;border-radius:4px}
.why{font-size:clamp(11px,1vw,14px);color:#363630;font-style:italic}
.err{color:#c44;font-size:clamp(12px,1.1vw,15px);padding:3px 0}
"""

# ── helpers ───────────────────────────────────────────────────────────────────

def e(s):
    return html.escape(str(s))

def chip_pos(t):
    if not isinstance(t, dict): return ""
    ax = (f'<span class="b-axis">{e(t["attribute_axis"])}</span>'
          if t.get("attribute_axis") else "")
    return f'<span class="chip pos">{e(t["text"])}{ax}</span>'

def chip_neg_a(n):
    if not isinstance(n, dict): return ""
    ed  = f'<span class="b-edit">{e(n["edit_type"])}</span>' if n.get("edit_type") else ""
    why = f'<span class="why">— {e(n["rationale"])}</span>'  if n.get("rationale") else ""
    return f'<span class="chip neg-a">{e(n["text"])}{ed}{why}</span>'

def chip_neg_s(n):
    if not isinstance(n, dict): return ""
    why = f'<span class="why">— {e(n["rationale"])}</span>' if n.get("rationale") else ""
    return f'<span class="chip neg-s">{e(n["text"])}{why}</span>'

def chip_plain(text, cls):
    return f'<span class="chip {cls}">{e(text)}</span>'

def obj_block(obj, in_bottom=False):
    """Render one object: name header, positives, separator, negatives."""
    name = obj.get("object_name", "")
    sal  = obj.get("salience",    "")

    pos  = "".join(chip_pos(t)          for t in obj.get("positive_texts",          []))
    weak = "".join(chip_plain(t, "weak") for t in obj.get("weak_positives",          []))
    na   = "".join(chip_neg_a(n)         for n in obj.get("hard_negatives_attribute", []))
    ns   = "".join(chip_neg_s(n)         for n in obj.get("hard_negatives_scene",     []))

    sep = '<div class="sep"></div>' if (pos or weak) and (na or ns) else ""
    return (f'<div class="obj">'
            f'<div class="obj-name">{e(name)}<span class="sal">{e(sal)}</span></div>'
            f'<div class="chips">{pos}{weak}{sep}{na}{ns}</div>'
            f'</div>')

# ── card renderers ─────────────────────────────────────────────────────────────

def render_card_raw(rec, image_dir):
    ann      = rec.get("annotation", {})
    img_path = rec.get("image_path", "")
    if not os.path.isabs(img_path) and image_dir:
        img_path = os.path.join(image_dir, os.path.basename(img_path))

    uri, img_w, img_h = load_image(img_path)
    objects = ann.get("objects", [])

    # ── header ───────────────────────────────────────────────────────
    tags_h  = "".join(f'<span class="tag">{e(t)}</span>'
                      for t in ann.get("challenge_tags", []))
    globs_h = " &nbsp;·&nbsp; ".join(e(g) for g in ann.get("global_texts", []))
    err_h   = f'<div class="err">ERROR: {e(rec["error"])}</div>' if rec.get("error") else ""
    header  = (f'<div class="card-head">'
               f'<span class="fname">{e(os.path.basename(img_path))}</span>'
               f'<span style="font-size:10px;color:#333">{img_w}×{img_h}</span>'
               f'{tags_h}<span class="global">{globs_h}</span>{err_h}</div>')

    # ── distribute objects ────────────────────────────────────────────
    n_T, n_L, n_R, n_B = distribute(len(objects), img_w, img_h)
    top_objs    = objects[:n_T]
    left_objs   = objects[n_T:n_T + n_L]
    right_objs  = objects[n_T + n_L:n_T + n_L + n_R]
    bottom_objs = objects[n_T + n_L + n_R:]

    # ── top row ──────────────────────────────────────────────────────
    top = ""
    if top_objs:
        top_h = "".join(obj_block(o) for o in top_objs)
        top = f'<div class="top">{top_h}</div>'

    # ── mid row: left col | image | right col ────────────────────────
    hint = "<span style='color:#1c1c28;font-size:11px;padding:4px 0'>—</span>"
    left_h  = "".join(obj_block(o) for o in left_objs)  or hint
    right_h = "".join(obj_block(o) for o in right_objs) or hint
    mid = (f'<div class="mid">'
           f'<div class="col">{left_h}</div>'
           f'<div class="img-cell">'
           f'<img src="{uri}" alt="{e(os.path.basename(img_path))}">'
           f'</div>'
           f'<div class="col">{right_h}</div>'
           f'</div>')

    # ── bottom row ───────────────────────────────────────────────────
    bottom = ""
    if bottom_objs:
        bottom_h = "".join(obj_block(o, in_bottom=True) for o in bottom_objs)
        bottom = f'<div class="bottom">{bottom_h}</div>'

    # ── footer: relational + text_on_object ──────────────────────────
    rel = "".join(chip_plain(t, "rel") for o in objects for t in o.get("relational_texts", []))
    txt = "".join(chip_plain(t, "txt") for o in objects for t in o.get("text_on_object",  []))
    foot = ""
    if rel or txt:
        lbl  = '<span class="foot-lbl">relational &amp; text on object</span>' if rel and txt else ""
        foot = f'<div class="foot">{lbl}{rel}{txt}</div>'

    return f'<div class="card">{header}{top}{mid}{bottom}{foot}</div>'


def render_card_contrastive(rec, image_dir):
    img_path = rec.get("image_path", "")
    if not os.path.isabs(img_path) and image_dir:
        img_path = os.path.join(image_dir, os.path.basename(img_path))

    uri, img_w, img_h = load_image(img_path)

    header = (f'<div class="card-head">'
              f'<span class="fname">{e(os.path.basename(img_path))}</span>'
              f'<span style="font-size:clamp(10px,0.9vw,13px);color:#333">{img_w}×{img_h}</span>'
              f'</div>')

    pos_h = "".join(chip_plain(t, "pos") for t in rec.get("positives",      []))
    neg_h = "".join(chip_plain(t, "neg") for t in rec.get("hard_negatives", []))
    sep   = '<div class="sep"></div>' if pos_h and neg_h else ""
    block = (f'<div class="obj">'
             f'<div class="chips">{pos_h}{sep}{neg_h}</div></div>')

    mid = (f'<div class="mid">'
           f'<div class="col">{block}</div>'
           f'<div class="img-cell">'
           f'<img src="{uri}" alt="{e(os.path.basename(img_path))}">'
           f'</div>'
           f'<div class="col"></div>'
           f'</div>')

    return f'<div class="card">{header}{mid}</div>'

# ── main ──────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--jsonl",      required=True)
    ap.add_argument("--image_dir",  default="")
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--n",          type=int, default=500)
    ap.add_argument("--mode",       default="auto",
                    choices=["auto", "raw", "contrastive"])
    a = ap.parse_args()

    mode   = a.mode
    if mode == "auto":
        mode = "contrastive" if "contrastive" in Path(a.jsonl).name else "raw"
    render = render_card_contrastive if mode == "contrastive" else render_card_raw

    cards, n = [], 0
    with open(a.jsonl) as f:
        for line in f:
            if n >= a.n: break
            try: rec = json.loads(line)
            except Exception: continue
            cards.append(render(rec, a.image_dir))
            n += 1

    src   = Path(a.jsonl).name
    title = f"LPCV 2026 — {src} ({n} records, mode={mode})"
    doc   = (f"<!doctype html><html><head><meta charset='utf-8'>"
             f"<title>{title}</title><style>{CSS}</style></head>"
             f"<body><h2>{title}</h2>" + "".join(cards) + "</body></html>")

    out_dir = Path(a.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_html = out_dir / "vis.html"
    out_html.write_text(doc, encoding="utf-8")
    print(f"[ok] {out_html}  ({n} cards, mode={mode})")

if __name__ == "__main__":
    main()
