from PIL import Image, ImageDraw, ImageFont
import os

from PIL import Image, ImageDraw, ImageFont
import os
import PIL

from typing import Optional

import os
from PIL import ImageFont
# If you already import matplotlib elsewhere, you can reuse it
try:
    from matplotlib import font_manager as fm
except ImportError:
    fm = None

def resolve_ttf(font_path: Optional[str] = None) -> str:
    # 1) User-provided path
    if font_path and os.path.exists(font_path):
        return font_path

    # 2) Matplotlib bundled DejaVu (most reliable in headless envs)
    if fm is not None:
        try:
            ttf = fm.findfont("DejaVu Sans", fallback_to_default=False)
            if ttf and os.path.exists(ttf):
                return ttf
        except Exception:
            pass

    # 3) Pillow-bundled fonts (sometimes present)
    try:
        import PIL
        pil_dir = os.path.dirname(PIL.__file__)
        for name in ("DejaVuSans.ttf", "DejaVuSans-Bold.ttf"):
            cand = os.path.join(pil_dir, "fonts", name)
            if os.path.exists(cand):
                return cand
    except Exception:
        pass

    # 4) Common OS locations
    candidates = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",            # Linux
        "/usr/share/fonts/truetype/freefont/FreeSans.ttf",
        "/Library/Fonts/Arial.ttf",                                   # macOS
        "/System/Library/Fonts/Supplemental/Arial.ttf",
        "C:\\Windows\\Fonts\\Arial.ttf",                              # Windows
        "C:\\Windows\\Fonts\\DejaVuSans.ttf",
    ]
    for cand in candidates:
        if os.path.exists(cand):
            return cand

    raise RuntimeError(
        "Couldn't find a TTF font. Provide font_path=... or install one "
        "(e.g., apt-get install fonts-dejavu-core, or use Matplotlib)."
)


def _fit_letterbox(im: Image.Image, size=(512, 512), bg=(255, 255, 255)) -> Image.Image:
    """Resize to fit within size, keep aspect, pad with bg."""
    im = im.convert("RGB")
    W, H = size
    w, h = im.size
    scale = min(W / w, H / h)
    new = (max(1, int(w * scale)), max(1, int(h * scale)))
    im = im.resize(new, Image.LANCZOS)
    canvas = Image.new("RGB", (W, H), bg)
    off = ((W - new[0]) // 2, (H - new[1]) // 2)
    canvas.paste(im, off)
    return canvas

def make_frame(triplet_paths,
               size=(512, 512),
               labels=("Input Image", "Re-rendered Mesh", "Inpainted image"),
               caption_h=None,              # auto if None
               font_path: Optional[str]=None,
               font_size=34,
               bg=(255, 255, 255),
               text_color=(0, 0, 0),
               stroke_width=2,
               stroke_fill=(255, 255, 255)) -> Image.Image:

    # Resolve font (your resolve_ttf() from above)
    ttf = resolve_ttf(font_path)
    font = ImageFont.truetype(ttf, font_size)

    # Auto caption height if not provided
    if caption_h is None:
        ascent, descent = font.getmetrics()
        caption_h = int((ascent + descent) * 1.6)

    # --- LOAD & FIT THE THREE IMAGES ---
    imgs = []
    for p in triplet_paths:
        if isinstance(p, Image.Image):
            im = p
        else:
            if not os.path.exists(p):
                # Optional: draw a placeholder “missing” tile instead of failing
                placeholder = Image.new("RGB", size, (240, 240, 240))
                d = ImageDraw.Draw(placeholder)
                msg = "Missing:\n" + os.path.basename(p)
                d.multiline_text((10, 10), msg, fill=(0, 0, 0))
                im = placeholder
            else:
                im = Image.open(p)
        imgs.append(_fit_letterbox(im, size=size, bg=bg))

    # --- COMPOSE THE CANVAS ---
    W = size[0] * 3
    H = size[1] + caption_h
    canvas = Image.new("RGB", (W, H), bg)
    draw = ImageDraw.Draw(canvas)

    # Paste the 3 tiles side-by-side
    for i, im in enumerate(imgs):
        x = i * size[0]
        canvas.paste(im, (x, 0))

    # Draw centered labels under each tile
    for i, label in enumerate(labels):
        x0 = i * size[0]
        bbox = draw.textbbox((0, 0), label, font=font, stroke_width=stroke_width)
        tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
        tx = x0 + (size[0] - tw) // 2
        ty = size[1] + (caption_h - th) // 2
        draw.text((tx, ty), label, font=font, fill=text_color,
                  stroke_width=stroke_width, stroke_fill=stroke_fill)

    return canvas



def build_gif(triplets, out_path="comparison.gif",
              size=(512, 512),
              duration_ms=900,  # per-frame duration in ms
              loop=0,
              **frame_kwargs):
    """
    triplets: list of 3-tuples (input_path, warped_path, inpainted_path)
    """
    assert len(triplets) > 0, "Need at least one frame"
    frames = [make_frame(t, size=size, **frame_kwargs) for t in triplets]

    # PIL can write GIF directly
    frames[0].save(
        out_path,
        save_all=True,
        append_images=frames[1:],
        duration=duration_ms,
        loop=loop,
        optimize=True,
        disposal=2,  # helps reduce frame artifacts
    )
    print(f"Saved: {out_path}")

triplets = [
    ("examples/69.png",   "gradio_cached_examples/26/Warped View Kaolin/1218e658b8452a72065b/image.webp",  "gradio_cached_examples/26/Inpainted Result/1b9622e7fe007cf48897/image.webp"),
    ("examples/117.png",  "gradio_cached_examples/26/Warped View Kaolin/aee089803814225aed8f/image.webp",  "gradio_cached_examples/26/Inpainted Result/7d7d9dea3eb53d57b39d/image.webp"),
    ("examples/129.png",  "gradio_cached_examples/26/Warped View Kaolin/e85960daccad8bf3d18a/image.webp",  "gradio_cached_examples/26/Inpainted Result/c1711371cf54389e4364/image.webp"),
    ("examples/201.png",  "gradio_cached_examples/26/Warped View Kaolin/b8ee38f6085bdc3168e3/image.webp",  "gradio_cached_examples/26/Inpainted Result/9de84913b8bb7d79890c/image.webp"),
    ("examples/237.png",  "gradio_cached_examples/26/Warped View Kaolin/98380fa3e4d0c4120366/image.webp",  "gradio_cached_examples/26/Inpainted Result/bb711b8d1f68571f0647/image.webp"),
    ("examples/1521.png", "gradio_cached_examples/26/Warped View Kaolin/c83e2286777590d5513a/image.webp",  "gradio_cached_examples/26/Inpainted Result/66242d73bfd713c1dba9/image.webp"),
    ("examples/1533.png", "gradio_cached_examples/26/Warped View Kaolin/8e857c4019908c399c9b/image.webp",  "gradio_cached_examples/26/Inpainted Result/bcaaccc80471d05f10ac/image.webp"),
]

build_gif(
    triplets,
    out_path="perspective_series.gif",
    size=(512, 512),
    duration_ms=900,    # ~1.1 fps; tweak as you like
    font_path=None,     # or path to a .ttf (e.g., "/usr/share/fonts/DejaVuSans.ttf")
    font_size=30,
)
