"""
Gradio demo for perspective change using MoGe + Kaolin rasterizer + Stable Diffusion Inpainting
----------------------------------------------------------------------------------------------
- Upload an RGB image
- (Optional) enter a subject phrase for GroundingDINO (e.g., "a person", "the car")
- Choose a tilt angle and distance multiplier
- Provide a prompt for the inpainting step
- The app will:
    1) Run MoGe to get depth, intrinsics, and a colored point cloud
    2) Use GroundingDINO to localize the subject and pick a center point (or fallback to center)
    3) Build a textured mesh from MoGe outputs
    4) Render a tilted view with Kaolin (DIB-R)
    5) Create an edit mask (everything *except* the rendered subject by default)
    6) Run Stable Diffusion Inpainting to fill/extend the scene to the new view

Notes
-----
• You need working CUDA + PyTorch, Kaolin, trimesh, GroundingDINO, and diffusers installed.
• You also need the GroundingDINO weights locally (default path below) and HF access for the SD2 Inpainting model.
• This is a reference implementation; adjust paths / devices for your setup. 
"""

import os
import math
import ast
import cv2
import gc
import numpy as np
import torch
import gradio as gr
import trimesh
from io import BytesIO
from functools import lru_cache
from typing import Tuple, Optional
import utils3d


import PIL
from PIL import Image, ImageOps

# Kaolin
import kaolin as kal
from kaolin.render.camera import PinholeIntrinsics

# MoGe
from MoGe.moge.model.v1 import MoGeModel

# GroundingDINO
from groundingdino.util.inference import load_model, predict, annotate, Model

# Diffusers
from diffusers import StableDiffusionInpaintPipeline

# ----- Utils you had in your script (ported inline) -----

def resize_and_pad_to_square(img: np.ndarray, target_long_side: int = 512) -> np.ndarray:
    h, w = img.shape[:2]
    scale = target_long_side / max(h, w)
    new_h, new_w = int(round(h * scale)), int(round(w * scale))
    resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)

    pad_h = target_long_side - new_h
    pad_w = target_long_side - new_w

    pad_top = pad_h // 2
    pad_bottom = pad_h - pad_top
    pad_left = pad_w // 2
    pad_right = pad_w - pad_left

    if img.ndim == 3:
        padded_img = cv2.copyMakeBorder(
            resized, pad_top, pad_bottom, pad_left, pad_right,
            borderType=cv2.BORDER_CONSTANT, value=(0, 0, 0)
        )
    else:
        padded_img = cv2.copyMakeBorder(
            resized, pad_top, pad_bottom, pad_left, pad_right,
            borderType=cv2.BORDER_CONSTANT, value=0
        )
    return padded_img


def get_tilted_camera(center: torch.Tensor, tilt_degrees: float = 0.0,
                       pullback_distance: float = 2.0, device: str = "cuda") -> torch.Tensor:
    """Return a 4x4 world-to-camera matrix viewing the given center from a tilted eye.
    The mesh is assumed in camera space like MoGe output; we create a *view* matrix.
    """
    center = center.to(device)
    up = torch.tensor([0, 1, 0], device=device, dtype=torch.float32)
    forward = torch.tensor([0, 0, 1], device=device, dtype=torch.float32)

    tilt_rad = math.radians(tilt_degrees)
    cos_t, sin_t = math.cos(tilt_rad), math.sin(tilt_rad)
    rot_x = torch.tensor([
        [1, 0,      0],
        [0, cos_t, -sin_t],
        [0, sin_t,  cos_t]
    ], device=device, dtype=torch.float32)

    tilted_forward = rot_x @ forward
    eye = center + tilted_forward * pullback_distance

    z = (eye - center)
    z = z / torch.clamp(torch.norm(z), min=1e-8)
    x = torch.cross(up, z)
    x = x / torch.clamp(torch.norm(x), min=1e-8)
    y = torch.cross(z, x)

    R = torch.stack([x, y, z], dim=1)

    view = torch.eye(4, device=device)
    view[:3, :3] = R.T
    view[:3, 3] = -(R.T @ eye)
    return view.unsqueeze(0)


# --------------- Model Loaders (cached) -----------------

DEFAULT_DINO_CFG = "GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py"
DEFAULT_DINO_WEIGHTS = "model_cache/groundingdino_swint_ogc.pth"
DEFAULT_DEVICE = os.environ.get("MOGE_DEVICE", "cuda:0")


@lru_cache(maxsize=1)
def get_device() -> torch.device:
    return torch.device(DEFAULT_DEVICE if torch.cuda.is_available() else "cpu")


@lru_cache(maxsize=1)
def load_moge() -> MoGeModel:
    device = get_device()
    model = MoGeModel.from_pretrained("Ruicheng/moge-vitl").to(device)
    model.eval()
    return model


@lru_cache(maxsize=1)
def load_dino(cfg_path: str = DEFAULT_DINO_CFG,
             weights_path: str = DEFAULT_DINO_WEIGHTS):
    if not os.path.isfile(weights_path):
        raise FileNotFoundError(
            f"GroundingDINO weights not found at '{weights_path}'. Update the path in the UI settings."
        )
    model = load_model(cfg_path, weights_path)
    return model


@lru_cache(maxsize=1)
def load_inpainter(model_id: str = "stabilityai/stable-diffusion-2-inpainting") -> StableDiffusionInpaintPipeline:
    device = get_device()
    pipe = StableDiffusionInpaintPipeline.from_pretrained(
        model_id, torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
    )
    pipe.to(device)
    return pipe


# ---------------- Core processing function ---------------

def run_pipeline(
    image: Image.Image,
    subject: str,
    prompt: str,
    dino_box_threshold: float = 0.35,
    dino_text_threshold: float = 0.25,
    tilt_degrees: float = 0.0,
    dist_multiplier: float = 1.5,
    dino_cfg_path: str = DEFAULT_DINO_CFG,
    dino_weights_path: str = DEFAULT_DINO_WEIGHTS,
    progress: Optional[gr.Progress] = None
) -> Tuple[Image.Image, Image.Image, Image.Image, Image.Image, Image.Image, str]:
    """
    Returns (annotated_bbox, depth_vis, warped_view, edit_mask, inpainted, log)
    """
    device = get_device()
    log_lines = []
    tilt_degrees = -tilt_degrees 
    # to make it such that > 0 is moving the camera upwards, < 0 is downwards wrt to the object

    # Safety checks
    if image is None:
        raise gr.Error("Please upload an image.")
    if prompt is None or len(prompt.strip()) == 0:
        raise gr.Error("Please provide a prompt for inpainting.")

    # Convert PIL to numpy RGB
    rgb = np.array(image.convert("RGB"))
    H, W, _ = rgb.shape

    if progress:
        progress(0.05, desc="Loading models…")

    MoGe = load_moge()
    pipe = load_inpainter()
    GroundingDINO = load_dino(dino_cfg_path, dino_weights_path)

    if progress:
        progress(0.15, desc="Running GroundingDINO…")

    # GroundingDINO preprocessing expects BGR
    transformed_image = Model.preprocess_image(cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))

    try:
        boxes, logits, phrases = predict(
            model=GroundingDINO,
            image=transformed_image,
            caption=(subject or ""),
            box_threshold=dino_box_threshold,
            text_threshold=dino_text_threshold,
        )
    except Exception as e:
        boxes, logits, phrases = [], [], []
        log_lines.append(f"GroundingDINO failed: {e}")

    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    annotated = annotate(image_source=bgr, boxes=boxes, logits=logits, phrases=phrases)

    if boxes is not None and len(boxes) > 0:
        best = boxes[np.argmax(logits)]
        box_pix = best * np.array([W, H, W, H])
        cx, cy = int(box_pix[0]), int(box_pix[1])
        log_lines.append(f"Using bbox center at ({cx},{cy})")
    else:
        cx, cy = W // 2, H // 2
        log_lines.append("No bbox detected; using image center")

    if progress:
        progress(0.35, desc="Running MoGe…")

    with torch.no_grad():
        # MoGe expects CHW float32 in [0,1]
        input_tensor = torch.tensor(rgb / 255.0, dtype=torch.float32, device=device).permute(2, 0, 1)
        out = MoGe.infer(input_tensor)

    # move to CPU numpy for further processing
    out = {k: (v.detach().cpu().numpy() if isinstance(v, torch.Tensor) else v) for k, v in out.items()}
    out["rgb"] = rgb

    points = out["points"]  # (H,W,3)
    depth = out["depth"]    # (H,W)
    mask = out["mask"]      # (H,W) bool

    # Depth visualization
    valid_depth = depth[np.isfinite(depth)]
    if valid_depth.size == 0:
        raise gr.Error("MoGe produced invalid depth for this image.")
    min_val = np.percentile(valid_depth, 5)
    max_val = np.percentile(valid_depth, 90)
    clipped = np.clip(depth, min_val, max_val)
    depth_im = 255 * (clipped - min_val) / (max_val - min_val + 1e-8)
    depth_vis = Image.fromarray(depth_im.astype(np.uint8))

    # Camera intrinsics in pixels
    K = out['intrinsics'].copy()
    K[0, 0] *= W
    K[1, 1] *= H
    K[0, 2] *= W
    K[1, 2] *= H

    # Prepare mesh from image points (mirror original utils3d path)
    # Flip YZ like your script
    pts = points.copy()
    pts[:, :, 1] = -pts[:, :, 1]
    pts[:, :, 2] = -pts[:, :, 2]

    # Recompute normals and edge masks to avoid tearing (matches your original pipeline)
    normals, normals_mask = utils3d.numpy.points_to_normals(points, mask=mask)
    depth_edge   = utils3d.numpy.depth_edge(depth, rtol=0.03, mask=mask)
    normals_edge = utils3d.numpy.normals_edge(normals, tol=5, mask=normals_mask)
    good_mask = mask & ~(depth_edge & normals_edge)

    faces, vertices, vertex_colors, vertex_uvs = utils3d.numpy.image_mesh(
        pts,
        rgb.astype(np.float32) / 255.0,
        utils3d.numpy.image_uv(width=W, height=H),
        mask=good_mask,
        tri=True,
    )

    mesh = trimesh.Trimesh(
        vertices=vertices,
        faces=faces,
        vertex_colors=(vertex_colors * 255).astype(np.uint8),
    )
    # Select camera center
    P_center = points[cy, cx]
    center = torch.tensor(P_center, device=device, dtype=torch.float32)
    center[1:3] = -center[1:3]
    center[0] = 0.0

    radius = float(P_center[2]) * float(dist_multiplier)
    extrinsics = get_tilted_camera(center, tilt_degrees=float(tilt_degrees), pullback_distance=radius, device=str(device))

    # Rasterize with Kaolin DIB-R
    if progress:
        progress(0.6, desc="Rasterizing tilted view…")

    # to CUDA
    vertices_t = torch.tensor(vertices, dtype=torch.float32, device=device)
    faces_t = torch.tensor(faces, dtype=torch.int64, device=device)
    vcolors_t = torch.tensor(vertex_colors[:, [2, 1, 0]], dtype=torch.float32, device=device)  # BGR->RGB swap like orig

    fx, fy = K[0, 0], K[1, 1]
    cxp, cyp = K[0, 2], K[1, 2]
    width, height = W, H

    # Project vertices under new camera
    V_h = torch.cat([vertices_t, torch.ones_like(vertices_t[:, :1])], dim=-1).T  # (4,V)
    vertices_cam = (extrinsics[0] @ V_h).T[:, :3]  # (V,3)

    u = fx * (vertices_cam[:, 0] / -vertices_cam[:, 2]) + cxp
    v = fy * (-vertices_cam[:, 1] / -vertices_cam[:, 2]) + cyp
    u_ndc = (u / width) * 2 - 1
    v_ndc = (v / height) * 2 - 1
    vertex_image_coords = torch.stack([u_ndc, v_ndc], dim=-1)

    face_vertices = vertices_t[faces_t]
    face_vertices_image = vertex_image_coords[faces_t].unsqueeze(0)  # (1,F,3,2)

    # Z and normals for DIB-R
    face_vertices_z = face_vertices[:, :, 2].unsqueeze(0)
    v0 = face_vertices[:, 1] - face_vertices[:, 0]
    v1 = face_vertices[:, 2] - face_vertices[:, 0]
    face_normals = torch.cross(v0, v1)
    face_normals_z = face_normals[:, 2].unsqueeze(0)

    face_features = vcolors_t[faces_t].unsqueeze(0)  # (1,F,3,3)

    rendered_features, soft_mask, face_idx = kal.render.mesh.dibr_rasterization(
        height=512,
        width=512,
        face_vertices_z=face_vertices_z,
        face_vertices_image=face_vertices_image,
        face_features=face_features,
        face_normals_z=face_normals_z,
    )

    # Convert to images
    image_np = rendered_features[0].detach().cpu().numpy()[::-1]
    image_np = (image_np * 255.0).astype(np.uint8)
    image_np = cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB)
    warped = Image.fromarray(resize_and_pad_to_square(image_np, 512))

    mask_np = (soft_mask[0].detach().cpu().numpy() > 0.99).astype(np.uint8)[::-1]
    mask_np = resize_and_pad_to_square(mask_np, 512)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask_np = cv2.erode(mask_np, kernel, iterations=2)
    edit_mask = Image.fromarray((mask_np * 255).astype(np.uint8))

    mask_for_pipe = ImageOps.invert(edit_mask.convert("L"))

    if progress:
        progress(0.8, desc="Running inpainting…")

    with torch.autocast("cuda", enabled=torch.cuda.is_available()):
        result = pipe(prompt=prompt, image=warped, mask_image=mask_for_pipe)
    inpainted = result.images[0]

    log_lines.append(f"fx, fy, cx, cy = {fx:.2f}, {fy:.2f}, {cxp:.2f}, {cyp:.2f}")
    log_lines.append(f"tilt = {tilt_degrees}°, radius = {radius:.3f}, device = {device}")

    # Clean up VRAM a bit
    torch.cuda.empty_cache() if torch.cuda.is_available() else None
    gc.collect()

    return (
        Image.fromarray(annotated),
        depth_vis,
        warped,
        edit_mask,
        inpainted,
        "\n".join(log_lines),
    )


# ------------------------- Gradio UI ------------------------------

def build_ui():
    with gr.Blocks(title="Perspective Change (MoGe + SD Inpaint)") as demo:
        gr.Markdown("""# Perspective Change
Upload an image, choose subject (for bbox), set tilt and distance, and provide an inpainting prompt.
""")
        with gr.Row():
            with gr.Column(scale=1):
                in_image = gr.Image(type="pil", label="Upload Image")
                subject = gr.Textbox(label="Subject to focus on", value="")
                prompt = gr.Textbox(label="Image Caption", value="high-quality photorealistic scene")
                tilt = gr.Slider(-10, 90, value=0, step=1, label="Tilt (degrees)")
                dist_mult = gr.Slider(0.5, 3.0, value=1.5, step=0.05, label="Distance Multiplier")
                with gr.Accordion("Advanced", open=False):
                    box_thr = gr.Slider(0.05, 0.9, value=0.35, step=0.01, label="DINO Box Threshold")
                    txt_thr = gr.Slider(0.05, 0.9, value=0.25, step=0.01, label="DINO Text Threshold")
                    dino_cfg = gr.Textbox(label="GroundingDINO Config Path", value=DEFAULT_DINO_CFG)
                    dino_wts = gr.Textbox(label="GroundingDINO Weights Path", value=DEFAULT_DINO_WEIGHTS)
                run_btn = gr.Button("Run", variant="primary")

            with gr.Column(scale=1):
                out_bbox = gr.Image(label="Annotated BBox")
                out_depth = gr.Image(label="Depth (viz)")

            with gr.Column(scale=1):
                out_warped = gr.Image(label="Warped View (Kaolin)")
                out_mask = gr.Image(label="Edit Mask")

        out_final = gr.Image(label="Inpainted Result", height=512)
        out_log = gr.Textbox(label="Log", lines=6)

        # --- Examples: place a few sample images under ./examples/ ---
        with gr.Accordion("Examples", open=False):
            gr.Examples(
                label="Click to load (and run)",
                examples=[
                    [
                        "examples/69.png",
                        "Rose in a vase",
                        'A vibrant rose in a clear vase by a window, with a fallen petal and a smaller rose lying on the wooden surface.',
                        0.35,
                        0.25,
                        45,
                        1.5,
                        DEFAULT_DINO_CFG,
                        DEFAULT_DINO_WEIGHTS,
                    ],
                    [
                        "examples/117.png",
                        "Teapot",
                        'A silver teapot with intricate floral patterns sits on a polished wooden table, reflecting on the surface, with a dark paneled wall and a draped curtain in the background.',
                        0.35,
                        0.25,
                        45,
                        1.5,
                        DEFAULT_DINO_CFG,
                        DEFAULT_DINO_WEIGHTS,
                    ],
                    [
                        "examples/129.png",
                        'Wooden canoe',
                        'A wooden canoe floats on a calm lake at sunrise, with the sky and clouds reflecting orange and pink hues, and a mist rising from the water surface.',
                        0.35,
                        0.25,
                        45,
                        1.5,
                        DEFAULT_DINO_CFG,
                        DEFAULT_DINO_WEIGHTS,
                    ],
                    [
                        "examples/201.png",
                        'Elderly woman in traditional dress',
                        'An elderly woman in traditional Korean dress sitting on a bench with cherry blossoms in full bloom around her. The scene depicts a vibrant spring atmosphere with a clear sky, a body of water reflecting the blossoms, and a bridge in the background.',
                        0.35,
                        0.25,
                        45,
                        1.5,
                        DEFAULT_DINO_CFG,
                        DEFAULT_DINO_WEIGHTS,
                    ],
                    [
                        "examples/237.png",
                        "The pond",
                        'A cold pond with swans, plants covered in snow, and surrounding foliage in winter.',
                        0.35,
                        0.25,
                        45,
                        1.5,
                        DEFAULT_DINO_CFG,
                        DEFAULT_DINO_WEIGHTS,
                    ],
                    [
                        "examples/1521.png",
                        'Young girl in savannah',
                        'A young girl stands in the savannah during the daytime, with the sun high and bright, casting a clear light over the scene.',
                        0.35,
                        0.25,
                        45,
                        1.5,
                        DEFAULT_DINO_CFG,
                        DEFAULT_DINO_WEIGHTS,
                    ],
                    [
                        "examples/1533.png",
                        'Canoe on the lake',
                        'A serene mountain lake surrounded by dense forests with a clear view of the lakebed and a canoe on the surface',
                        0.35,
                        0.25,
                        60,
                        1.5,
                        DEFAULT_DINO_CFG,
                        DEFAULT_DINO_WEIGHTS,
                    ],
                ],
                inputs=[
                    in_image,
                    subject,
                    prompt,
                    box_thr,
                    txt_thr,
                    tilt,
                    dist_mult,
                    dino_cfg,
                    dino_wts,
                ],
                outputs=[
                    out_bbox,
                    out_depth,
                    out_warped,
                    out_mask,
                    out_final,
                    out_log,
                ],
                fn=run_pipeline,
                cache_examples=True,
            )

        run_btn.click(
            fn=run_pipeline,
            inputs=[
                in_image, subject, prompt, box_thr, txt_thr, tilt, dist_mult, dino_cfg, dino_wts
            ],
            outputs=[out_bbox, out_depth, out_warped, out_mask, out_final, out_log],
            api_name="run",
        )

        gr.Markdown("""
### Setup tips
- Ensure **GroundingDINO** weights exist at the given path (default: `model_cache/groundingdino_swint_ogc.pth`).
- Set environment variable `MOGE_DEVICE` to select a GPU (e.g., `cuda:1`). Defaults to `cuda:0` if available.
- Login to Hugging Face if needed for the SD2 Inpainting model.
- If VRAM is tight, reduce the image resolution before upload.
""")
    return demo


if __name__ == "__main__":
    demo = build_ui()
    demo.launch(server_name="127.0.0.1", server_port=7860)