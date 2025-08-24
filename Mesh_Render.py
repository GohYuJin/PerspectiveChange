import cv2
import kaolin as kal
import utils3d
import trimesh
import numpy as np
import torch
import math
import matplotlib.pyplot as plt
from MoGe.moge.model.v1 import MoGeModel
import PIL
import pandas as pd
from io import BytesIO
import ast, os
from groundingdino.util.inference import load_model, predict, annotate, Model
from kaolin.render.camera import Camera, PinholeIntrinsics
from diffusers import StableDiffusionPipeline, StableDiffusionInpaintPipeline, StableDiffusion3InpaintPipeline
from diffusers import AutoPipelineForImage2Image, AutoPipelineForInpainting, RePaintScheduler


def get_tilted_camera(center, tilt_degrees=0, pullback_distance=2.0, device='cuda'):
    """
    Generate a *camera-to-mesh* extrinsic transform suitable for MoGe mesh,
    assuming the mesh is in camera space (as from MoGe).
    """

    center = center.to(device)
    up = torch.tensor([0, 1, 0], device=device, dtype=torch.float32)
    forward = torch.tensor([0, 0, 1], device=device, dtype=torch.float32)

    # Apply tilt around X
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
    z = z / torch.norm(z)
    x = torch.cross(up, z)
    x = x / torch.norm(x)
    y = torch.cross(z, x)

    R = torch.stack([x, y, z], dim=1)

    # Final view matrix (world-to-camera): [R.T | -R.T @ eye]
    view = torch.eye(4, device=device)
    view[:3, :3] = R.T
    view[:3, 3] = -(R.T @ eye)
    return view.unsqueeze(0)

def resize_and_pad_to_square(img, target_long_side=512):
    """
    Resize the image so that the longer side becomes target_long_side,
    and pad it with zeros (black) to make it square.

    Args:
        img (np.ndarray): Input image (H x W x C or H x W).
        target_long_side (int): Desired size for the longer side after resize.

    Returns:
        padded_img (np.ndarray): Square image of shape (target_long_side, target_long_side, C).
        scale_factor (float): The factor the image was scaled by.
    """
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
        padded_img = cv2.copyMakeBorder(resized, pad_top, pad_bottom, pad_left, pad_right,
                                        borderType=cv2.BORDER_CONSTANT, value=(0, 0, 0))
    else:
        padded_img = cv2.copyMakeBorder(resized, pad_top, pad_bottom, pad_left, pad_right,
                                        borderType=cv2.BORDER_CONSTANT, value=0)

    return padded_img


def transform_points_world_to_cam(points: np.ndarray, extrinsic: np.ndarray) -> np.ndarray:
    """
    Apply a world-to-camera extrinsic transform (i.e., view matrix) to points.
    """
    N = points.shape[0]
    points_h = np.hstack([points, np.ones((N, 1))])  # (N, 4)
    transformed_h = (extrinsic @ points_h.T).T       # (N, 4)
    return transformed_h[:, :3]

device = torch.device("cuda:1")
MoGe = MoGeModel.from_pretrained("Ruicheng/moge-vitl").to(device)

pipe = StableDiffusionInpaintPipeline.from_pretrained(
    "stabilityai/stable-diffusion-2-inpainting",
    torch_dtype=torch.float16,
)

pipe.to(device)

GroundingDINO = load_model("GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py", "model_cache/groundingdino_swint_ogc.pth")

df = pd.read_csv(r"/home/yujing/code/HQ-Edit/MMML_project/Data/High-Low/high_low_final_w_idx_subject_degrees.csv")
df_filtered = df[df["category"] == "perspective"]

file = open("edit_intstructions.txt", "w")
file.close()

dist_multiplier = 1.5
BOX_TRESHOLD = 0.35
TEXT_TRESHOLD = 0.25

os.makedirs("Outputs2", exist_ok=True)

with torch.no_grad():
    for i in df_filtered.index:
        # if i != 1761:
        #     continue
        row = df.iloc[i]
        tilt_degrees = row["degrees"]
        # tilt_degrees = -60
        rgb = np.array(PIL.Image.open(BytesIO(ast.literal_eval(row["input_image"]))).convert('RGB'))
        H, W, _ = rgb.shape
        # input_caption = row["input"]
        edit_instruction = row["new_edit"]
        prompt = row["new_output"]
        subject = row["subject"]

        transformed_image = Model.preprocess_image(cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))

        boxes, logits, phrases = predict(
            model=GroundingDINO,
            image=transformed_image,
            caption=subject,
            box_threshold=BOX_TRESHOLD,
            text_threshold=TEXT_TRESHOLD
        )

        annotated_frame = annotate(image_source=rgb, boxes=boxes, logits=logits, phrases=phrases)
        if len(boxes) > 0:
            box = boxes[logits.argmax()]
            box = box * np.array([W, H, W, H])
            cx, cy = int(box[0]), int(box[1])
            cv2.imwrite("Outputs2/{0}_Bbox.jpg".format(i), annotated_frame)
        else:
            cx, cy = W//2, H//2
            cv2.imwrite("Outputs2/{0}_Bbox.jpg".format(i), annotated_frame)

        input_tensor = torch.tensor(rgb / 255, dtype=torch.float32, device=device).permute(2, 0, 1)
        output = MoGe.infer(input_tensor)
        output = {k: v.cpu().numpy() for k, v in output.items()}
        output["rgb"] = rgb
        points, depth, mask = output['points'], output['depth'], output['mask']

        valid_depth = depth[np.isfinite(depth)]

        min_val = np.percentile(valid_depth, 5)
        max_val = np.percentile(valid_depth, 90)

        clipped = np.clip(depth, min_val, max_val)
        depth_im = 255 * (clipped - min_val) / (max_val - min_val)
        cv2.imwrite("Outputs2/{0}_Depth.jpg".format(i), depth_im.astype(np.uint8))

        P_center = output["points"][cy, cx]

        # arr = np.array(output["points"])
         #np.percentile(arr[np.isfinite(arr)], 75)*2

        # Convert normalized intrinsics to pixel intrinsics
        K = output['intrinsics'].copy()
        K[0, 0] *= W
        K[1, 1] *= H
        K[0, 2] *= W
        K[1, 2] *= H
        
        
        normals, normals_mask = utils3d.numpy.points_to_normals(points, mask=mask)
        fov_x, fov_y = utils3d.numpy.intrinsics_to_fov(output['intrinsics'])
        fov_x, fov_y = np.rad2deg([fov_x, fov_y])


        points = output['points'].copy()
        points[:,:, 1] = -points[:,:, 1]
        points[:,:, 2] = -points[:,:, 2]

        faces, vertices, vertex_colors, vertex_uvs = utils3d.numpy.image_mesh(
            points,
            output['rgb'].astype(np.float32) / 255,
            utils3d.numpy.image_uv(width=W, height=H),
            mask=mask & ~(utils3d.numpy.depth_edge(output['depth'], rtol=0.03, mask=mask) &
                        utils3d.numpy.normals_edge(normals, tol=5, mask=normals_mask)),
            tri=True
        )

        
        mesh = trimesh.Trimesh(vertices=vertices, faces=faces, vertex_colors=(vertex_colors * 255).astype(np.uint8))

        # # Export to PLY
        mesh.export("debug_output_mesh.ply")

        # Inputs
        vertices = torch.tensor(vertices, dtype=torch.float32).cuda()        # (V, 3)
        faces = torch.tensor(faces, dtype=torch.int64).cuda()                # (F, 3)
        vertex_colors = torch.tensor(vertex_colors, dtype=torch.float32).cuda()  # (V, 3), in [0,1]
        vertex_colors = vertex_colors[:, [2, 1, 0]]

        # Define camera parameters

        fx, fy = K[0, 0], K[1, 1]
        cx, cy = K[0, 2], K[1, 2]
        width, height = W, H

        intrinsics = PinholeIntrinsics.from_focal(width=width, height=height, 
                                                focal_x=fx, focal_y=fy, 
                                                x0=cx / W * 2 - 1, y0=cy / H * 2 - 1, device='cuda')
        
        # Face data (F, 3, 3)
        face_vertices = vertices[faces]

        # Face features (colors)
        face_features = vertex_colors[faces]  # (F, 3, 3)

        # Z-buffer and image coords
        face_vertices_z = face_vertices[:, :, 2].unsqueeze(0)         # (1, F, 3)

        # Normals
        v0 = face_vertices[:, 1] - face_vertices[:, 0]
        v1 = face_vertices[:, 2] - face_vertices[:, 0]
        face_normals = torch.cross(v0, v1)
        face_normals_z = face_normals[:, 2].unsqueeze(0)              # (1, F)

        # Features
        face_features = face_features.unsqueeze(0)                    # (1, F, 3, 3)

        

        center = torch.tensor(P_center).to(device)
        center[1:3] = -center[1:3]
        center[0] = 0
        radius = P_center[2]*dist_multiplier
        extrinsics = get_tilted_camera(center, tilt_degrees=tilt_degrees, pullback_distance=radius)
        
        # Transform vertices to camera space
        V_h = torch.cat([vertices, torch.ones_like(vertices[:, :1])], dim=-1).T
        vertices_cam = (extrinsics[0] @ V_h).T[:, :3]

        # Project
        u = fx * (vertices_cam[:, 0] / -vertices_cam[:, 2]) + cx
        v = fy * (-vertices_cam[:, 1] / -vertices_cam[:, 2]) + cy
        u_ndc = (u / width) * 2 - 1
        v_ndc = (v / height) * 2 - 1
        vertex_image_coords = torch.stack([u_ndc, v_ndc], dim=-1)

        # Final per-face image coords
        face_vertices_image = vertex_image_coords[faces].unsqueeze(0)

        print("vertices (min/max):", vertices.min().item(), vertices.max().item())
        print("extrinsic:\n", extrinsics)
        print("fx fy cx cy:", fx, fy, cx, cy)

        print("camera center:", center)

        # Rasterize
        rendered_features, soft_mask, face_idx = kal.render.mesh.dibr_rasterization(
            height=512,
            width=512,
            face_vertices_z=face_vertices_z,
            face_vertices_image=face_vertices_image,
            face_features=face_features,
            face_normals_z=face_normals_z
        )

        # Convert rendered features to image
        image = rendered_features[0].cpu().numpy()
        image = image[::-1]
        image = (image * 255).astype(np.uint8)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = resize_and_pad_to_square(image, 512)

        ksize = 5
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ksize,ksize))
        mask = (soft_mask[0].cpu().numpy() > 0.99).astype(np.uint8)
        mask = mask[::-1]
        mask = resize_and_pad_to_square(mask, 512)
        mask = cv2.erode(mask, kernel, iterations=2)    

        init_image = PIL.Image.fromarray(image)
        init_image.save("Outputs2/{0}_Warped.png".format(i))

        mask_image = PIL.Image.fromarray(mask*255)
        mask_image.save("Outputs2/{0}_EditMask.png".format(i))

        # # Access outputs
        # rendered_rgb = (rendered['rgb'] * 255).astype(np.uint8)
        # rendered_mask = rendered['mask']  # optional for compositing

        # image = pipe(prompt=prompt, image=init_image, mask_image=mask_image).images[0]
        image = pipe(prompt=prompt, image=init_image, mask_image=PIL.ImageOps.invert(mask_image)).images[0]
        image.save("Outputs2/{0}_Inpainted.png".format(i))

        # refined = refiner(prompt=prompt, image=image)
        print(edit_instruction)

        # if i > 50:
        #     break