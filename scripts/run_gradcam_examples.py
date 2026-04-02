import json
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

import torch
import torch.nn.functional as F

from src.models_cnn.predict_efficientnet_frames import load_model, get_device


# ---- SETTINGS ----
CKPT_PATH = "outputs/models_cnn/efficientnet_b0_linearprobe.pt"
DEVICE = get_device("")
OUT_DIR = Path("outputs/xai/gradcam")
OUT_DIR.mkdir(parents=True, exist_ok=True)

VIDEO_TAGS = [
    "fake_033",
    "fake_017",
    "real_042",
    "real_049",
    "fake_019",
    "real_005",
    "real_008",
]

BASE_DIR = Path("outputs/frames_faces")


def get_mid_frame(video_tag: str):
    frames = sorted((BASE_DIR / video_tag).glob("*.jpg"))
    if not frames:
        return None
    return frames[len(frames) // 2]


def compute_gradcam(backbone, head, tfm, image_path, device):
    img_pil = Image.open(image_path).convert("RGB")
    orig = np.array(img_pil)

    x = tfm(img_pil).unsqueeze(0).to(device)
    x.requires_grad_(True)

    backbone.eval()
    head.eval()

    feat_maps = backbone.forward_features(x)   # [1, C, H, W]
    feat_maps.retain_grad()

    pooled = F.adaptive_avg_pool2d(feat_maps, 1).flatten(1)  # [1, C]
    logits = head(pooled)                                    # [1, 1]
    prob_fake = torch.sigmoid(logits)[0, 0]

    backbone.zero_grad()
    head.zero_grad()
    prob_fake.backward()

    grads = feat_maps.grad[0]   # [C, H, W]
    fmap = feat_maps[0]         # [C, H, W]

    weights = grads.mean(dim=(1, 2))
    cam = torch.zeros(fmap.shape[1:], device=device)

    for i, w in enumerate(weights):
        cam += w * fmap[i]

    cam = torch.relu(cam)
    cam = cam.detach().cpu().numpy()

    cam = cam - cam.min()
    cam = cam / (cam.max() + 1e-8)

    cam_resized = cv2.resize(cam, (orig.shape[1], orig.shape[0]))

    heatmap = np.uint8(255 * cam_resized)
    heatmap_color = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

    overlay = cv2.addWeighted(
        cv2.cvtColor(orig, cv2.COLOR_RGB2BGR), 0.6, heatmap_color, 0.4, 0
    )

    return orig, heatmap_color, overlay, float(prob_fake.item())


def main():
    backbone, head, tfm = load_model(CKPT_PATH, DEVICE)

    for tag in VIDEO_TAGS:
        p = get_mid_frame(tag)
        if p is None:
            print(f"Skipping missing video/tag: {tag}")
            continue

        orig, heatmap, overlay, prob = compute_gradcam(backbone, head, tfm, p, DEVICE)

        stem = f"{p.parent.name}_{p.stem}"

        cv2.imwrite(str(OUT_DIR / f"{stem}_heatmap.png"), heatmap)
        cv2.imwrite(str(OUT_DIR / f"{stem}_overlay.png"), overlay)
        cv2.imwrite(str(OUT_DIR / f"{stem}_original.png"), cv2.cvtColor(orig, cv2.COLOR_RGB2BGR))

        meta = {
            "video_tag": tag,
            "image_path": str(p),
            "p_fake": prob,
        }
        with open(OUT_DIR / f"{stem}_meta.json", "w") as f:
            json.dump(meta, f, indent=2)

        print(f"Saved Grad-CAM for {p} | p_fake={prob:.4f}")

    print(f"\nDone. Outputs saved to: {OUT_DIR}")


if __name__ == "__main__":
    main()