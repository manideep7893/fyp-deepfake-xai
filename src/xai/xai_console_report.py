# src/xai/xai_console_report.py
# ------------------------------------------------------------
# Deepfake XAI console report (Grad-CAM + region attribution + faithfulness)
#
# Usage:
#   python src/xai/xai_console_report.py \
#     --frame outputs/frames/test_fake/frame_000001.jpg \
#     --model_id prithivMLmods/Deep-Fake-Detector-Model \
#     --device mps \
#     --out_dir outputs/xai/test_fake
#
# Output:
#   - prints console report (Prediction + Salient regions + Faithfulness + Explanation)
#   - saves: <out_dir>/<frame_name>_gradcam.jpg
#   - saves: <out_dir>/<frame_name>_xai_report.json
# ------------------------------------------------------------

import argparse
import json
from pathlib import Path

import numpy as np
import cv2
from PIL import Image

import torch
import torch.nn as nn
from transformers import AutoImageProcessor, AutoModelForImageClassification

from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image


# ----------------------------
# Helpers: device
# ----------------------------
def get_torch_device(device_str: str) -> torch.device:
    device_str = device_str.lower()
    if device_str == "mps":
        if not torch.backends.mps.is_available():
            raise RuntimeError("MPS requested but not available. Use --device cpu")
        return torch.device("mps")
    if device_str == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but not available. Use --device cpu or mps")
        return torch.device("cuda")
    return torch.device("cpu")


# ----------------------------
# HF model wrapper for Grad-CAM:
# Grad-CAM expects model forward -> Tensor (logits)
# ----------------------------
class HFLogitsWrapper(nn.Module):
    def __init__(self, hf_model: nn.Module):
        super().__init__()
        self.model = hf_model

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x is pixel_values: [B,3,H,W]
        out = self.model(pixel_values=x)
        return out.logits


# ----------------------------
# ViT/SigLIP reshape transform:
# tokens [B,N,C] -> [B,C,H,W]
# ----------------------------
def reshape_transform(tensor: torch.Tensor) -> torch.Tensor:
    # tensor: [B, N, C]
    if tensor.dim() != 3:
        # already in conv format or something else
        return tensor

    B, N, C = tensor.shape
    s = int(N ** 0.5)

    if s * s == N:
        grid = tensor
        H = W = s
    else:
        # drop CLS if present
        s = int((N - 1) ** 0.5)
        if s * s != (N - 1):
            raise RuntimeError(f"Can't reshape tokens to square grid. N={N}")
        grid = tensor[:, 1:, :]
        H = W = s

    grid = grid.reshape(B, H, W, C)              # [B,H,W,C]
    grid = grid.permute(0, 3, 1, 2).contiguous() # [B,C,H,W]
    return grid


# ----------------------------
# Pick a reasonable target layer for transformer vision models
# Works for many HF image classifiers incl SigLIP/ViT.
# ----------------------------
def pick_target_layer(hf_model: nn.Module) -> nn.Module:
    # SigLIP: often model.vision_model.encoder.layers[-1].layer_norm2
    if hasattr(hf_model, "vision_model"):
        vm = hf_model.vision_model
        if hasattr(vm, "encoder") and hasattr(vm.encoder, "layers"):
            last = vm.encoder.layers[-1]
            if hasattr(last, "layer_norm2"):
                return last.layer_norm2
            return last

    # ViT-like: base_model.vision_model.encoder.layers...
    base = getattr(hf_model, "base_model", None)
    if base is not None and hasattr(base, "vision_model"):
        vm = base.vision_model
        if hasattr(vm, "encoder") and hasattr(vm.encoder, "layers"):
            last = vm.encoder.layers[-1]
            if hasattr(last, "layer_norm2"):
                return last.layer_norm2
            return last

    # Fallback: try to find last LayerNorm or last module
    for m in reversed(list(hf_model.modules())):
        name = m.__class__.__name__.lower()
        if "layernorm" in name or name == "layernorm":
            return m
    # absolute fallback
    return list(hf_model.modules())[-1]


# ----------------------------
# Region masks via MediaPipe (FaceMesh)
# Mouth / left-eye / face boundary percentages from CAM heatmap
# If mediapipe isn't available, falls back to "whole image".
# ----------------------------
def build_region_masks(img_bgr: np.ndarray):
    H, W = img_bgr.shape[:2]
    mouth_mask = np.zeros((H, W), dtype=np.uint8)
    leye_mask = np.zeros((H, W), dtype=np.uint8)
    face_mask = np.ones((H, W), dtype=np.uint8)  # default: whole image

    try:
        import mediapipe as mp  # type: ignore
        mp_face = mp.solutions.face_mesh

        # FaceMesh expects RGB
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        with mp_face.FaceMesh(
            static_image_mode=True,
            refine_landmarks=True,
            max_num_faces=1,
            min_detection_confidence=0.5
        ) as fm:
            res = fm.process(img_rgb)

        if not res.multi_face_landmarks:
            return mouth_mask, leye_mask, face_mask, False

        lm = res.multi_face_landmarks[0].landmark

        def pts(indices):
            arr = []
            for i in indices:
                x = int(np.clip(lm[i].x * W, 0, W - 1))
                y = int(np.clip(lm[i].y * H, 0, H - 1))
                arr.append([x, y])
            return np.array(arr, dtype=np.int32)

        # MediaPipe landmark groups (good-enough, stable)
        # Mouth: outer lips ring-ish
        MOUTH = [61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291]
        # Left eye: around left eye
        LEFT_EYE = [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246]
        # Face boundary (oval): use face oval indices
        FACE_OVAL = [10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288, 397, 365, 379,
                     378, 400, 377, 152, 148, 176, 149, 150, 136, 172, 58, 132, 93, 234, 127,
                     162, 21, 54, 103, 67, 109]

        cv2.fillPoly(mouth_mask, [pts(MOUTH)], 1)
        cv2.fillPoly(leye_mask, [pts(LEFT_EYE)], 1)

        face_mask = np.zeros((H, W), dtype=np.uint8)
        cv2.fillPoly(face_mask, [pts(FACE_OVAL)], 1)

        # "Face boundary" = ring near edge of face oval (band)
        # We'll approximate boundary by erode face mask and subtract.
        k = max(3, int(min(H, W) * 0.02))  # ~2% thickness
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
        inner = cv2.erode(face_mask, kernel, iterations=1)
        boundary_mask = (face_mask.astype(np.int16) - inner.astype(np.int16))
        boundary_mask = (boundary_mask > 0).astype(np.uint8)

        return mouth_mask, leye_mask, boundary_mask, True

    except Exception:
        # mediapipe not installed or face not detected
        return mouth_mask, leye_mask, face_mask, False


def region_percentages(cam: np.ndarray, img_bgr: np.ndarray):
    """
    cam: [H,W] float in [0,1] (already resized to image)
    returns dict region -> percentage (0-100)
    """
    mouth_mask, leye_mask, boundary_mask, ok = build_region_masks(img_bgr)

    eps = 1e-9
    total = float(cam.sum() + eps)

    def pct(mask):
        if mask.sum() == 0:
            return 0.0
        return 100.0 * float((cam * mask).sum()) / total

    return {
        "mouth_pct": pct(mouth_mask),
        "left_eye_pct": pct(leye_mask),
        "face_boundary_pct": pct(boundary_mask),
        "has_face_mesh": ok
    }


# ----------------------------
# Faithfulness (Insertion/Deletion AUC) — lightweight but legit
# We progressively insert/delete top-cam pixels and track P(fake).
# ----------------------------
def trapz(y, x):
    y = np.asarray(y, dtype=np.float64)
    x = np.asarray(x, dtype=np.float64)
    return np.sum((x[1:] - x[:-1]) * (y[1:] + y[:-1]) * 0.5)

def faithfulness_auc(
    wrapper_model: nn.Module,
    inputs_pixel_values: torch.Tensor,
    cam: np.ndarray,
    target_class: int,
    device: torch.device,
    steps: int = 20
):
    """
    inputs_pixel_values: [1,3,H,W] float tensor
    cam: [H,W] in [0,1]
    returns deletion_auc, insertion_auc
    """
    with torch.no_grad():
        base_logits = wrapper_model(inputs_pixel_values)
        base_prob = torch.softmax(base_logits, dim=1)[0, target_class].item()

    x = inputs_pixel_values.detach().clone()
    x_np = x[0].permute(1, 2, 0).cpu().numpy()  # H,W,3 in [0,1] (processor usually makes 0-1)
    # --- IMPORTANT: CAM must match image H,W for masking ---
    cam = np.squeeze(cam)  # make sure cam is 2D (H,W)

    H_img, W_img = x_np.shape[:2]

    # resize CAM to image size (W, H)
    cam = cv2.resize(cam.astype(np.float32), (W_img, H_img), interpolation=cv2.INTER_LINEAR)

    # normalize to [0,1]
    cam = cam - cam.min()
    cam = cam / (cam.max() + 1e-8)

    H, W = cam.shape

    flat = cam.reshape(-1)
    order = np.argsort(-flat)  # descending

    # Create masks for fractions
    probs_del = []
    probs_ins = []

    # Deletion: start from original, replace top pixels with baseline (blur)
    baseline = cv2.GaussianBlur(x_np, (0, 0), sigmaX=7)

    # Insertion: start from baseline, insert top pixels from original
    cur_del = x_np.copy()
    cur_ins = baseline.copy()

    N = H * W
    for i in range(steps + 1):
        frac = i / steps
        k = int(frac * N)
        idx = order[:k]

        mask = np.zeros(N, dtype=np.uint8)
        mask[idx] = 1
        mask = mask.reshape(H, W)

        # deletion: masked pixels -> baseline
        cur_del = x_np * (1 - mask[..., None]) + baseline * (mask[..., None])

        # insertion: masked pixels -> original (start baseline)
        cur_ins = baseline * (1 - mask[..., None]) + x_np * (mask[..., None])

        def prob(img_hw3):
            t = torch.from_numpy(img_hw3).float().permute(2, 0, 1).unsqueeze(0).to(device)
            with torch.no_grad():
                logits = wrapper_model(t)
                return torch.softmax(logits, dim=1)[0, target_class].item()

        probs_del.append(prob(cur_del))
        probs_ins.append(prob(cur_ins))

    # AUC via trapezoid rule over fractions [0..1]
    xs = np.linspace(0, 1, steps + 1)
    deletion_auc = float(trapz(probs_del, xs))
    insertion_auc = float(trapz(probs_ins, xs))

    # Optional: interpret deletion confidence drop (base - end)
    return base_prob, deletion_auc, insertion_auc


# ----------------------------
# Simple explanation templating
# ----------------------------
def make_explanation(pred_label: str, region_pcts: dict, has_face: bool):
    if not has_face:
        return (
            "No face landmarks were detected reliably, so the explanation is based on general "
            "image regions rather than facial sub-regions."
        )

    # pick top regions by percentage
    regions = [
        ("mouth", region_pcts["mouth_pct"]),
        ("left eye", region_pcts["left_eye_pct"]),
        ("face boundary", region_pcts["face_boundary_pct"]),
    ]
    regions_sorted = sorted(regions, key=lambda x: x[1], reverse=True)
    top_names = [r[0] for r in regions_sorted if r[1] > 0][:2]
    if not top_names:
        top_names = ["face region"]

    return (
        f"The model’s prediction relies primarily on localized artefacts around the "
        f"{' and '.join(top_names)} regions, which are common areas of manipulation in "
        f"deepfake generation."
    )


# ----------------------------
# Main
# ----------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--frame", required=True, help="Path to a single frame (.jpg/.png)")
    ap.add_argument("--model_id", default="prithivMLmods/Deep-Fake-Detector-Model", help="HF model id")
    ap.add_argument("--device", default="mps", choices=["mps", "cpu", "cuda"])
    ap.add_argument("--out_dir", required=True, help="Output folder for CAM + report")
    ap.add_argument("--target_class", type=int, default=0, help="Target class index for Grad-CAM / faithfulness (0=Fake for this model)")
    ap.add_argument("--faith_steps", type=int, default=20, help="Steps for insertion/deletion AUC")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    device = get_torch_device(args.device)

    # Load model + processor
    processor = AutoImageProcessor.from_pretrained(args.model_id)
    hf_model = AutoModelForImageClassification.from_pretrained(args.model_id)
    hf_model.to(device)
    hf_model.eval()

    wrapper = HFLogitsWrapper(hf_model).to(device).eval()

    # Load image
    frame_path = Path(args.frame)
    if not frame_path.exists():
        raise FileNotFoundError(f"Frame not found: {frame_path}")

    image = Image.open(frame_path).convert("RGB")
    img_np = np.array(image).astype(np.float32) / 255.0  # RGB [0,1]
    img_bgr = cv2.cvtColor((img_np * 255).astype(np.uint8), cv2.COLOR_RGB2BGR)

    inputs = processor(images=image, return_tensors="pt")
    pixel_values = inputs["pixel_values"].to(device)

    with torch.no_grad():
        logits = hf_model(**{k: v.to(device) for k, v in inputs.items()}).logits
    probs = torch.softmax(logits, dim=1)[0].detach().cpu().numpy()

    id2label = {int(k): str(v) for k, v in hf_model.config.id2label.items()}

    # Use model's own predicted label for REAL/FAKE
    pred_idx = int(probs.argmax())
    pred_label = id2label.get(pred_idx, str(pred_idx)).upper()
    confidence = float(probs[pred_idx])

    # Always compute P(fake) using the model's own label mapping
    fake_idx = None
    for i, name in id2label.items():
        if "fake" in name.lower():
            fake_idx = i
            break

    fake_prob = float(probs[fake_idx]) if fake_idx is not None else None

    # Grad-CAM
    target_layer = pick_target_layer(hf_model)

    cam = GradCAM(
        model=wrapper,
        target_layers=[target_layer],
        reshape_transform=reshape_transform
    )

    targets = [ClassifierOutputTarget(args.target_class)]
    grayscale_cam = cam(input_tensor=pixel_values, targets=targets)[0]  # [h,w] in [0,1] typically
    grayscale_cam = np.clip(grayscale_cam, 0, 1)

    # Resize CAM to image size (for region masks + overlay)
    H, W = img_np.shape[:2]
    cam_resized = cv2.resize(grayscale_cam, (W, H), interpolation=cv2.INTER_LINEAR)

    cam_overlay = show_cam_on_image(img_np, cam_resized, use_rgb=True)  # RGB uint8
    cam_overlay_path = out_dir / f"{frame_path.stem}_gradcam.jpg"
    cv2.imwrite(str(cam_overlay_path), cv2.cvtColor(cam_overlay, cv2.COLOR_RGB2BGR))

    # Region attribution
    pcts = region_percentages(cam_resized, img_bgr)
    has_face = bool(pcts.get("has_face_mesh", False))

    # Faithfulness
    base_prob, deletion_auc, insertion_auc = faithfulness_auc(
        wrapper_model=wrapper,
        inputs_pixel_values=pixel_values,
        cam=cam_resized,
        target_class=args.target_class,
        device=device,
        steps=args.faith_steps
    )

    explanation = make_explanation(pred_label, pcts, has_face)

    # Console output (matches your target style)
    print(f"\nPrediction: {pred_label} (confidence: {confidence:.2f})\n")

    print("Salient regions:")
    print(f"- Mouth region: {pcts['mouth_pct']:.0f}%")
    print(f"- Left eye region: {pcts['left_eye_pct']:.0f}%")
    print(f"- Face boundary: {pcts['face_boundary_pct']:.0f}%\n")

    print("Faithfulness score:")
    print(f"- Deletion AUC: {deletion_auc:.2f} (sharp confidence drop)" if deletion_auc < insertion_auc else f"- Deletion AUC: {deletion_auc:.2f}")
    print(f"- Insertion AUC: {insertion_auc:.2f}\n")

    print("Explanation:")
    print(explanation)
    print()

    # Save report JSON
    report = {
        "frame": str(frame_path),
        "model_id": args.model_id,
        "device": str(device),
        "prediction": {
            "label": pred_label,
            "fake_probability": fake_prob,
            "confidence": confidence,
            "target_class_index": args.target_class
        },
        "salient_regions_pct": {
            "mouth": pcts["mouth_pct"],
            "left_eye": pcts["left_eye_pct"],
            "face_boundary": pcts["face_boundary_pct"],
            "used_facemesh": has_face
        },
        "faithfulness": {
            "base_p_fake": base_prob,
            "deletion_auc": deletion_auc,
            "insertion_auc": insertion_auc,
            "steps": args.faith_steps
        },
        "artifacts": {
            "gradcam_image": str(cam_overlay_path)
        },
        "explanation": explanation
    }

    report_path = out_dir / f"{frame_path.stem}_xai_report.json"
    report_path.write_text(json.dumps(report, indent=2))
    # Done
    # print(f"Saved: {cam_overlay_path}")
    # print(f"Saved: {report_path}")


if __name__ == "__main__":
    main()