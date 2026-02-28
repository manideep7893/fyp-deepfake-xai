import torch
import timm
from PIL import Image
import torchvision.transforms as T
from pathlib import Path

def main():
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print("Device:", device)

    # EfficientNet-B0 pretrained on ImageNet (smoke test only)
    model = timm.create_model("efficientnet_b0", pretrained=True)
    model.eval().to(device)

    # find one existing face crop from your outputs
    sample = next(Path("outputs/frames_faces").rglob("*.jpg"), None)
    if sample is None:
        raise FileNotFoundError("No face crops found under outputs/frames_faces. Run extraction first.")
    print("Sample image:", sample)

    tfm = T.Compose([
        T.Resize((224, 224)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    img = Image.open(sample).convert("RGB")
    x = tfm(img).unsqueeze(0).to(device)

    with torch.no_grad():
        logits = model(x)
        probs = torch.softmax(logits, dim=1)
        top = torch.topk(probs, k=5, dim=1)

    print("Top-5 ImageNet classes (ids):", top.indices.cpu().tolist()[0])
    print("Top-5 probs:", [round(float(p), 4) for p in top.values.cpu().tolist()[0]])
    print("✅ Smoke test passed (CNN runs on your machine).")

if __name__ == "__main__":
    main()