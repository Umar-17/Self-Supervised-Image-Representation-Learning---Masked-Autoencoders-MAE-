import torch
import torchvision.transforms as T
import gradio as gr
import numpy as np
from PIL import Image
from torchmetrics.image import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure
from model_lib import MaskedAutoencoder 

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = MaskedAutoencoder().to(device)
weights_path = "mae_weights_40ep.pth"

try:
    model.load_state_dict(torch.load(weights_path, map_location=device))
    model.eval()
    print("Model loaded successfully.")
except FileNotFoundError:
    print(f"Error: {weights_path} not found. Upload it directly to the HF Space.")

transform = T.Compose([
    T.Resize((224, 224)),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

def patchify(imgs):
    p = 16
    h = w = imgs.shape[2] // p
    x = imgs.reshape(shape=(imgs.shape[0], 3, h, p, w, p))
    x = torch.einsum('nchpwq->nhwpqc', x)
    return x.reshape(shape=(imgs.shape[0], h * w, p**2 * 3))

def unpatchify(x):
    p = 16
    h = w = int(x.shape[1]**0.5)
    x = x.reshape(shape=(x.shape[0], h, w, p, p, 3))
    x = torch.einsum('nhwpqc->nchpwq', x)
    return x.reshape(shape=(x.shape[0], 3, h * p, h * p))

psnr_metric = PeakSignalNoiseRatio(data_range=1.0).to(device)
ssim_metric = StructuralSimilarityIndexMeasure(data_range=1.0).to(device)

def predict(input_image, mask_ratio):
    if input_image is None: return None, None, "Upload an image."
    img_tensor = transform(input_image).unsqueeze(0).to(device)
    
    with torch.no_grad():
        pred, mask = model(img_tensor, mask_ratio=mask_ratio)
    
    target = patchify(img_tensor)
    mean_p, var_p = target.mean(dim=-1, keepdim=True), target.var(dim=-1, keepdim=True)
    pred_unnorm = pred * (var_p + 1.e-6)**.5 + mean_p
    mask_vis = mask.unsqueeze(-1).repeat(1, 1, 16**2 * 3)
    recon_imgs = unpatchify(target * (1 - mask_vis) + pred_unnorm * mask_vis)
    
    # Denormalize
    mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)
    orig_denorm = torch.clamp(img_tensor * std + mean, 0, 1)
    recon_denorm = torch.clamp(recon_imgs * std + mean, 0, 1)
    
    metrics = f"PSNR: {psnr_metric(recon_denorm, orig_denorm).item():.2f} dB\nSSIM: {ssim_metric(recon_denorm, orig_denorm).item():.4f}"
    return orig_denorm[0].cpu().permute(1,2,0).numpy(), recon_denorm[0].cpu().permute(1,2,0).numpy(), metrics

demo = gr.Interface(
    fn=predict,
    inputs=[gr.Image(type="pil"), gr.Slider(0.1, 0.9, value=0.75, label="Mask Ratio")],
    outputs=[gr.Image(label="Original"), gr.Image(label="Reconstructed"), gr.Textbox(label="Metrics")],
    title="MAE Image Reconstruction"
)
demo.launch()