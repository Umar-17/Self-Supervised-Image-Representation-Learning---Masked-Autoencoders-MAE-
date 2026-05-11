import torch
import torch.nn as nn
import torchvision.transforms as transforms
import gradio as gr
import numpy as np
from PIL import Image
from torchmetrics.image import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure

class PatchEmbedding(nn.Module):
    def __init__(self, img_size=224, patch_size=16, in_channels=3, embed_dim=768):
        super().__init__()
        self.patch_size = patch_size
        self.num_patches = (img_size // patch_size) ** 2
        self.proj = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.pos_embed = nn.Parameter(torch.randn(1, self.num_patches, embed_dim) * 0.02)

    def forward(self, x):
        x = self.proj(x)
        x = x.flatten(2).transpose(1, 2)
        x = x + self.pos_embed
        return x

def random_masking(x, mask_ratio=0.75):
    B, N, D = x.shape
    num_keep = int(N * (1 - mask_ratio))
    noise = torch.rand(B, N, device=x.device)
    ids_shuffle = torch.argsort(noise, dim=1)
    ids_restore = torch.argsort(ids_shuffle, dim=1)
    ids_keep = ids_shuffle[:, :num_keep]
    x_visible = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).expand(-1, -1, D))
    mask = torch.ones([B, N], device=x.device)
    mask[:, :num_keep] = 0
    mask = torch.gather(mask, dim=1, index=ids_restore)
    return x_visible, mask, ids_restore

class MAEEncoder(nn.Module):
    def __init__(self, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4.0):
        super().__init__()
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, nhead=num_heads, dim_feedforward=int(embed_dim * mlp_ratio),
            activation='gelu', batch_first=True, norm_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=depth)
        self.norm = nn.LayerNorm(embed_dim)
    def forward(self, x):
        return self.norm(self.transformer(x))

class MAEDecoder(nn.Module):
    def __init__(self, encoder_dim=768, decoder_dim=384, depth=12, num_heads=6, num_patches=196, patch_size=16):
        super().__init__()
        self.proj = nn.Linear(encoder_dim, decoder_dim)
        self.mask_token = nn.Parameter(torch.randn(1, 1, decoder_dim) * 0.02)
        self.pos_embed = nn.Parameter(torch.randn(1, num_patches, decoder_dim) * 0.02)
        decoder_layer = nn.TransformerEncoderLayer(
            d_model=decoder_dim, nhead=num_heads, dim_feedforward=decoder_dim * 4,
            activation='gelu', batch_first=True, norm_first=True
        )
        self.transformer = nn.TransformerEncoder(decoder_layer, num_layers=depth)
        self.norm = nn.LayerNorm(decoder_dim)
        self.head = nn.Linear(decoder_dim, patch_size ** 2 * 3)

    def forward(self, x_visible, ids_restore):
        B, N_vis, _ = x_visible.shape
        x = self.proj(x_visible)
        num_missing = ids_restore.shape[1] - N_vis
        mask_tokens = self.mask_token.expand(B, num_missing, -1)
        x_full = torch.cat([x, mask_tokens], dim=1)
        x_full = torch.gather(x_full, dim=1, index=ids_restore.unsqueeze(-1).expand(-1, -1, x_full.shape[-1]))
        x_full = x_full + self.pos_embed
        x_full = self.norm(self.transformer(x_full))
        return self.head(x_full)

class MAE(nn.Module):
    def __init__(self, mask_ratio=0.75):
        super().__init__()
        self.mask_ratio = mask_ratio
        self.patch_embed = PatchEmbedding()
        self.encoder = MAEEncoder()
        self.decoder = MAEDecoder()

    def forward(self, x, mask_ratio=None):
        if mask_ratio is None: mask_ratio = self.mask_ratio
        patches = self.patch_embed(x)
        x_visible, mask, ids_restore = random_masking(patches, mask_ratio)
        latent = self.encoder(x_visible)
        pred = self.decoder(latent, ids_restore)
        return pred, mask


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = MAE().to(device)
model.load_state_dict(torch.load("mae_weights.pth", map_location=device))
model.eval()

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

def patchify(imgs):
    B, C, H, W = imgs.shape
    p = 16
    x = imgs.reshape(B, C, H // p, p, W // p, p)
    x = x.permute(0, 2, 4, 1, 3, 5).reshape(B, (H // p) * (W // p), p * p * C)
    return x

def unpatchify(x):
    B, N, _ = x.shape
    p = 16
    h = w = int(N**0.5)
    x = x.reshape(B, h, w, 3, p, p)
    x = x.permute(0, 3, 1, 4, 2, 5).reshape(B, 3, h * p, w * p)
    return x

psnr_metric = PeakSignalNoiseRatio(data_range=1.0).to(device)
ssim_metric = StructuralSimilarityIndexMeasure(data_range=1.0).to(device)

def predict(input_image, mask_ratio):
    if input_image is None: return None, None, "Upload image."
    img_tensor = transform(input_image).unsqueeze(0).to(device)
    
    with torch.no_grad():
        pred, mask = model(img_tensor, mask_ratio=mask_ratio)
    
    target = patchify(img_tensor)
    
    mean = target.mean(dim=-1, keepdim=True)
    var = target.var(dim=-1, keepdim=True)
    pred_unnorm = pred * (var + 1e-6).sqrt() + mean
    
    mask_vis = mask.unsqueeze(-1).expand_as(target)
    recon_patches = target * (1 - mask_vis) + pred_unnorm * mask_vis
    recon_imgs = unpatchify(recon_patches)
    
    norm_mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
    norm_std = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)
    orig_denorm = torch.clamp(img_tensor * norm_std + norm_mean, 0, 1)
    recon_denorm = torch.clamp(recon_imgs * norm_std + norm_mean, 0, 1)
    
    psnr = psnr_metric(recon_denorm, orig_denorm).item()
    ssim = ssim_metric(recon_denorm, orig_denorm).item()
    metrics = f"PSNR: {psnr:.2f} dB\nSSIM: {ssim:.4f}"
    
    return (
        orig_denorm[0].cpu().permute(1,2,0).numpy(), 
        recon_denorm[0].cpu().permute(1,2,0).numpy(), 
        metrics
    )

with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.Markdown("# 🧩 MAE Image Reconstruction")
    gr.Markdown("Reconstructing high-resolution structural details from masked input patches.")
    
    with gr.Row():
        with gr.Column():
            input_img = gr.Image(type="pil", label="Input Image")
            mask_slider = gr.Slider(0.1, 0.9, value=0.75, label="Masking Ratio")
            btn = gr.Button("Process Reconstruction", variant="primary")
        
        with gr.Column():
            out_orig = gr.Image(label="Processed Original")
            out_recon = gr.Image(label="MAE Reconstruction")
            out_metrics = gr.Textbox(label="Model Performance Metrics")

    btn.click(predict, inputs=[input_img, mask_slider], outputs=[out_orig, out_recon, out_metrics])

demo.launch()