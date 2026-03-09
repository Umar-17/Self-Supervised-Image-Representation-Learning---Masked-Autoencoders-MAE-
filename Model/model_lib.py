import torch
import torch.nn as nn

class PatchEmbed(nn.Module):
    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768):
        super().__init__()
        self.num_patches = (img_size // patch_size) ** 2
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        x = self.proj(x).flatten(2).transpose(1, 2)
        return x

class MaskedAutoencoder(nn.Module):
    def __init__(self):
        super().__init__()
        
   
        self.patch_embed = PatchEmbed(img_size=224, patch_size=16, embed_dim=768)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, 768))
        self.pos_embed = nn.Parameter(torch.zeros(1, self.patch_embed.num_patches + 1, 768))
        
        encoder_layer = nn.TransformerEncoderLayer(d_model=768, nhead=12, dim_feedforward=3072, activation='gelu', batch_first=True)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=12) 
        
        
        self.decoder_embed = nn.Linear(768, 384) 
        self.mask_token = nn.Parameter(torch.zeros(1, 1, 384)) 
        self.decoder_pos_embed = nn.Parameter(torch.zeros(1, self.patch_embed.num_patches + 1, 384))
        
        decoder_layer = nn.TransformerEncoderLayer(d_model=384, nhead=6, dim_feedforward=1536, activation='gelu', batch_first=True)
        self.decoder = nn.TransformerEncoder(decoder_layer, num_layers=12)
        
        
        self.decoder_pred = nn.Linear(384, 16**2 * 3) 

    def random_masking(self, x, mask_ratio=0.75):
        B, N, D = x.shape
        len_keep = int(N * (1 - mask_ratio))
        
        noise = torch.rand(B, N, device=x.device)
        ids_shuffle = torch.argsort(noise, dim=1)
        ids_restore = torch.argsort(ids_shuffle, dim=1)
        
        ids_keep = ids_shuffle[:, :len_keep]
        x_masked = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D))
        
        mask = torch.ones([B, N], device=x.device)
        mask[:, :len_keep] = 0
        mask = torch.gather(mask, dim=1, index=ids_restore)
        
        return x_masked, mask, ids_restore

    def forward_encoder(self, x, mask_ratio):
        x = self.patch_embed(x)
        x = x + self.pos_embed[:, 1:, :]
        
        x, mask, ids_restore = self.random_masking(x, mask_ratio)
        
        cls_token = self.cls_token + self.pos_embed[:, :1, :]
        cls_tokens = cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        
        x = self.encoder(x)
        return x, mask, ids_restore

    def forward_decoder(self, x, ids_restore):
        x = self.decoder_embed(x)
        
        mask_tokens = self.mask_token.repeat(x.shape[0], ids_restore.shape[1] + 1 - x.shape[1], 1)
        x_ = torch.cat([x[:, 1:, :], mask_tokens], dim=1)
        x_ = torch.gather(x_, dim=1, index=ids_restore.unsqueeze(-1).repeat(1, 1, x.shape[2]))
        
        x = torch.cat([x[:, :1, :], x_], dim=1)
        x = x + self.decoder_pos_embed
        x = self.decoder(x)
        
        x = self.decoder_pred(x)
        return x
