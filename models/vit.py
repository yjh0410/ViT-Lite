# A Lite version of VisTransformer
import torch
import torch.nn as nn
from .transformer import TransformerEncoder


class ViT(nn.Module):
    def __init__(self,
                 img_size, 
                 num_patch,
                 num_classes,
                 hidden_dim=256,
                 num_encoders=6,
                 num_heads=8,
                 mlp_dim=2048,
                 dropout=0.):
        super().__init__()
        image_h, image_w = img_size, img_size
        patch_h, patch_w = image_h // num_patch, image_w // num_patch

        assert image_h % num_patch == 0 and image_w % num_patch == 0

        N = num_patch ** 2
        # image to patch
        self.patch_embedding = nn.Conv2d(3, hidden_dim, kernel_size=(patch_h, patch_w), stride=(patch_h, patch_w))
        # position embedding
        self.pos_embedding = nn.Parameter(torch.randn(1, N, hidden_dim))
        # transformer
        self.transformer = TransformerEncoder(dim=hidden_dim, 
                                              depth=num_encoders,
                                              heads=num_heads,
                                              dim_head=hidden_dim // num_heads, 
                                              mlp_dim=mlp_dim, 
                                              dropout=dropout)
        # output
        self.mlp_head = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, num_classes)
        )

    def forward(self, x):
        # [B, 3, H, W] -> [B, N, dim]
        x = self.patch_embedding(x)
        B, C = x.shape[:2]

        x = x.view(B, C, -1).permute(0, 2, 1).contiguous()
        # transformer
        x = x + self.pos_embedding
        print(x.size())
        x = self.transformer(x)
        print(x.size())
        # classify
        x = x.mean(1)
        x = self.mlp_head(x)
        return x
