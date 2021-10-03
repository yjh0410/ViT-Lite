# A Lite version of VisTransformer
from einops.einops import repeat
import torch
import torch.nn as nn
from einops.layers.torch import Rearrange
from utils.modules import Transformer


class ViT(nn.Module):
    def __init__(self,
                 img_size, 
                 num_patch,
                 num_classes,
                 dim,
                 depth,
                 mlp_dims,
                 heads=8,
                 dim_head=64,
                 pool='cls',
                 channels=3,
                 dropout=0.,
                 emb_dropout=0.):
        super().__init__()
        image_h, image_w = img_size, img_size
        patch_h, patch_w = image_h // num_patch, image_w // num_patch

        assert image_h % num_patch == 0 and image_w % num_patch == 0

        num_patch = num_patch ** 2
        patch_dim = channels * patch_h * patch_w
        assert pool in {'cls', 'mean'}, 'pool type must be either cls (cls token) or mean (mean pooling)'

        self.to_patch_embedding = nn.Sequential(
            Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1=patch_h, p2=patch_w),
            nn.Linear(patch_dim, dim),
        )
        # self.to_patch_embedding = nn.Conv2d(channels, dim, kernel_size=(patch_h, patch_w), stride=(patch_h, patch_w))
        
        self.pos_embedding = nn.Parameter(torch.randn(1, num_patch + 1, dim))
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.dropout = nn.Dropout(emb_dropout)

        self.transformer = Transformer(dim=dim, 
                                       depth=depth, 
                                       heads=heads, 
                                       dim_head=dim_head, 
                                       mlp_dim=mlp_dims, 
                                       dropout=dropout)

        self.pool = pool
        self.to_latent = nn.Identity()

        self.mlp_head = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, num_classes)
        )

    def forward(self, x):
        # [B, 3, H, W] -> [B, N, dim]
        x = self.to_patch_embedding(x)
        B, N = x.shape[:2]

        cls_tokens = repeat(self.cls_token, '() n d -> b n d', b = B)
        x = torch.cat((cls_tokens, x), dim=1)
        x += self.pos_embedding[:, :(N + 1)]
        x = self.dropout(x)

        x = self.transformer(x)

        x = x.mean(1) if self.pool == 'mean' else x[:, 0]
        x = self.to_latent(x)

        return self.mlp_head(x)
