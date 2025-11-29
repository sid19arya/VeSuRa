import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

class PositionalEncoding2D(nn.Module):
    def __init__(self, d_model, height, width):
        super().__init__()
        self.proj = nn.Parameter(torch.randn(1, d_model, height, width) * 0.01)

    def forward(self, x):
        return x + self.proj

class TransformerSVGNetSmall(nn.Module):
    def __init__(
        self,
        max_paths: int = 8,
        max_curves_per_path: int = 8,
        image_size: int = 128,
        d_model: int = 256,
        nhead: int = 8,
        num_encoder_layers: int = 2,
        num_decoder_layers: int = 2,
        latent_dim: int = 512,
        use_pretrained_resnet: bool = False
    ):
        super().__init__()
        self.max_paths = max_paths
        self.max_curves = max_curves_per_path
        self.total_curves = max_paths * max_curves_per_path

        # --- ResNet18 backbone ---
        resnet = models.resnet18(pretrained=use_pretrained_resnet)
        old_conv = resnet.conv1
        resnet.conv1 = nn.Conv2d(
            3,
            old_conv.out_channels,
            kernel_size=old_conv.kernel_size,
            stride=old_conv.stride,
            padding=old_conv.padding,
            bias=(old_conv.bias is not None)
        )
        self.resnet_backbone = nn.Sequential(
            resnet.conv1, resnet.bn1, resnet.relu, resnet.maxpool,
            resnet.layer1, resnet.layer2, resnet.layer3, resnet.layer4
        )

        # --- Project to transformer dimension ---
        self.feat_proj = nn.Conv2d(512, d_model, kernel_size=1)
        self.pos_enc = PositionalEncoding2D(d_model, image_size//32, image_size//32)

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead,
            dim_feedforward=d_model*2, activation='gelu',
            dropout=0.1
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=num_encoder_layers
        )

        # Transformer decoder + learned queries
        self.num_queries = self.total_curves
        self.query_embed = nn.Parameter(torch.randn(self.num_queries, d_model) * 0.1)

        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model, nhead=nhead,
            dim_feedforward=d_model*2, activation='gelu',
            dropout=0.1
        )
        self.transformer_decoder = nn.TransformerDecoder(
            decoder_layer, num_layers=num_decoder_layers
        )

        # --- SVG coordinate head ---
        # Directly predict all 8 values (4 control points × 2D each)
        self.coord_head = nn.Sequential(
          nn.Linear(d_model, d_model//2),
          nn.ReLU(),
          nn.Dropout(0.1),
          nn.Linear(d_model//2, 8)
        )

        # --- SVG mask head ---
        self.mask_head = nn.Sequential(
            nn.Linear(d_model, d_model//2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(d_model//2, 1)
        )

        # --- Raster decoder ---
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.fc_mu = nn.Linear(512, latent_dim)
        self.fc_dec = nn.Linear(latent_dim, 256*4*4)

        self.dropout_latent = nn.Dropout(0.1)


        self.raster_decoder = nn.Sequential(
            nn.ConvTranspose2d(256, 128, 4, 2, 1), nn.BatchNorm2d(128), nn.ReLU(),
            nn.ConvTranspose2d(128, 64, 4, 2, 1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.ConvTranspose2d(64, 32, 4, 2, 1), nn.BatchNorm2d(32), nn.ReLU(),
            nn.ConvTranspose2d(32, 16, 4, 2, 1), nn.BatchNorm2d(16), nn.ReLU(),
            nn.ConvTranspose2d(16, 3, 4, 2, 1), nn.Sigmoid()
        )

        self._reset_parameters()

    def _reset_parameters(self):
        nn.init.normal_(self.query_embed, std=0.02)
        for m in self.coord_head.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)
        for m in self.mask_head.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        B = x.shape[0]

        # --- Backbone ---
        feat = self.resnet_backbone(x)   # [B, 512, 4, 4]

        pooled = self.global_pool(feat).view(B, -1)
        z = self.fc_mu(pooled)
        z = self.dropout_latent(z)
        z_spatial = self.fc_dec(z).view(B, 256, 4, 4)
        raster = self.raster_decoder(z_spatial)

        # --- Transformer encoder ---
        feat_proj = self.feat_proj(feat)
        feat_pos = self.pos_enc(feat_proj)

        S = feat_pos.shape[2] * feat_pos.shape[3]
        memory = feat_pos.flatten(2).permute(2, 0, 1)    # [S, B, d_model]
        memory = self.transformer_encoder(memory)

        # --- Transformer decoder ---
        queries = self.query_embed.unsqueeze(1).repeat(1, B, 1)
        decoded = self.transformer_decoder(tgt=queries, memory=memory)
        decoded = decoded.permute(1, 0, 2)  # [B, total_curves, d_model]

        # --- Coordinate prediction ---
        coords_pred = self.coord_head(decoded)      # [B, total_curves, 8]

        coords_out = coords_pred.view(
            B,
            self.max_paths,
            self.max_curves,
            4,   # control points
            2    # (x, y)
        )

        # --- Mask prediction ---
        mask_logits = self.mask_head(decoded).squeeze(-1)
        mask_out = mask_logits.view(B, self.max_paths, self.max_curves)

        return raster, coords_out, mask_out
