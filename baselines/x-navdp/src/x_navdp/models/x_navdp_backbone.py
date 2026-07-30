"""Visual backbones and positional encodings used by the X-NavDP policy."""

import math
from typing import Any, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
from third_party.depth_anything.depth_anything_v2.dpt import DepthAnythingV2


class SinusoidalPosEmb(nn.Module):
    """Sinusoidal positional encoding module."""
    def __init__(self, dim: int) -> None:
        """Store the output embedding width."""
        super().__init__()
        self.dim = dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor of shape (batch_size, seq_len)
        Returns:
            Positional embeddings of shape (batch_size, seq_len, dim)
        """
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb


class LearnablePositionalEncoding(nn.Module):
    """Learnable positional encoding module."""

    def __init__(self, embed_dim: int, max_len: int = 5000) -> None:
        """Create a learnable position table up to ``max_len`` tokens."""
        super(LearnablePositionalEncoding, self).__init__()
        self.embed_dim = embed_dim
        self.max_len = max_len
        self.position_embedding = nn.Embedding(max_len, embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor of shape (batch_size, seq_len, embed_dim)

        Returns:
            Positional encodings of shape (batch_size, seq_len, embed_dim)
        """
        batch_size, seq_len, _ = x.shape
        position_ids = torch.arange(seq_len, dtype=torch.long, device=x.device)  # (seq_len,)
        position_ids = position_ids.unsqueeze(0).expand(batch_size, -1)  # (batch_size, seq_len)
        position_encoding = self.position_embedding(position_ids)  # (batch_size, seq_len, embed_dim)
        return position_encoding

class RGBDBackbone(nn.Module):
    """RGB-D backbone network for extracting and fusing RGB and depth features."""

    def __init__(
        self,
        image_size: int = 224,
        embed_size: int = 512,
        memory_size: int = 8,
        rgb_training: bool = False,
        depth_training: bool = False,
        fusion_training: bool = False,
        device: str = 'cuda:0'
    ) -> None:
        """Initialize RGB/depth encoders and transformer fusion layers."""
        super().__init__()
        self.device = device
        self.memory_size = memory_size
        self.image_size = image_size
        self.embed_size = embed_size
        self.rgb_training = rgb_training
        self.depth_training = depth_training
        self.fusion_training = fusion_training
        model_configs = {'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]}}
        self.preprocess_mean = torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32)
        self.preprocess_std = torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32)
        self.rgb_model = DepthAnythingV2(**model_configs['vits'])
        self.rgb_model = self.rgb_model.pretrained.float()
        self.depth_model = DepthAnythingV2(**model_configs['vits'])
        self.depth_model = self.depth_model.pretrained.float()
        self.former_query = LearnablePositionalEncoding(384, self.memory_size * 16)
        self.former_pe = LearnablePositionalEncoding(384, (self.memory_size + 1) * 256)
        self.former_net = nn.TransformerDecoder(
            nn.TransformerDecoderLayer(384, 8, dropout=0.0, batch_first=True), 2
        )
        self.project_layer = nn.Linear(384, embed_size)

    def _extract_rgb_feature(self, images: Any) -> torch.Tensor:
        """
        Extract RGB features from input images.

        Args:
            images: Input images of shape (B, H, W, C) or (B, T, H, W, C)

        Returns:
            RGB feature tokens of shape (B*T, 256, 384) or (B, T*256, 384)
        """
        if self.rgb_training:
            if len(images.shape) == 4:
                tensor_images = torch.as_tensor(
                    images, dtype=torch.float32, device=self.device
                ).permute(0, 3, 1, 2)
                tensor_images = tensor_images.reshape(-1, 3, self.image_size, self.image_size)
                tensor_norm_images = (
                    tensor_images - self.preprocess_mean.reshape(1, 3, 1, 1).to(self.device)
                ) / self.preprocess_std.reshape(1, 3, 1, 1).to(self.device)
                image_token = self.rgb_model.get_intermediate_layers(tensor_norm_images)[0]

            elif len(images.shape) == 5:
                tensor_images = torch.as_tensor(
                    images, dtype=torch.float32, device=self.device
                ).permute(0, 1, 4, 2, 3)
                B, T, C, H, W = tensor_images.shape
                tensor_images = tensor_images.reshape(-1, 3, self.image_size, self.image_size)
                tensor_norm_images = (
                    tensor_images - self.preprocess_mean.reshape(1, 3, 1, 1).to(self.device)
                ) / self.preprocess_std.reshape(1, 3, 1, 1).to(self.device)
                image_token = self.rgb_model.get_intermediate_layers(tensor_norm_images)[0].reshape(
                    B, T * 256, -1
                )

        else:
            with torch.no_grad():
                if len(images.shape) == 4:
                    tensor_images = torch.as_tensor(
                        images, dtype=torch.float32, device=self.device
                    ).permute(0, 3, 1, 2)
                    tensor_images = tensor_images.reshape(-1, 3, self.image_size, self.image_size)
                    tensor_norm_images = (
                        tensor_images - self.preprocess_mean.reshape(1, 3, 1, 1).to(self.device)
                    ) / self.preprocess_std.reshape(1, 3, 1, 1).to(self.device)
                    image_token = self.rgb_model.get_intermediate_layers(tensor_norm_images)[0]

                elif len(images.shape) == 5:
                    tensor_images = torch.as_tensor(
                        images, dtype=torch.float32, device=self.device
                    ).permute(0, 1, 4, 2, 3)
                    B, T, C, H, W = tensor_images.shape
                    tensor_images = tensor_images.reshape(-1, 3, self.image_size, self.image_size)
                    tensor_norm_images = (
                        tensor_images - self.preprocess_mean.reshape(1, 3, 1, 1).to(self.device)
                    ) / self.preprocess_std.reshape(1, 3, 1, 1).to(self.device)
                    image_token = self.rgb_model.get_intermediate_layers(tensor_norm_images)[0].reshape(
                        B, T * 256, -1
                    )
        return image_token

    def _extract_depth_feature(self, depths: Any) -> torch.Tensor:
        """
        Extract depth features from input depth maps.
        Args:
            depths: Input depth maps of shape (B, H, W, C) or (B, T, H, W, C)
        Returns:
            Depth feature tokens of shape (B*T, 256, 384) or (B, T*256, 384)
        """
        if self.depth_training:
            if len(depths.shape) == 4:
                tensor_depths = torch.as_tensor(
                    depths, dtype=torch.float32, device=self.device
                ).permute(0, 3, 1, 2)
                tensor_depths = tensor_depths.reshape(-1, 1, self.image_size, self.image_size)
                tensor_depths = torch.cat([tensor_depths, tensor_depths, tensor_depths], dim=1)
                depth_token = self.depth_model.get_intermediate_layers(tensor_depths)[0]
            elif len(depths.shape) == 5:
                tensor_depths = torch.as_tensor(
                    depths, dtype=torch.float32, device=self.device
                ).permute(0, 1, 4, 2, 3)
                B, T, C, H, W = tensor_depths.shape
                tensor_depths = tensor_depths.reshape(-1, 1, self.image_size, self.image_size)
                tensor_depths = torch.cat([tensor_depths, tensor_depths, tensor_depths], dim=1)
                depth_token = self.depth_model.get_intermediate_layers(tensor_depths)[0].reshape(
                    B, T * 256, -1
                )

        else:
            with torch.no_grad():
                if len(depths.shape) == 4:
                    tensor_depths = torch.as_tensor(
                        depths, dtype=torch.float32, device=self.device
                    ).permute(0, 3, 1, 2)
                    tensor_depths = tensor_depths.reshape(-1, 1, self.image_size, self.image_size)
                    tensor_depths = torch.cat([tensor_depths, tensor_depths, tensor_depths], dim=1)
                    depth_token = self.depth_model.get_intermediate_layers(tensor_depths)[0]
                elif len(depths.shape) == 5:
                    tensor_depths = torch.as_tensor(
                        depths, dtype=torch.float32, device=self.device
                    ).permute(0, 1, 4, 2, 3)
                    B, T, C, H, W = tensor_depths.shape
                    tensor_depths = tensor_depths.reshape(-1, 1, self.image_size, self.image_size)
                    tensor_depths = torch.cat([tensor_depths, tensor_depths, tensor_depths], dim=1)
                    depth_token = self.depth_model.get_intermediate_layers(tensor_depths)[0].reshape(
                        B, T * 256, -1
                    )
        return depth_token

    def _fuse_rgbd_feature(
        self,
        rgb_features: torch.Tensor,
        depth_features: torch.Tensor,
        memory_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Fuse RGB and depth features using a transformer decoder.

        Args:
            rgb_features: RGB feature tokens of shape (B, N, 384)
            depth_features: Depth feature tokens of shape (B, M, 384)
            memory_mask: Optional additive mask for fusion cross-attn, shape (L, S) or (B*nhead, L, S).

        Returns:
            Fused memory tokens of shape (B, memory_size * 16, embed_size)
        """
        if self.fusion_training:
            concatenated = torch.cat((rgb_features, depth_features), dim=1)
            former_token = concatenated + self.former_pe(concatenated)
            former_query = self.former_query(
                torch.zeros(
                    (rgb_features.shape[0], self.memory_size * 16, 384), device=self.device
                )
            )
            memory_token = self.former_net(former_query, former_token, memory_mask=memory_mask)
            memory_token = self.project_layer(memory_token)

        else:
            with torch.no_grad():
                concatenated = torch.cat((rgb_features, depth_features), dim=1)
                former_token = concatenated + self.former_pe(concatenated)
                former_query = self.former_query(
                    torch.zeros(
                        (rgb_features.shape[0], self.memory_size * 16, 384), device=self.device
                    )
                )
                memory_token = self.former_net(former_query, former_token, memory_mask=memory_mask)
                memory_token = self.project_layer(memory_token)

        return memory_token

    def forward(
        self,
        images: Any,
        depths: Any,
        memory_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Forward pass through the RGB-D backbone.

        Args:
            images: Input RGB images of shape (B, H, W, C) or (B, T, H, W, C)
            depths: Input depth maps of shape (B, H, W, C) or (B, T, H, W, C)
            memory_mask: Optional fusion cross-attention memory mask.

        Returns:
            Memory tokens of shape (B, memory_size * 16, embed_size)
        """
        rgb_token = self._extract_rgb_feature(images)
        depth_token = self._extract_depth_feature(depths)
        memory_token = self._fuse_rgbd_feature(rgb_token, depth_token, memory_mask=memory_mask)
        return memory_token


class ImageGoalBackbone(nn.Module):
    """Image goal backbone network for encoding image goals."""

    def __init__(
        self,
        image_size: int = 224,
        embed_size: int = 512,
        training: bool = True,
        device: str = 'cuda:0'
    ) -> None:
        """Initialize the image-goal encoder and projection layer."""
        super().__init__()
        self.device = device
        self.image_size = image_size
        self.embed_size = embed_size
        self.training = training
        model_configs = {'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]}}
        self.imagegoal_encoder = DepthAnythingV2(**model_configs['vits'])
        self.imagegoal_encoder = self.imagegoal_encoder.pretrained.float()
        self.imagegoal_encoder.patch_embed.proj = nn.Conv2d(
            in_channels=6,
            out_channels=self.imagegoal_encoder.patch_embed.proj.out_channels,
            kernel_size=self.imagegoal_encoder.patch_embed.proj.kernel_size,
            stride=self.imagegoal_encoder.patch_embed.proj.stride,
            padding=self.imagegoal_encoder.patch_embed.proj.padding
        )
        self.imagegoal_encoder.eval()
        self.project_layer = nn.Linear(384, embed_size)

    def forward(self, images: Any) -> torch.Tensor:
        """
        Forward pass through the image goal backbone.

        Args:
            images: Input images of shape (B, H, W, C=6)

        Returns:
            Image goal embeddings of shape (B, embed_size)
        """
        assert len(images.shape) == 4, (
            f'Error: ImageGoalBackbone receives input with shape (B,H,W,C=6), '
            f'but accept input with shape {images.shape}'
        )
        if self.training:
            tensor_images = torch.as_tensor(
                images, dtype=torch.float32, device=self.device
            ).permute(0, 3, 1, 2)
            image_token = self.imagegoal_encoder.get_intermediate_layers(tensor_images)[0].mean(dim=1)
            image_token = self.project_layer(image_token)
        else:
            with torch.no_grad():
                tensor_images = torch.as_tensor(
                    images, dtype=torch.float32, device=self.device
                ).permute(0, 3, 1, 2)
                image_token = self.imagegoal_encoder.get_intermediate_layers(tensor_images)[0].mean(dim=1)
                image_token = self.project_layer(image_token)
        return image_token


class PixelGoalBackbone(nn.Module):
    """Pixel goal backbone network for encoding pixel goals."""

    def __init__(
        self,
        image_size: int = 224,
        embed_size: int = 512,
        training: bool = False,
        device: str = 'cuda:0'
    ) -> None:
        """Initialize the pixel-goal encoder and projection layer."""
        super().__init__()
        self.device = device
        self.image_size = image_size
        self.embed_size = embed_size
        self.training = training
        model_configs = {'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]}}
        self.pixelgoal_encoder = DepthAnythingV2(**model_configs['vits'])
        self.pixelgoal_encoder = self.pixelgoal_encoder.pretrained.float()
        self.pixelgoal_encoder.patch_embed.proj = nn.Conv2d(
            in_channels=4,
            out_channels=self.pixelgoal_encoder.patch_embed.proj.out_channels,
            kernel_size=self.pixelgoal_encoder.patch_embed.proj.kernel_size,
            stride=self.pixelgoal_encoder.patch_embed.proj.stride,
            padding=self.pixelgoal_encoder.patch_embed.proj.padding
        )
        self.pixelgoal_encoder.eval()
        self.project_layer = nn.Linear(384, embed_size)

    def forward(self, images: Any) -> torch.Tensor:
        """
        Forward pass through the pixel goal backbone.

        Args:
            images: Input images of shape (B, H, W, C=4)

        Returns:
            Pixel goal embeddings of shape (B, embed_size)
        """

        assert len(images.shape) == 4, f"Expected 4D input (B, H, W, C), got shape {images.shape}"
        if self.training:
            tensor_images = torch.as_tensor(
                images, dtype=torch.float32, device=self.device
            ).permute(0, 3, 1, 2)
            image_token = self.pixelgoal_encoder.get_intermediate_layers(tensor_images)[0].mean(dim=1)
            image_token = self.project_layer(image_token)
        else:
            with torch.no_grad():
                tensor_images = torch.as_tensor(
                    images, dtype=torch.float32, device=self.device
                ).permute(0, 3, 1, 2)
                image_token = self.pixelgoal_encoder.get_intermediate_layers(tensor_images)[0].mean(dim=1)
                image_token = self.project_layer(image_token)
        return image_token


if __name__ == "__main__":
    rgbd_backbone = RGBDBackbone()
    rgbd_backbone.to('cuda:0')
    rgb = torch.rand(1,8,224,224,3)
    depth = torch.rand(1,224,224,1)
    features = rgbd_backbone(rgb,depth)
    print(features.shape)
    backbone = PixelGoalBackbone()
    backbone = backbone.to("cuda:0")
    images = torch.rand(1,224,224,4)
    print(backbone(images).shape)
    backbone = ImageGoalBackbone()
    backbone = backbone.to("cuda:0")
    images = torch.rand(1,224,224,6)
    print(backbone(images).shape)
