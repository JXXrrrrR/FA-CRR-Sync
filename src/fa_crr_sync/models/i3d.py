"""Checkpoint-compatible I3D feature extractor used by CRR-Sync."""

from __future__ import annotations

from pathlib import Path

import torch
from torch import nn


def get_padding_shape(
    kernel_size: tuple[int, int, int], stride: tuple[int, int, int]
) -> tuple[int, ...]:
    padding = []
    for kernel, step in zip(kernel_size, stride):
        total = max(kernel - step, 0)
        before = total // 2
        padding.extend((before, total - before))
    depth_before, depth_after = padding[:2]
    return tuple(padding[2:] + [depth_before, depth_after])


def simplify_padding(padding: tuple[int, ...]) -> tuple[bool, int]:
    first = padding[0]
    return all(value == first for value in padding), first


class Unit3Dpy(nn.Module):
    """TensorFlow-SAME compatible 3D convolution unit."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: tuple[int, int, int] = (1, 1, 1),
        stride: tuple[int, int, int] = (1, 1, 1),
        activation: str | None = "relu",
        padding: str = "SAME",
        use_bias: bool = False,
        use_bn: bool = True,
    ) -> None:
        super().__init__()
        self.padding = padding
        self.activation = activation
        self.use_bn = use_bn

        if padding == "SAME":
            padding_shape = get_padding_shape(kernel_size, stride)
            self.simplify_pad, pad_size = simplify_padding(padding_shape)
            if not self.simplify_pad:
                self.pad = nn.ConstantPad3d(padding_shape, 0)
                convolution_padding: int | tuple[int, int, int] = 0
            else:
                convolution_padding = pad_size
        elif padding == "VALID":
            self.simplify_pad = True
            convolution_padding = 0
        else:
            raise ValueError(f"Unsupported padding mode: {padding}")

        self.conv3d = nn.Conv3d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=convolution_padding,
            bias=use_bias,
        )
        if use_bn:
            self.batch3d = nn.BatchNorm3d(out_channels)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        if self.padding == "SAME" and not self.simplify_pad:
            inputs = self.pad(inputs)
        outputs = self.conv3d(inputs)
        if self.use_bn:
            outputs = self.batch3d(outputs)
        if self.activation == "relu":
            outputs = torch.relu(outputs)
        return outputs


class MaxPool3dTFPadding(nn.Module):
    def __init__(
        self,
        kernel_size: tuple[int, int, int],
        stride: tuple[int, int, int],
    ) -> None:
        super().__init__()
        self.pad = nn.ConstantPad3d(get_padding_shape(kernel_size, stride), 0)
        self.pool = nn.MaxPool3d(kernel_size, stride, ceil_mode=True)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.pool(self.pad(inputs))


class Mixed(nn.Module):
    def __init__(self, in_channels: int, out_channels: list[int]) -> None:
        super().__init__()
        self.branch_0 = Unit3Dpy(
            in_channels, out_channels[0], kernel_size=(1, 1, 1)
        )
        self.branch_1 = nn.Sequential(
            Unit3Dpy(in_channels, out_channels[1], kernel_size=(1, 1, 1)),
            Unit3Dpy(out_channels[1], out_channels[2], kernel_size=(3, 3, 3)),
        )
        self.branch_2 = nn.Sequential(
            Unit3Dpy(in_channels, out_channels[3], kernel_size=(1, 1, 1)),
            Unit3Dpy(out_channels[3], out_channels[4], kernel_size=(3, 3, 3)),
        )
        self.branch_3 = nn.Sequential(
            MaxPool3dTFPadding(kernel_size=(3, 3, 3), stride=(1, 1, 1)),
            Unit3Dpy(in_channels, out_channels[5], kernel_size=(1, 1, 1)),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return torch.cat(
            (
                self.branch_0(inputs),
                self.branch_1(inputs),
                self.branch_2(inputs),
                self.branch_3(inputs),
            ),
            dim=1,
        )


class I3D(nn.Module):
    """I3D trunk that returns the final map and pooled 1,024-D feature."""

    def __init__(
        self,
        num_classes: int = 400,
        modality: str = "rgb",
        dropout_prob: float = 0.5,
    ) -> None:
        super().__init__()
        if modality == "rgb":
            in_channels = 3
        elif modality == "mask":
            in_channels = 1
        elif modality == "flow":
            in_channels = 2
        else:
            raise ValueError(f"Unsupported modality: {modality}")

        self.num_classes = num_classes
        self.modality = modality
        self.conv3d_1a_7x7 = Unit3Dpy(
            in_channels,
            64,
            kernel_size=(7, 7, 7),
            stride=(2, 2, 2),
        )
        self.maxPool3d_2a_3x3 = MaxPool3dTFPadding(
            kernel_size=(1, 3, 3), stride=(1, 2, 2)
        )
        self.conv3d_2b_1x1 = Unit3Dpy(64, 64, kernel_size=(1, 1, 1))
        self.conv3d_2c_3x3 = Unit3Dpy(64, 192, kernel_size=(3, 3, 3))
        self.maxPool3d_3a_3x3 = MaxPool3dTFPadding(
            kernel_size=(1, 3, 3), stride=(1, 2, 2)
        )
        self.mixed_3b = Mixed(192, [64, 96, 128, 16, 32, 32])
        self.mixed_3c = Mixed(256, [128, 128, 192, 32, 96, 64])
        self.maxPool3d_4a_3x3 = MaxPool3dTFPadding(
            kernel_size=(3, 3, 3), stride=(2, 2, 2)
        )
        self.mixed_4b = Mixed(480, [192, 96, 208, 16, 48, 64])
        self.mixed_4c = Mixed(512, [160, 112, 224, 24, 64, 64])
        self.mixed_4d = Mixed(512, [128, 128, 256, 24, 64, 64])
        self.mixed_4e = Mixed(512, [112, 144, 288, 32, 64, 64])
        self.mixed_4f = Mixed(528, [256, 160, 320, 32, 128, 128])
        self.maxPool3d_5a_2x2 = MaxPool3dTFPadding(
            kernel_size=(2, 2, 2), stride=(2, 2, 2)
        )
        self.mixed_5b = Mixed(832, [256, 160, 320, 32, 128, 128])
        self.mixed_5c = Mixed(832, [384, 192, 384, 48, 128, 128])
        self.avg_pool = nn.AvgPool3d((2, 4, 4), (1, 1, 1))
        self.dropout = nn.Dropout(dropout_prob)
        self.conv3d_0c_1x1 = Unit3Dpy(
            1024,
            num_classes,
            kernel_size=(1, 1, 1),
            activation=None,
            use_bias=True,
            use_bn=False,
        )
        self.softmax = nn.Softmax(dim=1)

    def forward(self, inputs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        outputs = self.conv3d_1a_7x7(inputs)
        outputs = self.maxPool3d_2a_3x3(outputs)
        outputs = self.conv3d_2b_1x1(outputs)
        outputs = self.conv3d_2c_3x3(outputs)
        outputs = self.maxPool3d_3a_3x3(outputs)
        outputs = self.mixed_3b(outputs)
        outputs = self.mixed_3c(outputs)
        outputs = self.maxPool3d_4a_3x3(outputs)
        outputs = self.mixed_4b(outputs)
        outputs = self.mixed_4c(outputs)
        outputs = self.mixed_4d(outputs)
        outputs = self.mixed_4e(outputs)
        outputs = self.mixed_4f(outputs)
        outputs = self.maxPool3d_5a_2x2(outputs)
        outputs = self.mixed_5b(outputs)
        outputs = self.mixed_5c(outputs)
        return outputs, self.avg_pool(outputs)


class I3DBackbone(nn.Module):
    """Convert 96-frame clips into nine snippet features and maps."""

    def __init__(
        self,
        snippet_length: int = 16,
        snippet_stride: int = 10,
        snippet_count: int = 9,
        modality: str = "rgb",
    ) -> None:
        super().__init__()
        if modality not in {"rgb", "mask"}:
            raise ValueError("I3DBackbone modality must be 'rgb' or 'mask'")
        self.modality = modality
        self.backbone = I3D(
            num_classes=400, modality=modality, dropout_prob=0.5
        )
        self.snippet_length = snippet_length
        self.snippet_stride = snippet_stride
        self.snippet_count = snippet_count

    @property
    def snippet_starts(self) -> tuple[int, ...]:
        return tuple(
            index * self.snippet_stride for index in range(self.snippet_count)
        )

    def load_pretrained(self, path: Path) -> None:
        state = torch.load(path, map_location="cpu", weights_only=True)
        if self.modality == "mask":
            key = "conv3d_1a_7x7.conv3d.weight"
            if key not in state:
                raise KeyError(f"Pretrained I3D checkpoint is missing {key!r}")
            if state[key].shape[1] != 3:
                raise ValueError(
                    "Mask I3D initialization requires a three-channel RGB checkpoint"
                )
            state[key] = state[key].mean(dim=1, keepdim=True)
        self.backbone.load_state_dict(state, strict=True)

    def forward(self, videos: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if videos.ndim != 5:
            raise ValueError("I3DBackbone expects videos [B,C,T,H,W]")
        required_frames = self.snippet_starts[-1] + self.snippet_length
        if videos.shape[2] < required_frames:
            raise ValueError(
                f"Need at least {required_frames} frames, got {videos.shape[2]}"
            )
        snippets = torch.cat(
            [
                videos[:, :, start : start + self.snippet_length]
                for start in self.snippet_starts
            ],
            dim=0,
        )
        feature_maps, features = self.backbone(snippets)
        batch = videos.shape[0]
        _, channels, temporal, height, width = feature_maps.shape
        features = (
            features.reshape(self.snippet_count, batch, -1)
            .transpose(0, 1)
            .contiguous()
        )
        feature_maps = (
            feature_maps.reshape(
                self.snippet_count,
                batch,
                channels,
                temporal,
                height,
                width,
            )
            .transpose(0, 1)
            .contiguous()
        )
        return features, feature_maps
