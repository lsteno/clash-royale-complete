import numpy as np
import torch
import torch.distributions as td
import torch.nn as nn
from dataclasses import dataclass
from typing import List, Tuple


class ObservationEncoder(nn.Module):
    def __init__(self, depth=32, stride=2, shape=(3, 64, 64), activation=nn.ReLU):
        super().__init__()
        self.shape = shape
        self.stride = stride
        self.depth = depth
        self._conv_plan = _build_conv_plan(shape, depth, stride, kernel_size=4)
        layers: List[nn.Module] = []
        for spec in self._conv_plan:
            layers.append(
                nn.Conv2d(
                    spec.in_channels,
                    spec.out_channels,
                    spec.kernel_size,
                    stride=spec.stride,
                    padding=spec.padding,
                )
            )
            layers.append(activation())
        self.convolutions = nn.Sequential(*layers)

    def forward(self, obs):
        batch_shape = obs.shape[:-3]
        img_shape = obs.shape[-3:]
        embed = self.convolutions(obs.reshape(-1, *img_shape))
        embed = torch.reshape(embed, (*batch_shape, -1))
        return embed

    @property
    def embed_size(self):
        last_spec = self._conv_plan[-1]
        return last_spec.out_channels * int(np.prod(last_spec.output_shape))


class ObservationDecoder(nn.Module):
    def __init__(
        self, depth=32, stride=2, activation=nn.ReLU, embed_size=1024, shape=(3, 64, 64)
    ):
        super().__init__()
        self.depth = depth
        self.shape = shape

        self._conv_plan = _build_conv_plan(shape, depth, stride, kernel_size=4)
        last_shape = self._conv_plan[-1].output_shape
        conv_feature_size = 32 * depth * int(np.prod(last_shape))
        self.conv_shape = (32 * depth, *last_shape)
        self.linear = nn.Linear(embed_size, conv_feature_size)

        layers: List[nn.Module] = []
        in_channels = 32 * depth
        channel_schedule = [4 * depth, 2 * depth, 1 * depth, shape[0]]
        for spec, out_channels in zip(reversed(self._conv_plan), channel_schedule):
            output_pad = output_padding_shape(
                spec.input_shape,
                spec.output_shape,
                spec.padding,
                spec.kernel_size,
                spec.stride,
            )
            layers.append(
                nn.ConvTranspose2d(
                    in_channels,
                    out_channels,
                    spec.kernel_size,
                    stride=spec.stride,
                    padding=spec.padding,
                    output_padding=output_pad,
                )
            )
            if out_channels != shape[0]:
                layers.append(activation())
            in_channels = out_channels
        self.decoder = nn.Sequential(*layers)

    def forward(self, x):
        """
        :param x: size(*batch_shape, embed_size)
        :return: obs_dist = size(*batch_shape, *self.shape)
        """
        batch_shape = x.shape[:-1]
        embed_size = x.shape[-1]
        squeezed_size = np.prod(batch_shape).item()
        x = x.reshape(squeezed_size, embed_size)
        x = self.linear(x)
        x = torch.reshape(x, (squeezed_size, *self.conv_shape))
        x = self.decoder(x)
        mean = torch.reshape(x, (*batch_shape, *self.shape))
        obs_dist = td.Independent(td.Normal(mean, 1), len(self.shape))
        return obs_dist


def _ensure_pair(value):
    if isinstance(value, (tuple, list)):
        if len(value) != 2:
            raise ValueError("Expected pair of elements for 2D operations.")
        return (int(value[0]), int(value[1]))
    return (int(value), int(value))


def conv_out(h_in, padding, kernel_size, stride):
    return int((h_in + 2.0 * padding - (kernel_size - 1.0) - 1.0) / stride + 1.0)


def output_padding(h_in, conv_out, padding, kernel_size, stride):
    return h_in - (conv_out - 1) * stride + 2 * padding - (kernel_size - 1) - 1


def conv_out_shape(h_in, padding, kernel_size, stride):
    padding = _ensure_pair(padding)
    kernel_size = _ensure_pair(kernel_size)
    stride = _ensure_pair(stride)
    return tuple(
        conv_out(h_in[i], padding[i], kernel_size[i], stride[i]) for i in range(len(h_in))
    )


def output_padding_shape(h_in, conv_out, padding, kernel_size, stride):
    padding = _ensure_pair(padding)
    kernel_size = _ensure_pair(kernel_size)
    stride = _ensure_pair(stride)
    return tuple(
        output_padding(h_in[i], conv_out[i], padding[i], kernel_size[i], stride[i])
        for i in range(len(h_in))
    )


@dataclass(frozen=True)
class ConvLayerPlan:
    in_channels: int
    out_channels: int
    kernel_size: Tuple[int, int]
    stride: Tuple[int, int]
    padding: Tuple[int, int]
    input_shape: Tuple[int, int]
    output_shape: Tuple[int, int]


def _build_conv_plan(shape, depth, stride, kernel_size, num_layers=4):
    stride_pair = _ensure_pair(stride)
    plan: List[ConvLayerPlan] = []
    spatial = shape[1:]
    in_channels = shape[0]
    for idx in range(num_layers):
        kernels = [max(1, min(kernel_size, spatial[d])) for d in range(2)]
        strides = [stride_pair[0], stride_pair[1]]
        paddings = [0, 0]
        targets = [spatial[d] if spatial[d] <= 2 else 2 for d in range(2)]
        while True:
            out_shape = conv_out_shape(spatial, tuple(paddings), tuple(kernels), tuple(strides))
            adjusted = False
            for dim in range(2):
                target = max(1, targets[dim])
                if out_shape[dim] < target:
                    adjusted = True
                    if strides[dim] > 1:
                        strides[dim] = 1
                    elif kernels[dim] > 1:
                        kernels[dim] -= 1
                    else:
                        targets[dim] = 1
            if not adjusted:
                break
        plan.append(
            ConvLayerPlan(
                in_channels=in_channels,
                out_channels=depth * (2 ** idx),
                kernel_size=(kernels[0], kernels[1]),
                stride=(strides[0], strides[1]),
                padding=(paddings[0], paddings[1]),
                input_shape=spatial,
                output_shape=out_shape,
            )
        )
        spatial = out_shape
        in_channels = plan[-1].out_channels
    return plan
