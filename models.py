import torch
import torch.nn as nn
import torch.nn.functional as F


class ResidualBlock(nn.Module):
    def __init__(self, channels, kernel_size=3, dilation=1):
        super(ResidualBlock, self).__init__()
        padding = (kernel_size // 2) * dilation
        self.conv1 = nn.Conv2d(
            channels,
            channels,
            kernel_size,
            padding=padding,
            dilation=dilation,
            bias=False,
        )
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(
            channels,
            channels,
            kernel_size,
            padding=padding,
            dilation=dilation,
            bias=False,
        )
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x):
        residual = x
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.bn2(self.conv2(x))
        return F.relu(x + residual)


class FluidNet(nn.Module):
    def __init__(
        self, input_channels=1, output_channels=1, hidden_channels=64, num_res_blocks=6
    ):
        super(FluidNet, self).__init__()

        self.input_layer = nn.Conv2d(
            input_channels, hidden_channels, kernel_size=3, padding=1
        )

        self.res_blocks = nn.Sequential(
            *[ResidualBlock(hidden_channels) for _ in range(num_res_blocks)]
        )

        self.dilated_conv = nn.Conv2d(
            hidden_channels, hidden_channels, kernel_size=3, padding=2, dilation=2
        )
        self.output_channels = output_channels
        self.output_layer = nn.Conv2d(
            hidden_channels, output_channels, kernel_size=3, padding=1
        )

    def forward(self, x):
        batch, channels, height, width = x.shape
        x = F.relu(self.input_layer(x))
        x = self.res_blocks(x)
        x = F.relu(self.dilated_conv(x))
        x = self.output_layer(x)
        if self.output_channels == 1:
            x = x.squeeze(1)
        return x

    def simulate(self, x, num_timesteps):
        output = torch.zeros(
            (num_timesteps, self.output_channels, x.shape[-2], x.shape[-1]),
            dtype=torch.float32,
            device=x.device,
        )
        x_im1 = x.unsqueeze(0)
        output[0] = x_im1[:, -self.output_channels :]
        for i in range(1, num_timesteps):
            out_i = self.forward(x_im1)
            output[i] = out_i[0]
            x_im1 = x_im1.clone()
            x_im1[:, -self.output_channels :] = out_i

        if self.output_channels:
            output = output.squeeze(1)

        return output

    def curl(self, ux, uy):
        # Get the shape of the tensors
        shape = ux.shape

        # Shift uy in the x-direction (left and right)
        uy_left = torch.cat([uy[:, 1:], uy[:, :1]], dim=1)
        uy_right = torch.cat([uy[:, -1:], uy[:, :-1]], dim=1)

        # Shift ux in the y-direction (up and down)
        ux_up = torch.cat([ux[1:, :], ux[:1, :]], dim=0)
        ux_down = torch.cat([ux[-1:, :], ux[:-1, :]], dim=0)

        # Compute the curl
        return uy_left - uy_right - ux_up + ux_down
