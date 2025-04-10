import torch
from torch.fft import fft2, ifft2
from torch.nn import Module

import time


class Propagate(Module):
    def __init__(
        self,
        preceding_shape: int,
        succeeding_shape: int,
        propagation_distance: float,
        wavelength: float,
        pixel_size: float,
        device,
    ):
        super().__init__()
        self.device = device  # 记录 device
        grid_extent = (preceding_shape + succeeding_shape) / 2
        coords = torch.arange(-grid_extent + 1, grid_extent, dtype=torch.double)
        x, y = torch.meshgrid(coords * pixel_size, coords * pixel_size, indexing="ij")

        r_squared = x**2 + y**2 + propagation_distance**2
        r = torch.sqrt(r_squared)
        impulse_response = (
            (propagation_distance / r_squared * (1 / (2 * torch.pi * r) - 1.0j / wavelength))
            * torch.exp(2j * torch.pi * r / wavelength)
            * pixel_size**2
        )
        
        self.register_buffer("impulse_response_ft", fft2((impulse_response).to(self.device)), persistent=False)

    def forward(self, field: torch.Tensor) -> torch.Tensor:
        
        field = field.to(self.device)
        return conv2d_fft(self.impulse_response_ft, field)


def conv2d_fft(H_fr: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    """Performs a 2D convolution using Fast Fourier Transforms (FFT).

    Args:
        H_fr (torch.Tensor): Fourier-transformed transfer function.
        x (torch.Tensor): Input complex field.

    Returns:
        torch.Tensor: Output field after convolution.
    """
    # start_time = time.time()  # 记录起始时间
    H_fr = H_fr.to(x.device)  # 确保 H_fr 和 x 在同一设备
    output_size = (H_fr.size(-2) - x.size(-2) + 1, H_fr.size(-1) - x.size(-1) + 1)
    x_fr = fft2(x.flip(-1, -2).conj(), s=(H_fr.size(-2), H_fr.size(-1)))
    output_fr = H_fr * x_fr.conj()
    output = ifft2(output_fr)[..., : output_size[0], : output_size[1]].clone()
    # end_time = time.time()  # 记录结束时间
    # print(f"运行时间: {end_time - start_time:.6f} 秒")
    return output
