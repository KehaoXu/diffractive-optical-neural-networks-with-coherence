import torch
from torch.nn import Module, Parameter


# class DOE(Module):
#     def __init__(self, shape: int):
#         super().__init__()
#         self.phase_params = Parameter(2 * torch.pi * torch.rand(shape, shape))

#     def forward(self, x):
#         return torch.exp(1j * self.phase_params) * x

class DOE(Module):
    def __init__(self, shape: int):
        super().__init__()
        self.phase_params = Parameter(2 * torch.pi * torch.rand(shape, shape))
        # self.I_sat = Parameter(torch.rand(1))
        # self.alpha_s = Parameter(torch.rand(1))
        # self.alpha_ns = Parameter(torch.rand(1))

    # def saturable_absorption(self, x: torch.Tensor) -> torch.Tensor:
    #     intensity = torch.abs(x)**2
    #     T = (1 - self.alpha_s / (1 + intensity / self.I_sat - self.alpha_ns))
    #     T = torch.clamp(T, min=1e-6, max=1.0)
    #     amplitude_modulation = torch.sqrt(T)
    #     self.last_T = amplitude_modulation.detach()
    #     return amplitude_modulation * x

    def forward(self, x):
        # return self.saturable_absorption(x) * torch.exp(1j * self.phase_params)
        return torch.exp(1j * self.phase_params) * x
