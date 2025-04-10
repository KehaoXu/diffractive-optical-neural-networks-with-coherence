import torch
from torch.nn import Module, Parameter


class Classifier(Module):
    def __init__(self, shape, region_size):
        super().__init__()

        self.alpha_s = (torch.tensor(0.8))
        self.alpha_ns = (torch.tensor(0.1))
        self.I_sat = Parameter((torch.tensor(1) * 1e-6))

        if shape < 4 * region_size:
            raise ValueError("shape must be at least 4*region_size")

        weight = torch.zeros(10, shape, shape, dtype=torch.double)
        row_offset = (shape - 4 * region_size) // 2
        col_offset = (shape - 3 * region_size) // 2

        # Function to set a region to 1
        def set_region(digit, row, col):
            start_row = row * (region_size) + row_offset
            start_col = col * (region_size) + col_offset
            weight[
                digit,
                start_row : start_row + region_size,
                start_col : start_col + region_size,
            ] = 1

        # Add the bottom row representing "zero" (special case)
        set_region(0, 3, 1)

        # Add the top three rows from left to right
        for digit in range(1, 10):
            row, col = (digit - 1) // 3, (digit - 1) % 3
            set_region(digit, row, col)

        self.register_buffer("weight", weight, persistent=False)

    def saturable_absorption(self, x: torch.Tensor) -> torch.Tensor:
        '''x is the intension of the field'''
        T = (1 - (self.alpha_s / (1 + x / self.I_sat)) - self.alpha_ns)
        # amplitude_modulation = torch.sqrt(T)
        return T * x

    def forward(self, x):
        # if torch.isnan(self.saturable_absorption(x)).any():
        #     print("Input tensor contains NaN values.")
        
        # print(self.I_sat)
        # print(x.mean(),x.min(),x.max())
        # print(self.saturable_absorption(x))
        # return torch.einsum("nxy,bxy->bn", self.weight, x)
        return torch.einsum("nxy,bxy->bn", self.weight, self.saturable_absorption(x))
