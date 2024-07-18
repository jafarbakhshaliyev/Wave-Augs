import numpy as np
import torch
import pywt
from PyEMD import EMD
from typing import List, Tuple

class Augmentation:
    def __init__(self):
        self.emd = EMD()

    @staticmethod
    def freq_mask(x: torch.Tensor, y: torch.Tensor, rate: float = 0.5, dim: int = 1) -> torch.Tensor:
        xy = torch.cat([x, y], dim=dim)
        xy_f = torch.fft.rfft(xy, dim=dim)
        mask = torch.rand_like(xy_f.real) < rate
        xy_f = xy_f * (~mask).type(xy_f.dtype)
        return torch.fft.irfft(xy_f, dim=dim)

    @staticmethod
    def freq_mix(x: torch.Tensor, y: torch.Tensor, rate: float = 0.5, dim: int = 1) -> torch.Tensor:
        xy = torch.cat([x, y], dim=dim)
        xy_f = torch.fft.rfft(xy, dim=dim)
        
        amp = xy_f.abs()
        _, index = amp.sort(dim=dim, descending=True)
        mask = (torch.rand_like(xy_f.real) < rate) & (index > 2)
        
        xy2 = torch.cat([x[torch.randperm(x.shape[0])], y[torch.randperm(y.shape[0])]], dim=dim)
        xy2_f = torch.fft.rfft(xy2, dim=dim)
        
        xy_f = torch.where(mask.unsqueeze(-1), xy2_f, xy_f)
        return torch.fft.irfft(xy_f, dim=dim)

    @staticmethod
    def wave_transform(xy: torch.Tensor, rates: List[float], wavelet: str, level: int, 
                       operation: str) -> torch.Tensor:
        result = []
        for col in range(xy.shape[-1]):
            coeffs = pywt.wavedec(xy[:, :, col], wavelet=wavelet, mode='symmetric', level=level)
            modified_coeffs = []
            for i, coeff in enumerate(coeffs):
                if operation == 'mask':
                    mask = torch.rand(coeff.shape) < rates[i]
                    modified_coeff = torch.from_numpy(coeff) * (~mask)
                elif operation == 'mix':
                    mask = torch.rand(coeff.shape) < rates[i]
                    coeff2 = torch.from_numpy(np.random.permutation(coeff))
                    modified_coeff = torch.where(mask, coeff2, torch.from_numpy(coeff))
                modified_coeffs.append(modified_coeff.numpy())
            reconstructed = pywt.waverec(modified_coeffs, wavelet=wavelet, mode='symmetric')
            result.append(torch.from_numpy(reconstructed[:, None]))
        return torch.cat(result, dim=-1)

    def wave_mask(self, x: torch.Tensor, y: torch.Tensor, rates: List[float], 
                  wavelet: str = 'db1', level: int = 2, dim: int = 1) -> torch.Tensor:
        xy = torch.cat([x, y], dim=1)
        return self.wave_transform(xy, rates, wavelet, level, 'mask')

    def wave_mix(self, x: torch.Tensor, y: torch.Tensor, rates: List[float], 
                 wavelet: str = 'db1', level: int = 2, dim: int = 1) -> torch.Tensor:
        xy = torch.cat([x, y], dim=1)
        return self.wave_transform(xy, rates, wavelet, level, 'mix')

    def emd_aug(self, x: torch.Tensor) -> torch.Tensor:
        b, n_imf, t, c = x.size()
        inp = x.permute(0, 2, 1, 3).reshape(b, t, n_imf * c)
        w = 2 * torch.rand((b, 1, n_imf * c)).to(x.device) if torch.rand(1) >= 0.5 else torch.ones((b, 1, n_imf * c)).to(x.device)
        out = (w.expand(-1, t, -1) * inp).reshape(b, t, n_imf, c).sum(dim=2)
        return out

    @staticmethod
    def mix_aug(batch_x: np.ndarray, batch_y: np.ndarray, lambd: float = 0.5) -> Tuple[np.ndarray, np.ndarray]:
        inds2 = np.random.permutation(len(batch_x))
        lam = np.random.beta(lambd, lambd)
        batch_x_mixed = lam * batch_x[inds2] + (1 - lam) * batch_x
        batch_y_mixed = lam * batch_y[inds2] + (1 - lam) * batch_y
        return batch_x_mixed, batch_y_mixed