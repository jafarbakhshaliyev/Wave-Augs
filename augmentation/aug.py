# =============================================================================
# The freq_mask, freq_mix methods are adapted from the following sources:
#  Chen, M., Xu, Z., Zeng, A., & Xu, Q. (2023). "FrAug: Frequency Domain Augmentation for Time Series Forecasting".
#  arXiv preprint arXiv:2302.09292.
#
# The emd_aug and mix_aug for STAug method are adapted from the following source:
#  https://github.com/xiyuanzh/STAug/tree/main
# =============================================================================

import numpy as np
import torch
import pywt
from pytorch_wavelets import DWT1DForward, DWT1DInverse
from typing import List, Tuple

class augmentation():

  """
    A class for data augmentation techniques used for Time Series Forecasting.

    Attributes:
    None

    Methods:
    - freq_mask: Apply frequency masking to input data.
    - freq_mix: Mix two input signals in the frequency domain.
    - wave_mask: Apply wavelet-based masking to input data.
    - wave_mix: Mix two input signals using wavelet transformation.
    - emd_aug: Apply empirical mode decomposition (EMD) based augmentation.
    - mix_aug: Mix two batches of data with a random interpolation factor.
    """

  def __init__(self):
        pass

  @staticmethod
  def freq_mask(x: torch.Tensor, y: torch.Tensor, rate: float = 0.5, dim: int = 1) -> torch.Tensor:
    """
        Apply frequency masking to input data.

        Args:
        - x (torch.Tensor): Look-back window.
        - y (torch.Tensor): Target horizon.
        - rate (float): Mask rate.
        - dim (int): Dimension along to concatenate and apply Fourier Transform.

        Returns:
        - torch.Tensor: Masked synthetic data tensor.
    """

    xy = torch.cat([x,y],dim=1)
    xy_f = torch.fft.rfft(xy,dim=dim)
    m = torch.FloatTensor(xy_f.shape).uniform_() < rate
    freal = xy_f.real.masked_fill(m,0)
    fimag = xy_f.imag.masked_fill(m,0)
    xy_f = torch.complex(freal,fimag)
    xy = torch.fft.irfft(xy_f,dim=dim)
    return xy
  
  @staticmethod
  def freq_mix(x: torch.Tensor, y: torch.Tensor, rate: float = 0.5, dim: int = 1) -> torch.Tensor:
    """
        Mix two input signals in the frequency domain.

        Args:
        - x (torch.Tensor): Look-back window.
        - y (torch.Tensor): Target horizon.
        - rate (float): Mix rate.
        - dim (int): Dimension along to concatenate and apply Fourier Transform.

        Returns:
        - torch.Tensor: Mixed synthetic data tensor.
    """

    xy = torch.cat([x,y],dim=dim)
    xy_f = torch.fft.rfft(xy,dim=dim)

    m = torch.FloatTensor(xy_f.shape).uniform_() < rate
    amp = abs(xy_f)
    _,index = amp.sort(dim=dim, descending=True)
    dominant_mask = index > 2
    m = torch.bitwise_and(m, dominant_mask)
    freal = xy_f.real.masked_fill(m,0)
    fimag = xy_f.imag.masked_fill(m,0)

    b_idx = np.arange(x.shape[0])
    np.random.shuffle(b_idx)
    x2, y2 = x[b_idx], y[b_idx]
    xy2 = torch.cat([x2,y2],dim=dim)
    xy2_f = torch.fft.rfft(xy2,dim=dim)

    m = torch.bitwise_not(m)
    freal2 = xy2_f.real.masked_fill(m,0)
    fimag2 = xy2_f.imag.masked_fill(m,0)

    freal += freal2
    fimag += fimag2

    xy_f = torch.complex(freal,fimag)

    xy = torch.fft.irfft(xy_f,dim=dim)

    return xy
  
  
  @staticmethod
  def wave_mask(x: torch.Tensor, y: torch.Tensor, rates: List[float], wavelet: str = 'db1', level: int = 2, dim: int= 1) -> torch.Tensor:
      """
          Apply wavelet-based masking to input data.

          Args:
          - x (torch.Tensor): Look-back window.
          - y (torch.Tensor): Target horizon.
          - rates (list of floats): List of mask rates for each wavelet level.
          - wavelet (str): Type of wavelet to use.
          - level (int): Number of decomposition levels.
          - dim (int): Dimension along which to concatenate and apply DWT.

          Returns:
          - torch.Tensor: Masked synthetic data tensor.
      """

      xy = torch.cat([x, y], dim=1)  # Concatenate along the time dimension
      rate_tensor = torch.tensor(rates, device=x.device) # Convert rates to tensor

      # Permute dimensions to match expected input: (batch_size, num_features, seq_len)
      xy = xy.permute(0, 2, 1).to(x.device)

      # Initialize & perform the DWT
      dwt = DWT1DForward(J=level, wave=wavelet, mode='symmetric').to(x.device)
      cA, cDs = dwt(xy)

      mask = torch.rand_like(cA).to(cA.device) < rate_tensor[0]
      cA = cA.masked_fill(mask, 0)

      # Apply masking to detail coefficients
      masked_cDs = []
      for i, cD in enumerate(cDs, 1):
          mask_cD = torch.rand_like(cD).to(cD.device) < rate_tensor[i]  # Create mask
          cD = cD.masked_fill(mask_cD, 0)
          masked_cDs.append(cD)

      # Initialize the inverse DWT & reconstruct the signal
      idwt = DWT1DInverse(wave=wavelet, mode='symmetric').to(x.device)
      reconstructed = idwt((cA, masked_cDs))

      # Permute back to original shape: (batch_size, seq_len, num_features)
      reconstructed = reconstructed.permute(0, 2, 1)

      return reconstructed
  
  @staticmethod
  def wave_mix(x: torch.Tensor, y: torch.Tensor, rates: List[float], wavelet: str = 'db1', level: int = 2, dim: int = 1) -> torch.Tensor:
      """
          Mix two input signals using wavelet transformation.

          Args:
          - x (torch.Tensor): Look-back window.
          - y (torch.Tensor): Target horizon.
          - rates (list of floats): List of mix rates for each wavelet level.
          - wavelet (str): Type of wavelet to use.
          - level (int): Number of decomposition levels.
          - dim (int): Dimension along which to concatenate and apply DWT.

          Returns:
          - torch.Tensor: Mixed synthetic data tensor.
      """

      xy = torch.cat([x, y], dim=1)  # Concatenate along the time dimension
      batch_size, _, _ = xy.shape
      rate_tensor = torch.tensor(rates, device=x.device) # Convert rates to tensor

      # Permute dimensions to match expected input: (batch_size, num_features, seq_len)
      xy = xy.permute(0, 2, 1).to(x.device)

      # Shuffle the batch for mixing
      b_idx = torch.randperm(batch_size)
      xy2 = xy[b_idx]

      # Initialize & perform the DWT on the both signals
      dwt = DWT1DForward(J=level, wave=wavelet, mode='symmetric').to(x.device)
      cA1, cDs1 = dwt(xy)
      cA2, cDs2 = dwt(xy2)

      # Mix the approximation coefficients
      mask = torch.rand_like(cA1).to(cA1.device) < rate_tensor[0] # Create mask
      cA_mixed = cA1.masked_fill(mask, 0) + cA2.masked_fill(~mask, 0)

      # Mix the coefficients
      mixed_cDs = []
      for i, (cD1, cD2) in enumerate(zip(cDs1, cDs2), 1):
          mask = torch.rand_like(cD1).to(cD1.device) < rate_tensor[i] # Create mask
          cD_mixed = cD1.masked_fill(mask, 0) + cD2.masked_fill(~mask, 0)
          mixed_cDs.append(cD_mixed)

      # Initialize the inverse DWT & reconstruct the signal
      idwt = DWT1DInverse(wave=wavelet, mode='symmetric').to(x.device)
      reconstructed = idwt((cA_mixed, mixed_cDs))

      # Permute back to original shape: (batch_size, seq_len, num_features)
      reconstructed = reconstructed.permute(0, 2, 1)

      return reconstructed

  # StAug: frequency-domain augmentation 
  def emd_aug(self, x: torch.Tensor) -> torch.Tensor:
      """
        Apply augmentation on empirical mode decomposition (EMD).

        Args:
        - x (torch.Tensor): Input tensor.

        Returns:
        - torch.Tensor: Augmented tensor.
      """

      b,n_imf,t,c = x.size()
      inp = x.permute(0,2,1,3).reshape(b,t,n_imf*c) #b,t,n_imf,c -> b,t,n_imf*c
      if(torch.rand(1) >= 0.5):
          w = 2 * torch.rand((b,1,n_imf*c)).cuda()
      else:
          w = torch.ones((b,1,n_imf*c)).cuda()
      w_exp = w.expand(-1,t,-1) #b,t,n_imf*c
      out = w_exp * inp
      out = out.reshape(b,t,n_imf,c).sum(dim=2) #b,t,c
      
      return out

# StAug: time-domain augmentation
  def mix_aug(self, batch_x: np.ndarray, batch_y: np.ndarray, lambd: float = 0.5) -> Tuple[np.ndarray, np.ndarray]:
      """
        Mix two batches of data with a random interpolation factor.

        Args:
        - batch_x (numpy.ndarray): Input batch 1.
        - batch_y (numpy.ndarray): Input batch 2.
        - lambd (float): Beta distribution parameter for interpolation.

        Returns:
        - numpy.ndarray: Mixed augmented batches.
      """

      inds2 = np.random.permutation(len(batch_x))
      lam = np.random.beta(lambd, lambd)
      batch_x = lam * batch_x[inds2] + (1-lam) * batch_x
      batch_y = lam * batch_y[inds2] + (1-lam) * batch_y

      return batch_x, batch_y
