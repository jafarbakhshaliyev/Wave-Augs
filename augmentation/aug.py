import numpy as np
import torch
import pywt
from torch.distributions import uniform, beta
from decompositions.decomposition import emd_augment

class augmentation():

  def __init__(self):
        pass

  def add_noise(self, x, rate = 0.5, noise_level = 0.1):
    m = torch.FloatTensor(x.shape).uniform_() < rate
    noise = torch.randn(x.shape)
    noise_applied = x + noise_level * noise
    return torch.where(m, noise_applied, x)

  def freq_mask(self, x, y, rate=0.5, dim=1):
    xy = torch.cat([x,y],dim=1)
    xy_f = torch.fft.rfft(xy,dim=dim)
    m = torch.FloatTensor(xy_f.shape).uniform_() < rate
    freal = xy_f.real.masked_fill(m,0)
    fimag = xy_f.imag.masked_fill(m,0)
    xy_f = torch.complex(freal,fimag)
    xy = torch.fft.irfft(xy_f,dim=dim)
    return xy

  def freq_mix(self, x, y, rate=0.5, dim=1):

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

  def wave_mask(self, x, y, rates, wavelet = 'db1', level = 2, dim = 1):
    xy = torch.cat([x,y],dim=1)
    s_list = []
    for col in range(xy.shape[-1]):
      coeffs = pywt.wavedec(xy[:,:, col], wavelet = wavelet, mode='symmetric', level=level)
      S = []
      for i in range(level + 1):
        coeffs_tensor = torch.FloatTensor(coeffs[i])  
        m = coeffs_tensor.uniform_() < rates[i]
        
        C = coeffs_tensor.masked_fill(m, 0)
        S.append(C.numpy())
      s = pywt.waverec(S, wavelet = wavelet, mode='symmetric')
      s_list.append(torch.from_numpy(s[:, :, None]))
    return torch.cat(s_list, dim=-1)

  def wave_mix(self, x, y, rates, wavelet = 'db1', level = 2, dim = 1):
      xy = torch.cat([x,y], dim = 1)
      b_idx = np.arange(x.shape[0])
      np.random.shuffle(b_idx)
      x2, y2 = x[b_idx], y[b_idx]
      xy2 = torch.cat([x2,y2],dim=dim)
      s_list = []
      for col in range(xy.shape[-1]):
        S = []
        coeffs_1 = pywt.wavedec(xy[:,:, col], wavelet = wavelet, mode='symmetric', level=level)
        coeffs_2 = pywt.wavedec(xy2[:,:, col], wavelet = wavelet, mode='symmetric', level=level)
        for i in range(level + 1):
          coeffs_tensor_1 = torch.FloatTensor(coeffs_1[i]) 
          coeffs_tensor_2 = torch.FloatTensor(coeffs_2[i])  
          m1 = coeffs_tensor_1.uniform_() < rates[i]
          m2 = torch.bitwise_not(m1)
          C1 = coeffs_tensor_1.masked_fill(m1, 0)
          C2 = coeffs_tensor_2.masked_fill(m2, 0)
          C = C1 + C2
          S.append(C.numpy())
        s = pywt.waverec(S, wavelet = wavelet, mode='symmetric')
        s_list.append(torch.from_numpy(s[:, :, None]))
      return torch.cat(s_list, dim=-1)

  def wave_mixup_wb(self, x, y, alphas, wavelet = 'db1', level = 2, dim = 1):

    xy = torch.cat([x,y], dim = 1)
    b_idx = np.arange(x.shape[0])
    np.random.shuffle(b_idx)
    x2 = x[b_idx]
    y2 = y[b_idx]
    xy2 = torch.cat([x2, y2], dim=dim)
    s_list = []
    for col in range(xy.shape[-1]):
      S = []
      coeffs_1 = pywt.wavedec(xy[:,:, col], wavelet = wavelet, mode='symmetric', level=level)
      coeffs_2 = pywt.wavedec(xy2[:,:, col], wavelet = wavelet, mode='symmetric', level=level)
      for i in range(level + 1):
        alpha = alphas[i]
        C = (1-alpha)*coeffs_1[i] + alpha*coeffs_2[i]
        S.append(C)
      s = pywt.waverec(S, wavelet = wavelet, mode='symmetric')
      s_list.append(torch.from_numpy(s[:, :, None]))
    return torch.cat(s_list, dim=-1)

  # StAug: frequency-domain augmentation 
  def emd_aug(self, x):

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
  def mix_aug(self, batch_x, batch_y, lambd = 0.5):

      inds2 = np.random.permutation(len(batch_x))
      lam = np.random.beta(lambd, lambd)
      batch_x = lam * batch_x[inds2] + (1-lam) * batch_x
      batch_y = lam * batch_y[inds2] + (1-lam) * batch_y

      return batch_x, batch_y
