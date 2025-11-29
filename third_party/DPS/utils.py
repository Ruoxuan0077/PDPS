import torch
import numpy as np

def dynamic_thresholding(img, s=0.95):
   scaling = torch.quantile(img.abs(), s)
   return torch.clip(img * scaling, -1., 1.)

def clear_color(x):
   if torch.is_complex(x):
       x = torch.abs(x)
   x = x.detach().cpu().squeeze().numpy()
   x = np.transpose(x, (1, 2, 0))
   x = (x - np.min(x)) / (np.max(x) - np.min(x))
   return x