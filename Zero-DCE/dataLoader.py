import torch
from torch.utils.data import Dataset
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from PIL import Image
import numpy as np

class LowLightLoader(Dataset):
  def __init__(self, images_path, size=512):
    self.size = size
    self.images_path = Path(images_path)
    files = sorted([
      f for f in self.images_path.iterdir()
      if f.is_file() and not f.name.startswith('.')
    ])
    if not files:
      raise RuntimeError(f'No input file found in {images_path}, make sure you put your images there')

    def _load_and_resize(file_path):
      with Image.open(file_path) as img:
          img = img.convert('RGB').resize((self.size, self.size), Image.BILINEAR)
          return np.asarray(img, dtype=np.uint8).transpose((2, 0, 1))

    with ThreadPoolExecutor() as executor:
      self.cached_images = list(executor.map(_load_and_resize, files))

    print(f'Created dataset with {len(self.cached_images)} examples')

  def __len__(self):
    return len(self.cached_images)

  def __getitem__(self, idx):
    return torch.from_numpy(self.cached_images[idx]).float().div_(255.0)