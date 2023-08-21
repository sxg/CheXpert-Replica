from torch.utils.data import Dataset
from PIL import Image
import numpy as np
import os


class ChexpertDataset(Dataset):
    def __init__(self, df, img_folder, transform=None, target_transform=None):
        self.df = df
        self.img_folder = img_folder
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, i):
        img_path = os.path.join(self.img_folder, self.df.iloc[i, 0])
        img = np.array(Image.open(img_path).convert("RGB")).astype(np.float32)
        label = self.df.iloc[i, 1]
        if self.transform:
            img = self.transform(img)
        if self.target_transform:
            label = self.target_transform(label)
        return img, label
