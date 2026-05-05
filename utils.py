import os
import pandas as pd
import numpy as np
from pathlib import Path
from PIL import Image
import matplotlib.pyplot as plt
import torch
from torch.utils.data import Dataset

"""
Return the project root directory for notebooks and Colab.
"""
def get_project_root():
    if 'COLAB_GPU' in os.environ or 'COLAB_RELEASE_TAG' in os.environ:
        nb_dir = Path(os.getcwd())
    else:
        import IPython
        ip = IPython.get_ipython()
        nb_dir = Path(ip.run_line_magic('pwd', '')).resolve() if ip else Path('.').resolve()
    return nb_dir.parent

"""
Custom PyTorch Dataset for loading crop disease images from CSV split files.

Args:
    csv_path (Path): Path to CSV file with 'path' and 'label' columns
    transform (callable, optional): Torchvision transforms to apply to images
    subset_fraction (float): Fraction of data to use (stratified per class). Default: 1.0 (all data)
"""
class PlantVillageDataset(Dataset):
    
    def __init__(self, csv_path, transform=None, subset_fraction=1.0):
        try:
            df = pd.read_csv(csv_path)
            
            # Apply stratified subset if specified
            if subset_fraction < 1.0:
                sampled_idx = df.groupby('label', group_keys=False).apply(
                    lambda x: x.sample(frac=subset_fraction, random_state=42)
                ).index
                df = df.loc[sampled_idx].reset_index(drop=True)
            
            self.df = df
            print(f"✅ Loaded {Path(csv_path).name} with {len(self.df)} samples")
        except Exception as e:
            raise FileNotFoundError(f"Error loading {csv_path}: {e}")
        
        self.transform = transform
        self.classes = sorted(self.df['label'].unique().tolist())
        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}
    
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        try:
            row = self.df.iloc[idx]
            img_path = row['path']
            
            if not Path(img_path).exists():
                raise FileNotFoundError(f"Image not found: {img_path}")
            
            image = Image.open(img_path).convert('RGB')
            label = self.class_to_idx[row['label']]
            
            if self.transform:
                image = self.transform(image)
            
            return image, label
        except Exception as e:
            print(f"Error loading image {idx}: {e}")
            raise


"""
Denormalize and display a tensor image.

Args:
    inp (torch.Tensor): Input tensor of shape (C, H, W)
    title (str, optional): Title for the plot
    mean (list, optional): Mean values for denormalization. 
                            Defaults to ImageNet mean [0.485, 0.456, 0.406]
    std (list, optional): Std values for denormalization. 
                            Defaults to ImageNet std [0.229, 0.224, 0.225]
"""
def imshow(inp, title=None, mean=None, std=None):
    if mean is None:
        mean = [0.485, 0.456, 0.406]
    if std is None:
        std = [0.229, 0.224, 0.225]
    
    inp = inp.numpy().transpose((1, 2, 0))
    mean = np.array(mean)
    std = np.array(std)
    inp = std * inp + mean
    inp = np.clip(inp, 0, 1)
    plt.imshow(inp)
    if title:
        plt.title(title, fontsize=9)
    plt.axis('off')