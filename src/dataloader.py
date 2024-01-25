from PIL import Image
import os
import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import numpy as np
import torchvision.transforms as t

class Source2TargetDataset(Dataset):
    def __init__(self, source_path, target_path, transform=None):
        self.source_path = source_path
        self.target_path = target_path
        self.transform = transform
        
        self.src_img, self.trg_img = [], []
        
        for p in self.source_path:
            self.src_img.extend([f'{p}/{img}' for img in os.listdir(p)])
            
        for p in self.target_path:
            self.trg_img.extend([f'{p}/{img}' for img in os.listdir(p)])
        
        self.length_dataset = max(len(self.src_img), len(self.trg_img))
        self.src_len, self.trg_len = len(self.src_img), len(self.trg_img)
        
        
    def __len__(self):
        return self.length_dataset
    
    def __getitem__(self, index):
        src_path = self.src_img[index % self.src_len]
        trg_path = self.trg_img[index % self.trg_len]
        
        src_img = Image.open(src_path).convert("RGB")
        trg_img = Image.open(trg_path).convert("RGB")
        
        if self.transform:
            src_img = self.transform(src_img)
            trg_img = self.transform(trg_img)
            
        return src_img, trg_img

def get_loaders(source_path, target_path, img_size, stats, batch_size, val_size):
    transform = t.Compose([t.Resize(img_size), 
                           t.CenterCrop(img_size), 
                           t.RandomHorizontalFlip(p=0.5),
                           t.ToTensor(),
                           t.Normalize(*stats)])
    
    
    dataset = Source2TargetDataset(source_path=source_path, 
                                   target_path=target_path, 
                                   transform=transform)
    
    
    val_size = int(len(dataset) * val_size)
    train_size = len(dataset) - val_size
    train_set, val_set = torch.utils.data.random_split(dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_set, batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_set, batch_size, shuffle=True, num_workers=2)
    
    return train_loader, val_loader