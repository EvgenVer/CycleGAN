from PIL import Image
import os
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torchvision.transforms as t
from torchvision.io import read_image

class Source2TargetDataset(Dataset):
    def __init__(self, source_path, target_path, transform=None, set_size=None):
        self.source_path = source_path
        self.target_path = target_path
        self.transform = transform
        
        self.src_img, self.trg_img = [], []
        
        for p in self.source_path:
            self.src_img.extend([f'{p}/{img}' for img in os.listdir(p)])
            self.src_img = self.src_img[:set_size]
            
        for p in self.target_path:
            self.trg_img.extend([f'{p}/{img}' for img in os.listdir(p)])
            self.trg_img = self.trg_img[:set_size]
        
        self.length_dataset = max(len(self.src_img), len(self.trg_img))
        self.src_len, self.trg_len = len(self.src_img), len(self.trg_img)
        
        
    def __len__(self):
        return self.length_dataset
    
    def __getitem__(self, index):
        src_path = self.src_img[index % self.src_len]
        trg_path = self.trg_img[index % self.trg_len]
        
        src_img = read_image(src_path) / 255
        trg_img = read_image(trg_path) / 255
        
        if self.transform:
            src_img = self.transform(src_img)
            trg_img = self.transform(trg_img)
            
        return src_img, trg_img      
        
        
def get_loader(source_path, target_path, img_size, stats, batch_size, set_size=None, stage='train'):
    
    if stage == 'train':
        transform = t.Compose([t.Resize((img_size+30, img_size+30)), 
                               t.RandomCrop((img_size, img_size)), 
                               t.RandomHorizontalFlip(p=0.5), 
                               t.Normalize(*stats)])
    elif stage == 'test':
        transform = t.Compose([t.Resize((img_size, img_size)), 
                               t.CenterCrop((img_size, img_size)),  
                               t.Normalize(*stats)])
    
    
    dataset = Source2TargetDataset(source_path=source_path, 
                                   target_path=target_path, 
                                   transform=transform, 
                                   set_size=set_size)
    
    loader = DataLoader(dataset, batch_size, shuffle=True, num_workers=3)
    
    return loader