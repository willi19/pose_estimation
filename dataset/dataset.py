from omegaconf.dictconfig import DictConfig
import pytorch_lightning as pl
from torch.utils.data import DataLoader, Dataset
import pandas as pd
from utils import GenerateHeatmap
from PIL import Image
import os
import numpy as np

class PoseDataset(Dataset):
    def __init__(self, istest):
        self.train_path = './train_imgs/'
        self.test_path = './test_imgs/'
        self.keypoint_path = './train_df.csv'
        self.train_list = os.listdir(self.train_path)
        self.test_list = os.listdir(self.test_path)
        self.keypoints = pd.read_csv(self.keypoint_path).values     #index 0 is name of image
        self.labels = pd.read_csv(self.keypoint_path).columns
        self.istest = istest
        
    
    def get_image(self, img_name):
        if self.istest:
            return np.asarray(Image.open(self.test_path+img_name))
        else:
            return np.asarray(Image.open(self.train_path+img_name))    
        
    #def get_hitmap(self,)        
        
    def __getitem__(self, idx):
        if self.istest:
            return self.get_image(self.test_list[idx])
        else:
            return self.get_image(self.train_list[idx]), self.keypoints[idx]
        
         
class DataModule(pl.LightningDataModule):
    def __init__(self, cfg: DictConfig):
        super(BaseDataModule, self).__init__()
        self.cfg = cfg
        self.batch_size = cfg.train.batch_size

    def prepare_data(self):
        pass

    def setup(self, stage=None):
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size)

if __name__ == '__main__':
    p = PoseDataset(False)
    print(p[0])
    p = PoseDataset(True)
    print(p[1].shape)