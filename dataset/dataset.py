from omegaconf.dictconfig import DictConfig
import pytorch_lightning as pl
from torch.utils.data import DataLoader, Dataset
import pandas as pd
from dataset.utils import GenerateHeatmap
from PIL import Image
import os
import numpy as np
import hydra
from hydra.utils import get_original_cwd
import matplotlib.pyplot as plt
import torch

class KPDataset(Dataset):
    def __init__(self, istest, cfg: DictConfig, db, kypt = None):    
        self.istest = istest
        self.db = db
        self.cfg = cfg
        if not istest:
            self.kypt = kypt
        self.heatmap = GenerateHeatmap(cfg)
    

    def get_relative(self, path):
        return os.path.join(get_original_cwd(),path)

    def get_image(self, img_name):
        if self.istest:
            return np.asarray(Image.open(self.get_relative('test_imgs/')+img_name).resize((self.cfg.heatmap.W, self.cfg.heatmap.H)))
        else:
            return np.asarray(Image.open(self.get_relative('train_imgs/')+img_name).resize((self.cfg.heatmap.W, self.cfg.heatmap.H)))    
        
        
    def __getitem__(self, idx):
        img = np.copy(self.get_image(self.db[idx]))
        if self.istest:
            return torch.from_numpy(img).float(), self.db[idx]
        else:
            hmp = self.heatmap(self.kypt[idx])
            return torch.from_numpy(img).float(), torch.from_numpy(hmp).float()

    def __len__(self):
        return len(self.db)
    
    def plot_img(self, idx):
        img = self.get_image(self.db[idx])
        hms = self.heatmap(self.kypt[idx])
        hms = hms.max(axis=0)
        fig = plt.figure()
        fig.add_subplot(2, 1, 1)
        plt.imshow(hms)
        fig.add_subplot(2, 1, 2)
        plt.imshow(img)
        plt.show()
         
class KPDataModule(pl.LightningDataModule):
    def __init__(self, cfg: DictConfig):
        super(KPDataModule, self).__init__()
        self.cfg = cfg
        self.batch_size = cfg.train.batch_size
        self.num_workers = cfg.train.num_workers

        self.train_path = self.get_relative('train_imgs/')
        self.test_path = self.get_relative('test_imgs/')
        self.keypoint_path = self.get_relative('train_df.csv')
        
        self.labels = pd.read_csv(self.keypoint_path).columns
        
        self.split_len = cfg.train.valid_split
        all_list = os.listdir(self.train_path)
        all_list.sort()

        self.train_list = all_list[:self.split_len]
        self.valid_list = all_list[self.split_len:]

        self.test_list = os.listdir(self.test_path)
        self.test_list.sort()


        data = pd.read_csv(self.keypoint_path).values
        self.keypoints = [[[d[2*i+1]/4, d[2*i+2]/4] for i in range(cfg.data.num_keypoints)] for d in data]    #index 0 is name of image
        self.train_keypoints = self.keypoints[:self.split_len]
        self.valid_keypoints = self.keypoints[self.split_len:]
        self.setup()

    def get_relative(self, path):
        return os.path.join(get_original_cwd(),path)

    def prepare_data(self):
        pass

    def setup(self, stage=None):
        self.train_dataset = KPDataset(False, self.cfg, self.train_list,self.train_keypoints)
        self.val_dataset = KPDataset(False, self.cfg, self.valid_list,self.valid_keypoints)
        self.test_dataset = KPDataset(True, self.cfg, self.test_list)

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size)

@hydra.main(config_path=os.path.join('../config', 'config.yaml'), strict=False)
def main(cfg: DictConfig):
    p = KPDataModule(cfg)
    p.setup()
    p = p.train_dataset
    p.plot_img(0)
    print(len(p))

if __name__ == '__main__':
    main()