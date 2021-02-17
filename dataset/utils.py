from omegaconf.dictconfig import DictConfig
import numpy as np
import pandas as pd
import os

class GenerateHeatmap():
    def __init__(self, cfg: DictConfig):
        self.num_parts = cfg.data.num_parts
        self.sigma = cfg.heatmap.sigma
        size = 6*sigma + 3
        x = np.arange(0, size, 1, float)
        y = x[:, np.newaxis]
        x0, y0 = 3*sigma + 1, 3*sigma + 1
        self.g = np.exp(- ((x - x0) ** 2 + (y - y0) ** 2) / (2 * sigma ** 2))
        self.H = cfg.heatmap.H
        self.W = cfg.heatmap.W

        
    def __call__(self, keypoints):
        hms = np.zeros(shape = (self.num_parts, self.output_res, self.output_res), dtype = np.float32)
        sigma = self.sigma
        for p in keypoints:
            for idx, pt in enumerate(p):
                x, y = int(pt[0]), int(pt[1])
                if x<0 or y<0 or x>=self.output_res or y>=self.output_res:
                    continue
                ul = int(x - 3*sigma - 1), int(y - 3*sigma - 1)
                br = int(x + 3*sigma + 2), int(y + 3*sigma + 2)

                c,d = max(0, -ul[0]), min(br[0], self.output_res) - ul[0]
                a,b = max(0, -ul[1]), min(br[1], self.output_res) - ul[1]

                cc,dd = max(0, ul[0]), min(br[0], self.output_res)
                aa,bb = max(0, ul[1]), min(br[1], self.output_res)
                hms[idx, aa:bb,cc:dd] = np.maximum(hms[idx, aa:bb,cc:dd], self.g[a:b,c:d])
        return hms
    

if __name__ == '__main__':
    print(os.getcwd())
    df = pd.read_csv('./train_df.csv').values
    head = pd.read_csv('./train_df.csv').columns
    print(df[0])
    print(head)