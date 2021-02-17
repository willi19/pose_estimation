from omegaconf.dictconfig import DictConfig
import numpy as np
import pandas as pd
import os
import hydra
import matplotlib.pyplot as plt


class GenerateHeatmap():
    def __init__(self, cfg: DictConfig):
        self.num_parts = cfg.data.num_parts
        self.sigma = cfg.heatmap.sigma
        size = 6*self.sigma + 3
        x = np.arange(0, size, 1, float)
        y = x[:, np.newaxis]
        x0, y0 = 3*self.sigma + 1, 3*self.sigma + 1
        self.g = np.exp(- ((x - x0) ** 2 + (y - y0) ** 2) / (2 * self.sigma ** 2))
        self.H = cfg.heatmap.H
        self.W = cfg.heatmap.W

    def __call__(self, keypoints):
        hms = np.zeros(shape = (len(keypoints), self.H, self.W), dtype = np.float32)
        sigma = self.sigma
        for idx, pt in enumerate(keypoints):
            x, y = int(pt[0]), int(pt[1])
            if x<0 or y<0 or x>=self.W or y>=self.H:
                continue
            ul = int(x - 3*sigma - 1), int(y - 3*sigma - 1)
            br = int(x + 3*sigma + 2), int(y + 3*sigma + 2)

            c,d = max(0, -ul[0]), min(br[0], self.W) - ul[0]
            a,b = max(0, -ul[1]), min(br[1], self.H) - ul[1]

            cc,dd = max(0, ul[0]), min(br[0], self.W)
            aa,bb = max(0, ul[1]), min(br[1], self.H)
            hms[idx, aa:bb,cc:dd] = np.maximum(hms[idx, aa:bb,cc:dd], self.g[a:b,c:d])
        return hms


@hydra.main(config_path=os.path.join('../config', 'config.yaml'), strict=False)
def main(cfg : DictConfig):
    ghm = GenerateHeatmap(cfg)
    heatmap = ghm([[[100,200]]])
    plt.imshow(heatmap[0])
    plt.show()


if __name__ == '__main__':
    main()