from model.model import SHRModel
import hydra
import torch
import torch.distributions as dist
import os
from omegaconf import DictConfig
import numpy as np
from dataset.dataset import KPDataModule
from tqdm import tqdm
from model.loss import get_max_cor
import pandas as pd


@hydra.main(config_path=os.path.join('config', 'config.yaml'), strict=True)
def main(cfg: DictConfig):
    base_path = hydra.utils.get_original_cwd()
    PATH = os.path.join(base_path, cfg.inference.checkpoint_path)
    OUT_PATH = os.path.join(base_path, 'result.csv')
    device = 'cuda' if cfg.train.gpus > 0 else 0
    model = SHRModel.load_from_checkpoint(PATH, cfg=cfg).to(device)
    model.eval()
    dataloader = KPDataModule(cfg).test_dataloader()
    label = KPDataModule(cfg).labels

    pbar = tqdm(dataloader)
    result = pd.DataFrame(columns=label)

    for imgs, names in pbar:
        hmaps = model(imgs.to(device))[:,-1]
        cors = get_max_cor(hmaps).squeeze().detach().cpu().numpy()
        ans = []

        for i in range(len(cors[0])):
            ans.append(cors[1][i]*4)
            ans.append(cors[0][i]*4)
        hl = {label[i]:ans[i] for i in range(len(label[1:]))}
        hl['image'] = names[0]

        result=result.append(hl , ignore_index=True)
    
    result.to_csv(OUT_PATH,index=False)

    

if __name__ == '__main__':
    main()