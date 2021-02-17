import os
import platform
import hydra
from omegaconf import DictConfig
from pytorch_lightning import Trainer, callbacks
from pytorch_lightning import loggers as pl_loggers

from model.model import SHRModel
from dataset.dataset import KPDataModule

@hydra.main(config_path=os.path.join('config', 'config.yaml'), strict=False)
def main(cfg: DictConfig):
    model = SHRModel(cfg)
    data = KPDataModule(cfg)
    logger = pl_loggers.TensorBoardLogger(save_dir=cfg.train.log_dir, version=cfg.train.version)
    checkpoint_callback = callbacks.ModelCheckpoint(
        monitor='val_loss',
        dirpath=cfg.train.checkpoint_dir,
        save_top_k=cfg.train.save_top_k
    )
    trainer = Trainer(
        accelerator=None,
        accumulate_grad_batches=cfg.train.accumulate,
        auto_scale_batch_size=True,
        callbacks=[checkpoint_callback],
        default_root_dir=cfg.train.log_dir,
        fast_dev_run=cfg.train.fast_dev_run,
        gpus=cfg.train.gpus,
        logger=logger,
        resume_from_checkpoint=None if cfg.train.resume is '' else cfg.train.resume,
        terminate_on_nan=True,
        weights_save_path=cfg.train.checkpoint_dir
    )

    trainer.fit(model, datamodule=data)

if __name__ == '__main__':
    main()
