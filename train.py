import os
import torch

torch.multiprocessing.set_sharing_strategy('file_system')
torch.set_float32_matmul_precision('high')

import random
import pytorch_lightning as pl
from libs.tools.config import parse_args, makepath
from libs.tools.parse import get_model, get_dataset
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor, RichProgressBar
from pytorch_lightning.loggers import WandbLogger

def main():
    cfg = parse_args()

    # set seed
    if not hasattr(cfg.train, 'seed'):
        cfg.train.seed = random.randint(0, 25536)
    pl.seed_everything(cfg.train.seed)

    model = get_model(cfg)
    datamodule = get_dataset(cfg)

    wandb_logger = WandbLogger(
        project=cfg.name,
    )

    callbacks = [
        RichProgressBar(),
        LearningRateMonitor(),
        ModelCheckpoint(
            dirpath=makepath(os.path.join(cfg.train.saved, "checkpoints"), isfile=True),
            filename="{epoch}_{val_loss:.2f}",
            verbose=True,
            save_top_k=-1,
            monitor='val_loss',
            mode='min',
            save_weights_only=True,
            every_n_epochs=100,
        ),
    ]
    
    pltrainer = pl.Trainer(
        logger=wandb_logger,
        gradient_clip_val=0.1, 
        gradient_clip_algorithm="value",
        benchmark=False,
        max_epochs=cfg.train.end_epoch,
        accelerator=cfg.train.accelerate,
        devices=cfg.train.device,
        callbacks=callbacks,
        check_val_every_n_epoch=cfg.train.val_frequency
    )
    pltrainer.fit(model, datamodule=datamodule)

if __name__ == "__main__":
    main()